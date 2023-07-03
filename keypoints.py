import os
import shutil
import tempfile
import typing as tp
from enum import Enum
from pathlib import Path

import librosa
import ffmpeg
import numpy as np
from dejavu import Dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer
from mysql.connector import connect, Error

from config import Config as cfg

class IntervalType(Enum):
    UNKNOWN = 0
    OPENING = 1
    ANNOUNCEMENT = 2
    MUSICAL_BREAK = 3
    ADVERTISEMENT = 4
    WEATHER_START = 5
    NEWS = 6
    CLOSING = 7

interval = tp.NewType('interval', tuple[IntervalType, tuple[float, float]])
    
class ProgramType(Enum):
    RUSSIA1_VESTI_PRIVOLZHIE = 'russia1/vesti_privolzhie'
    RUSSIA1_SOBITIYA_NEDELI = 'russia1/sobitiya_nedeli'

    RUSSIA24_VESTI_PRIVOLZHIE = 'russia24/vesti_privolzhie'
    RUSSIA24_VESTI_PFO = 'russia24/vesti_pfo'
    RUSSIA24_VESTI_NIZHNY_NOVGOROD = 'russia24/vesti_nizhny_novgorod'

    VOLGANN_NOVOSTI = 'volgann/novosti'
    VOLGANN_POSLESLOVIE_SOBITIYA_NEDELI = 'volgann/posleslovie_sobitiya_nedeli'

    NNTV_VREMYA_NOVOSTEY = 'nntv/vremya_novostei'
    
    DZERZHTV_NOVOSTI = 'dzerzhtv/novosti'

class KeyPoints:
    def __init__(self, config: map = cfg.djv_config, data: Path = None) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.data = Path(data if data is not None else cfg.data_dir)
        self.config = config
        self.cnx = connect(
            host=config["database"]["host"],
            user=config["database"]["user"],
            password=config["database"]["password"]
        )
        self.cursor = self.cnx.cursor()
        
        try:
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config['database']['database']}")
        except Error as e:
            print("Error while working with MySQL", e)
            if self.cnx.is_connected():
                self.cursor.close()
                self.cnx.close()
                print("MySQL connection is closed")
                return
            
        self.djv = Dejavu(config)
        
    def __del__(self) -> None:
        self.tempdir.cleanup()
        try:
            self.cursor.execute(f"DROP DATABASE IF EXISTS {self.config['database']['database']}")
        except Error as e:
            print("Error while working with MySQL", e)
        if self.cnx.is_connected():
            self.cursor.close()
            self.cnx.close()
            print("MySQL connection is closed")
    
    def fingerprint(self, path: Path) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        shutil.copy(path, self.tempdir.name)
        self.track = Path(os.path.join(self.tempdir.name, path.name))
        self.djv.fingerprint_directory(self.tempdir.name, [path.suffix])
    
    def keypoints(self, program_type) -> list[interval]:
        if self.track is None:
            print('not fingerprinted yet')
            return []
        return getattr(self, '_process_' + program_type.name.lower())(**self._init_function(program_type))
    
    def _init_function(self, program_type: ProgramType) -> dict[dict[str, tp.Any] | str]:
        pathes = {Path(file).stem + '_path': os.path.join(cfg.data_dir, program_type.value, file)
                     for file in os.listdir(os.path.join(cfg.data_dir, program_type.value)) if Path(file).suffix == '.mp3'}
        recognizes = {Path(file).stem: self.djv.recognize(FileRecognizer, file) for file in pathes.values()}
        return recognizes | pathes
    
    def _process_volgann_novosti(self, opening, intro, advt, outro, **path) -> list[interval]: 
        intervals = []
        
        opening_start = None
        if opening['results'][0]['input_confidence'] > 0.3:
            opening_start = opening['results'][0]['offset_seconds']
            opening_length = librosa.get_duration(path=path['opening_path'])
            intervals.append((IntervalType.UNKNOWN, (0, opening_start)))
            intervals.append((IntervalType.OPENING, (opening_start, opening_start + opening_length)))   
        
        intro_start, intro_end = None, None
        if intro['results'][0]['input_confidence'] > 0.3:
            intro_start = intro['results'][0]['offset_seconds']
            for r in intro['results']:
                if abs(intro_start - r['offset_seconds']) > 5:
                    intro_end = r['offset_seconds']
                    break
            if intro_start > intro_end:
                intro_start, intro_end = intro_end, intro_start
            intro_length = librosa.get_duration(path=path['intro_path'])
            if opening_start is not None and min(opening_start, intro_start, intro_end) != opening_start:
                intervals = [(IntervalType.UNKNOWN, (0, intro_start))]
            intervals.append((IntervalType.MUSICAL_BREAK, (intro_start, intro_start + intro_length)))
            intervals.append((IntervalType.ANNOUNCEMENT, (intro_start + intro_length, intro_end)))
            intervals.append((IntervalType.MUSICAL_BREAK, (intro_end, intro_end + intro_length)))
        
        advt_start, advt_end = None, None
        if advt['results'][0]['input_confidence'] > 0.3:
            advt_length = librosa.get_duration(path=path['advt_path'])
            offsets = [of['offset_seconds'] for of in advt['results']]
            pairs = self._get_pairs(path['advt_path'], self.track, offsets, 120000)
            last = 0
            if intervals != []:
                last = intervals[-1][1][1]
            for advt_start, advt_end in pairs:
                intervals.append((IntervalType.NEWS, (last, advt_start)))
                intervals.append((IntervalType.MUSICAL_BREAK, (advt_start, advt_start + advt_length)))
                intervals.append((IntervalType.ADVERTISEMENT, (advt_start + advt_length, advt_end)))
                intervals.append((IntervalType.MUSICAL_BREAK, (advt_end, advt_end + advt_length)))
                last = advt_end + advt_length
            
        outro_start = None
        if outro['results'][0]['input_confidence'] > 0.35:
            outro_length = librosa.get_duration(path=path['outro_path'])
            offsets = [of['offset_seconds'] for of in outro['results']]
            pairs = self._get_pairs(path['outro_path'], self.track, offsets, 50000)
            if pairs:
                for outro_start, _ in pairs:
                    intervals.append((IntervalType.NEWS, (intervals[-1][1][1], outro_start)))
                    intervals.append((IntervalType.CLOSING, (outro_start, outro_start + outro_length))) 
                    intervals.append((IntervalType.UNKNOWN, (outro_start + outro_length, librosa.get_duration(path=self.track))))
                    break
            elif self._compare_audio(path['outro_path'], self.track, outro['results'][0]['offset_seconds']) < 50000:
                outro_start = outro['results'][0]['offset_seconds']
                intervals.append((IntervalType.NEWS, (intervals[-1][1][1], outro_start)))
                intervals.append((IntervalType.CLOSING, (outro_start, outro_start + outro_length))) 
                intervals.append((IntervalType.UNKNOWN, (outro_start + outro_length, librosa.get_duration(path=self.track))))
            else:
                intervals.append((IntervalType.NEWS, (intervals[-1][1][1], librosa.get_duration(path=self.track))))
        else:
            intervals.append((IntervalType.NEWS, (intervals[-1][1][1], librosa.get_duration(path=self.track))))
        
        return intervals 
    
    def _process_volgann_posleslovie_sobitiya_nedeli(self, opening, intro, intro_close, advt, outro, **path) -> list[interval]:
        intervals = []

        opening_start = None
        if opening['results'][0]['input_confidence'] > 0.3:
            opening_start = opening['results'][0]['offset_seconds']
            opening_length = librosa.get_duration(path=path['opening_path'])
            intervals.append((IntervalType.UNKNOWN, (0, opening_start)))
            intervals.append((IntervalType.OPENING, (opening_start, opening_start + opening_length)))  

        intro_start = None
        if intro['results'][0]['input_confidence'] > 0.3:
            intro_length = librosa.get_duration(path=path['intro_path'] )
            r = [(w['offset_seconds'], self._compare_audio(path['intro_path'] , self.track, w['offset_seconds'])) 
                 for w in intro['results'] if w['offset_seconds'] < 300]
            if opening_start is not None:
                r = sorted([i for i in r if i[0] > opening_start and i[1] < 120000], key=lambda x: x[0])
            if r != []:
                intro_start = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.MUSICAL_BREAK, (intro_start, intro_start + intro_length)))

        intro_end = None
        if intro_close['results'][0]['input_confidence'] > 0.3:
            intro_close_length = librosa.get_duration(path=path['intro_close_path'])
            r = [(w['offset_seconds'], self._compare_audio(path['intro_close_path'], self.track, w['offset_seconds'])) 
                 for w in intro_close['results'] if w['offset_seconds'] < 600]
            if intro_start is not None:
                r = sorted([i for i in r if i[0] > intro_start and i[1] < 100000], key=lambda x: x[0])
                intro_end = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.ANNOUNCEMENT, (intro_start + intro_length, intro_end)))
            elif opening_start is not None:
                r = sorted([i for i in r if i[0] > opening_start and i[1] < 100000], key=lambda x: x[0])
                intro_end = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.ANNOUNCEMENT, (opening_start + opening_length, intro_end)))
            else:
                r = sorted([i for i in r if i[1] < 100000], key=lambda x: x[0])
                intro_end = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.ANNOUNCEMENT, (0, intro_end)))
            intervals.append((IntervalType.MUSICAL_BREAK, (intro_end, intro_end + intro_close_length)))

        advt_start, advt_end = None, None
        if advt['results'][0]['input_confidence'] > 0.3:
            advt_length = librosa.get_duration(path=path['advt_path'])
            offsets = [of['offset_seconds'] for of in advt['results']]
            pairs = self._get_pairs(path['advt_path'], self.track, offsets, 120000)
            last = 0
            if intervals != []:
                last = intervals[-1][1][1]
            for advt_start, advt_end in pairs:
                intervals.append((IntervalType.NEWS, (last, advt_start)))
                intervals.append((IntervalType.MUSICAL_BREAK, (advt_start, advt_start + advt_length)))
                intervals.append((IntervalType.ADVERTISEMENT, (advt_start + advt_length, advt_end)))
                intervals.append((IntervalType.MUSICAL_BREAK, (advt_end, advt_end + advt_length)))
                last = advt_end + advt_length

        outro_start = None
        if outro['results'][0]['input_confidence'] > 0.35:
            outro_length = librosa.get_duration(path=path['outro_path'])
            offsets = [of['offset_seconds'] for of in outro['results']]
            pairs = self._get_pairs(path['outro_path'], self.track, offsets, 50000)
            if pairs:
                for outro_start, _ in pairs:
                    intervals.append((IntervalType.NEWS, (intervals[-1][1][1], outro_start)))
                    intervals.append((IntervalType.CLOSING, (outro_start, outro_start + outro_length))) 
                    intervals.append((IntervalType.UNKNOWN, (outro_start + outro_length, librosa.get_duration(path=self.track))))
                    break
            elif self._compare_audio(path['outro_path'], self.track, outro['results'][0]['offset_seconds']) < 50000:
                outro_start = outro['results'][0]['offset_seconds']
                intervals.append((IntervalType.NEWS, (intervals[-1][1][1], outro_start)))
                intervals.append((IntervalType.CLOSING, (outro_start, outro_start + outro_length))) 
                intervals.append((IntervalType.UNKNOWN, (outro_start + outro_length, librosa.get_duration(path=self.track))))
            else:
                intervals.append((IntervalType.NEWS, (intervals[-1][1][1], librosa.get_duration(path=self.track))))
        else:
            intervals.append((IntervalType.NEWS, (intervals[-1][1][1], librosa.get_duration(path=self.track))))
     
        return intervals
    
    def _process_russia1_vesti_privolzhie(self, opening, intro, intro_close, weather, outro, **path) -> list[interval]:
        intervals = []

        opening_start = None
        if opening['results'][0]['input_confidence'] > 0.3:
            opening_start = opening['results'][0]['offset_seconds']
            opening_length = librosa.get_duration(path=path['opening_path'])
            intervals.append((IntervalType.UNKNOWN, (0, opening_start)))
            intervals.append((IntervalType.OPENING, (opening_start, opening_start + opening_length)))  

        intro_start = None
        if intro['results'][0]['input_confidence'] > 0.3:
            intro_length = librosa.get_duration(path=path['intro_path'])
            r = [(w['offset_seconds'], self._compare_audio(path['intro_path'], self.track, w['offset_seconds'])) 
                 for w in intro['results'] if w['offset_seconds'] < 300]
            if opening_start is not None:
                r = sorted([i for i in r if i[0] > opening_start and i[1] < 50000], key=lambda x: x[0])
            if r != []:
                intro_start = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.MUSICAL_BREAK, (intro_start, intro_start + intro_length)))

        intro_end = None
        if intro_close['results'][0]['input_confidence'] > 0.3:
            intro_close_length = librosa.get_duration(path=path['intro_close_path'])
            r = [(w['offset_seconds'], self._compare_audio(path['intro_close_path'], self.track, w['offset_seconds'])) 
                 for w in intro_close['results'] if w['offset_seconds'] < 600]
            if intro_start is not None:
                r = sorted([i for i in r if i[0] > intro_start and i[1] < 120000], key=lambda x: x[0])
                intro_end = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.ANNOUNCEMENT, (intro_start + intro_length, intro_end)))
            elif opening_start is not None:
                r = sorted([i for i in r if i[0] > opening_start and i[1] < 120000], key=lambda x: x[0])
                intro_end = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.ANNOUNCEMENT, (opening_start + opening_length, intro_end)))
            else:
                r = sorted([i for i in r if i[1] < 120000], key=lambda x: x[0])
                intro_end = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.ANNOUNCEMENT, (0, intro_end)))
            intervals.append((IntervalType.MUSICAL_BREAK, (intro_end, intro_end + intro_close_length)))

        weather_start = None
        if weather['results'][0]['input_confidence'] > 0.3:
            weather_start = weather['results'][0]['offset_seconds']
            weather_length = librosa.get_duration(path=path['weather_path'])
            intervals.append((IntervalType.NEWS, (intervals[-1][1][1], weather_start)))
            intervals.append((IntervalType.WEATHER_START, (weather_start, weather_start + weather_length))) 

        outro_start = None
        if outro['results'][0]['input_confidence'] > 0.35:
            outro_length = librosa.get_duration(path=path['outro_path'])
            if outro['results'][0]['offset_seconds'] > librosa.get_duration(path=self.track) - 60 or \
                self._compare_audio(path['outro_path'], self.track, outro['results'][0]['offset_seconds']) < 100000:
                outro_start = outro['results'][0]['offset_seconds']
                intervals.append((IntervalType.NEWS, (intervals[-1][1][1], outro_start)))
                intervals.append((IntervalType.CLOSING, (outro_start, outro_start + outro_length))) 
                intervals.append((IntervalType.UNKNOWN, (outro_start + outro_length, librosa.get_duration(path=self.track))))
            else:
                intervals.append((IntervalType.NEWS, (intervals[-1][1][1], librosa.get_duration(path=self.track))))
        else:
            intervals.append((IntervalType.NEWS, (intervals[-1][1][1], librosa.get_duration(path=self.track))))

        return intervals
    
    def _process_russia1_sobitiya_nedeli(self, opening, intro, intro_close, weather, outro, **path) -> list[interval]:       
        intervals = []

        opening_start = None
        if opening['results'][0]['input_confidence'] > 0.3:
            opening_start = opening['results'][0]['offset_seconds']
            opening_length = librosa.get_duration(path=path['opening_path'])
            intervals.append((IntervalType.UNKNOWN, (0, opening_start)))
            intervals.append((IntervalType.OPENING, (opening_start, opening_start + opening_length)))  

        intro_start = None
        if intro['results'][0]['input_confidence'] > 0.3:
            intro_length = librosa.get_duration(path=path['intro_path'])
            r = [(w['offset_seconds'], self._compare_audio(path['intro_path'], self.track, w['offset_seconds'])) 
                 for w in intro['results'] if w['offset_seconds'] < 300]
            if opening_start is not None:
                r = sorted([i for i in r if i[0] > opening_start and i[1] < 50000], key=lambda x: x[0])
            if r != []:
                intro_start = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.MUSICAL_BREAK, (intro_start, intro_start + intro_length)))

        intro_end = None
        if intro_close['results'][0]['input_confidence'] > 0.3:
            intro_close_length = librosa.get_duration(path=path['intro_close_path'])
            r = [(w['offset_seconds'], self._compare_audio(path['intro_close_path'], self.track, w['offset_seconds'])) 
                 for w in intro_close['results'] if w['offset_seconds'] < 600]
            if intro_start is not None:
                r = sorted([i for i in r if i[0] > intro_start and i[1] < 140000], key=lambda x: x[0])
                intro_end = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.ANNOUNCEMENT, (intro_start + intro_length, intro_end)))
            elif opening_start is not None:
                r = sorted([i for i in r if i[0] > opening_start and i[1] < 140000], key=lambda x: x[0])
                intro_end = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.ANNOUNCEMENT, (opening_start + opening_length, intro_end)))
            else:
                r = sorted([i for i in r if i[1] < 140000], key=lambda x: x[0])
                intro_end = min(r, key=lambda x: x[1])[0]
                intervals.append((IntervalType.ANNOUNCEMENT, (0, intro_end)))
            intervals.append((IntervalType.MUSICAL_BREAK, (intro_end, intro_end + intro_close_length)))

        weather_start = None
        if weather['results'][0]['input_confidence'] > 0.3:
            weather_start = weather['results'][0]['offset_seconds']
            weather_length = librosa.get_duration(path=path['weather_path'])
            intervals.append((IntervalType.NEWS, (intervals[-1][1][1], weather_start)))
            intervals.append((IntervalType.WEATHER_START, (weather_start, weather_start + weather_length))) 

        outro_start = None
        if outro['results'][0]['input_confidence'] > 0.35:
            outro_length = librosa.get_duration(path=path['outro_path'])
            if outro['results'][0]['offset_seconds'] > librosa.get_duration(path=self.track) - 60 or \
                self._compare_audio(path['outro_path'], self.track, outro['results'][0]['offset_seconds']) < 100000:
                outro_start = outro['results'][0]['offset_seconds']
                intervals.append((IntervalType.NEWS, (intervals[-1][1][1], outro_start)))
                intervals.append((IntervalType.CLOSING, (outro_start, outro_start + outro_length))) 
                intervals.append((IntervalType.UNKNOWN, (outro_start + outro_length, librosa.get_duration(path=self.track))))
            else:
                intervals.append((IntervalType.NEWS, (intervals[-1][1][1], librosa.get_duration(path=self.track))))
        else:
            intervals.append((IntervalType.NEWS, (intervals[-1][1][1], librosa.get_duration(path=self.track))))

        return intervals
    
    def _process_russia24_vesti_nizhny_novgorod(self, opening, intro, closing, **path) -> list[interval]:
        intervals = []

        opening_start = None
        if opening['results'][0]['input_confidence'] > 0.3:
            opening_start = opening['results'][0]['offset_seconds']
            opening_length = librosa.get_duration(path=path['opening_path'])
            intervals.append((IntervalType.UNKNOWN, (0, opening_start)))
            intervals.append((IntervalType.OPENING, (opening_start, opening_start + opening_length)))  

        closing_start = None
        if closing['results'][0]['input_confidence'] > 0.3:
            r = [(w['offset_seconds'], self._compare_audio(path['closing_path'], self.track, w['offset_seconds'])) for w in closing['results']
                if (w['offset_seconds'] > opening_start + opening_length if opening_start is not None else True)]
            r = sorted([w for w in r if w[1] < 100000], key=lambda x: x[1])
            closing_start = r[0][0]
            closing_length = librosa.get_duration(path=path['closing_path'])

        if intro['results'][0]['input_confidence'] > 0.3:
            r = [(w['offset_seconds'], self._compare_audio(path['intro_path'], self.track, w['offset_seconds'])) for w in intro['results']]
            intro_length = librosa.get_duration(path=path['intro_path'])
            for w in sorted(r, key=lambda x: x[0]):
                if w[1] > 50000:
                    continue
                if opening_start is not None and w[0] < opening_start + opening_length:
                    continue
                if closing_start is not None and w[0] > closing_start:
                    continue
                if intervals != []:
                    if intervals[-1][0] == IntervalType.OPENING:
                        intervals.append((IntervalType.ANNOUNCEMENT, (intervals[-1][1][-1], w[0])))
                    else:
                        intervals.append((IntervalType.NEWS, (intervals[-1][1][-1], w[0])))
                else:
                    intervals.append((IntervalType.ANNOUNCEMENT, (0, w[0])))
                intervals.append((IntervalType.MUSICAL_BREAK, (w[0], w[0] + intro_length)))
                
        if closing is not None:
            intervals.append((IntervalType.NEWS, (intervals[-1][1][-1], closing_start)))
            intervals.append((IntervalType.CLOSING, (closing_start, closing_start + closing_length)))
            intervals.append((IntervalType.UNKNOWN, (closing_start + closing_length, librosa.get_duration(path=self.track))))

        return intervals
    
    def _get_pairs(self, sample: Path, track: Path, offsets: list[float], max_diff: float, 
                   min_delta: float = 5, max_delta: float = 300) -> set[tuple[float, float]]:

        h = []
        result = set()
        for offset in offsets:
            k = self._compare_audio(sample, track, offset_seconds=offset)
            if k < max_diff:
                h.append((offset, k))
        
        h = sorted(h, key=lambda x: x[0])
        hh = [h[0]]
        for i in h[1:]:
            if i[0] - hh[-1][0] < 1:
                if i[1] > hh[-1][1]:
                    hh[-1] = i
            else:
                hh.append(i)

        h = sorted(hh, reverse=True, key=lambda x: x[1])
        used = [False for _ in range(len(h))]
        for i, first in enumerate(h):
            if used[i]:
                continue
            for j, second in enumerate(h):
                if used[j]:
                    continue
                if min_delta < abs(first[0] - second[0]) < max_delta:
                    used[i] = True
                    used[j] = True
                    left, right = sorted([first[0], second[0]])
                    for interval in result:
                        if interval[0] < left < interval[1] and left - interval[0] > min_delta and right - interval[1] > min_delta:
                            result.remove(interval)
                            result.add((interval[0], left))
                            result.add((interval[1], right))
                            break
                        elif interval[0] < right < interval[1] and interval[0] - left > min_delta and interval[1] - right > min_delta:
                            result.remove(interval)
                            result.add((left, interval[0]))
                            result.add((right, interval[1]))
                            break
                    else:
                        result.add((left, right))
                    break
        return result
        
    def _compare_audio(self, sample: Path, track: Path, offset_seconds: float) -> float:
        # Загружаем первый аудиофайл
        y1, sr1 = librosa.load(str(sample))

        # Загружаем второй аудиофайл
        y2, sr2 = librosa.load(str(track), offset=offset_seconds, duration=librosa.get_duration(y=y1, sr=sr1))

        # Убедимся, что y2 имеет такую же длину, что и y1
        if len(y1) > len(y2):
            y2 = librosa.util.fix_length(y2, size=len(y1))

        # Получаем спектрограмму для каждого файла
        S1 = np.abs(librosa.stft(y1))
        S2 = np.abs(librosa.stft(y2))

        # Сравниваем спектрограммы
        diff = np.abs(S1 - S2)

        return np.sum(diff)
