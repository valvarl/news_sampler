import os.path as osp
import tempfile
import datetime
from pathlib import Path

import click
from moviepy.editor import VideoFileClip

import keypoints

@click.command()
@click.option('-i', '--input', 'filename', required=True, type=click.Path(exists=True, resolve_path=True, path_type=Path, dir_okay=False), help='Input file')
@click.option('-t', '--type', 'program_type', required=True, type=click.Choice([t for t in keypoints.ProgramType._member_names_], case_sensitive=False), help='Program type')
def main(filename, program_type):
    """Extracts fragments from a video/audio file."""
    tempdir = tempfile.TemporaryDirectory()
    if filename.suffix in ['.mp4']:
        clip = VideoFileClip(str(filename))
        filename = Path(osp.join(tempdir.name, filename.stem + '.mp3'))
        clip.audio.write_audiofile(str(filename))
    try:
        kp = keypoints.KeyPoints()
        kp.fingerprint(filename)
        points = kp.keypoints(keypoints.ProgramType._member_map_[program_type])
        for p in points:
            print(datetime.timedelta(seconds=p[1][0]), datetime.timedelta(seconds=p[1][1]), p[0].name)
    finally:
        tempdir.cleanup()
        