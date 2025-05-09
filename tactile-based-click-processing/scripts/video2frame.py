"""
this is just a script that call ffmpeg to generate a folder with images from the video.
"""

import os
import subprocess
from argparse import ArgumentParser
from sideseeing_tools import sideseeing

def split_video_into_frames(video_path, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    command = [
        'ffmpeg',
        '-i', video_path,
        os.path.join(output_dir, '%05d.jpg')
    ]

    subprocess.run(command)


def run_all(args):
    ds = sideseeing.SideSeeingDS(root_dir=args.path_dataset)
    for i in ds.iterator:
        split_video_into_frames(i.video, os.path.join(i.path, 'images'))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('path_dataset')
    args = parser.parse_args()
    run_all(args)

#python video2frame.py /data/side_seeing/urbanaccess