"""
is expected image folder contains enumerated frames xxxxx.jpg, eg:
images/
    00001.jpg
"""

import os
import random
from argparse import ArgumentParser
import cv2

class CustomInstance:
    def __init__(self, path):
        self.image_path = os.path.join(path, 'images')
        self.click_path = os.path.join(path, 'clicks')
        self.images = sorted([i for i in os.listdir(self.image_path) if i.endswith('.jpg')])
        self.clicks = sorted([i for i in os.listdir(self.click_path) if i.endswith('.txt')])

    def get_pair_path(self, idx=None):

        if idx is None:
            click = random.choice(self.clicks)
            idx = int(click.replace('.txt', ''))
        else:
            click = f'{idx}.txt'
        frame = str(idx).zfill(5)+'.jpg'
        print(frame)
        
        click_path = os.path.join(self.click_path, click)
        frame_path = os.path.join(self.image_path, frame)
        return (frame_path, click_path)

def read_click(txt_file_path):
    with open(txt_file_path, 'r') as f:
        line = f.readline().split(',')
        x,y = list(map(int,line))
    return x,y

def display_click(frame_path, point):
    image = cv2.imread(frame_path)
    
    image = cv2.circle(image, point, 15, (255,0,0), -1)

    cv2.imshow('click', image)
    cv2.waitKey(0)

def main(args):
    print(args)
    instance = CustomInstance(args.sideseeinginstance)
    
    p = instance.get_pair_path(args.frame_number)
    xypoints = read_click(p[1])
    display_click(p[0], xypoints)

    # draw click on image

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("sideseeinginstance", help="path to instance in SideSeeing")
    parser.add_argument("--frame-number", type=int, help="interger representing the frame to display")

    args = parser.parse_args()
    main(args)