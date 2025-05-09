"""
is expected image folder contains enumerated mask and frames xxxxx.jpg, eg:
images/
    00001.jpg
    00002.jpg
maps/
    00000.jpg
    00001.jpg
"""

import os
import random
from argparse import ArgumentParser
from PIL import Image

class CustomInstance:
    def __init__(self, path):
        self.image_path = os.path.join(path, 'images')
        self.map_path = os.path.join(path, 'maps')
        self.images = sorted([i for i in os.listdir(self.image_path) if i.endswith('.jpg')])
        self.maps = sorted([i for i in os.listdir(self.map_path) if i.endswith('.jpg')])
        print(self.images[0:10])
        print(self.maps[0:10])

        # verificar se os números dos frames batem (alguem começa com 0 e alguem comeca com 1)

    def get_pair_path(self, idx=None):

        if idx is None:
            idx = random.randint(0, len(self.images))

        frame = str(idx).zfill(5)+'.jpg'
        attmap = str(idx).zfill(5)+'.jpg'
        print(frame)
        
        m_path = os.path.join(self.mask_path, attmap)
        frame_path = os.path.join(self.image_path, frame)
        return (frame_path, m_path)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def display_map(frame_path, map_path):
    image = Image.open(frame_path)
    attmap = Image.open(map_path)
    
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(image)
    show_mask(mask, plt.gca(), obj_id=1)


def main(args):
    print(args)
    instance = CustomInstance(args.sideseeinginstance)
    
    p = instance.get_pair_path(args.frame_number)
    display_map(p[0], p[1])

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("sideseeinginstance", help="path to instance in SideSeeing")
    parser.add_argument("--frame-number", type=int, help="interger representing the frame to display")
    parser.add_argument("--overlay", action='store_true')

    args = parser.parse_args()
    main(args)

# python visualize_mask.py /scratch/suayder/urbanaccess/data/Jundiai_HSV/Block01-2024-02-28-15-06-34-538