"""
This function will generate the attention maps from the given clicks logs.

Expected folder structure (you will pass `clicks_path`):
```
clicks_path/
    user1_clicks_folder/
        video_name/
            *.txt
            ...
    user2_clicks_folder/
        video_name/
            *.txt
            ...
```
"""

import os
import argparse
import random
from queue import Queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal

from data import VideoIterator, ClickAnnotation, file_loader

class PATHS:
    # preciso criar uma forma de atualizar o path do video tambem conforme demanda, tem algumas decisões de design a serem feitas, por exemplo devo considerar a hierarquia atual dos videos? ou devo criar minha propria regra
    #fazer commit e depois fazer pull na rede vision, mudar os paths e rodar o script para gerar 
    #enquanto isso vou organizando os codigos de treinamento, separando os videos em frames, etc
    
    video_name = 'Block01-2024-02-28-15-06-34-538'
    video_path = f'/run/media/suayder/data/JBCS_paper/ds1/videos/{video_name}/video.mp4'
    annotation_path = '/run/media/suayder/data/JBCS_paper/ds1/clicks/'
    save_dirs = './attention_maps'

    # load clicks from path
    @classmethod
    def load_clicks(cls):
        user_folders = [f for f in os.listdir(cls.annotation_path) if os.path.isdir(os.path.join(cls.annotation_path, f))]
        all_clicks = []
        for i, user_folder in enumerate(user_folders):
            clicks = file_loader(os.path.join(cls.annotation_path, user_folder, cls.video_name))
            all_clicks.append(clicks)
        
        return all_clicks
    
    def set_video_name(cls, new_name):
        cls.video_path = cls.video_path.replace(cls.video_name, new_name)
        cls.video_name = new_name


    @classmethod
    def user_names(cls):
        return [f for f in os.listdir(cls.annotation_path) if os.path.isdir(os.path.join(cls.annotation_path, f))]
    
    @classmethod
    def video_names(cls):
        return [f for f in os.listdir(os.path.join(cls.annotation_path, cls.user_names()[0])) if os.path.isdir(os.path.join(cls.annotation_path, cls.user_names()[0], f))]

def gaussian_mixture(image_shape, points, sigma=1):
    """
    Generate a Gaussian mixture centered on the given points for the specified image shape.
    
    :param image_shape: tuple, shape of the image (height, width).
    :param points: list of tuples, each tuple is a point (x, y) representing the center of a Gaussian.
    :param sigma: float, standard deviation of the Gaussian distribution.
    :return: numpy array, image with Gaussian mixture.
    """
    height, width = image_shape
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    pos = np.dstack((x, y))
    
    image = np.zeros(image_shape)

    for point in points:
        if np.isnan(point).any():
            continue
        mean = point
        covariance = np.array([[sigma**2, 0], [0, sigma**2]])
        rv = multivariate_normal(mean, covariance)
        image += rv.pdf(pos)

    image = image - image.min()  # Shift to make the minimum zero
    image = image / image.max()  # Scale to make the maximum one
    image = (image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    return image

def save_worker(queue: Queue, save_dir: str, num_threads: int):
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 10
    def save_file(data):
        index, heatmap = data
        np.save(os.path.join(save_dir, f'{index}.npy'), heatmap)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        buffer = []  # Buffer to store batch data
        while True:
            try:
                item = queue.get(timeout=5)  # Avoid infinite blocking
                if item is None:
                    break
                buffer.append(item)

                # Process batch when enough items are accumulated
                if len(buffer) >= batch_size:
                    executor.map(save_file, buffer)
                    buffer.clear()
            except:
                pass  # Continue checking queue

        # Save any remaining files in buffer
        if buffer:
            executor.map(save_file, buffer)


def video_attention_maps(all=False):

    sigma=120

    if all:
        video_names = PATHS.video_names()
        for video_name in video_names:
            PATHS.set_video_name(PATHS, video_name)
            video = VideoIterator(PATHS.video_path)
            annotations = ClickAnnotation(PATHS.annotation_path, PATHS.video_name ,interpolate=True, sequence_length=video.num_frames+1)
            save_dir = os.path.join(PATHS.save_dirs, video_name)
            
            queue = multiprocessing.Queue(maxsize=30)
            save_process = multiprocessing.Process(target=save_worker, args=(queue, save_dir, 5))
            save_process.start()

            pbar = tqdm(total=len(annotations), desc=f'video: {video_name}')
            for i, ann in enumerate(annotations, start=1):
                heatmap = gaussian_mixture((video.height, video.width), ann, sigma)
                queue.put((i, heatmap))
                pbar.update(1)
            queue.put(None)
            save_process.join()
            pbar.close()

    else:
        video = VideoIterator(PATHS.video_path)
        annotations = ClickAnnotation(PATHS.annotation_path, PATHS.video_name ,interpolate=True, sequence_length=video.num_frames+1)

        render = Render(interpolate=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # change this to mp4
        out = cv2.VideoWriter('attention_maps.mp4', fourcc, video.fps//2, (video.width * 2, video.height))

        for i, ann in enumerate(annotations, start=1):
            if i % 2 == 0:
                continue
            print(f'Generating attention map for frame {i}')
            heatmap = render.render_all(i, show=False)
            out.write(heatmap)
 
        out.release()
        cv2.destroyAllWindows()


class Render:
    def __init__(self, clicks_path=PATHS.annotation_path, video_path=PATHS.video_path, video_name=PATHS.video_name, **kwargs):
        """
        Render attention maps in the image.

        Args:
            maps_path: Path to the attention maps.
            clicks_path: Path to the clicks.
        """
        self.video = VideoIterator(video_path)
        interpolate = kwargs.get('interpolate', False)
        self.annotations = ClickAnnotation(clicks_path, video_name ,interpolate=interpolate, sequence_length=self.video.num_frames+1)
        self.colors = {i: col for i, col in enumerate(self.random_colors(len(self.annotations.annotators)))}
        self.att_map_sigma = 120
    
    @staticmethod
    def random_colors(num_colors):
        """
        Generate random colors (RGB).
        """
        return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]
    
    @staticmethod
    def plot_map_on_frame(input_prob, frame):

        if input_prob.shape[0:1] != frame.shape[0:1]:
            input_prob = cv2.resize(input_prob, (frame.shape[1], frame.shape[0]))
        heatmap_img = cv2.applyColorMap(input_prob, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(heatmap_img, 0.35, frame, 0.65, 0)

        return frame

    def attention_point(self, click_frame, show=False):

        # heatmap = gaussian_mixture((self.video.height, self.video.width), self.annotations[click_frame], self.att_map_sigma)
        # draw points
        point_masks = np.zeros((self.video.height, self.video.width, 3), dtype=np.uint8)
        for i, point in enumerate(self.annotations[click_frame]):
            if np.isnan(point).any():
                continue
            point = list(map(int, point))
            point_masks = cv2.circle(point_masks, (point[0], point[1]), 5, self.colors[i], -1)

        if show:
            cv2.imshow('points in the mask', point_masks)
            cv2.waitKey(0)

        return point_masks
    
    def attention_map(self, click_frame, show=False):
        gaussian = gaussian_mixture((self.video.height, self.video.width), self.annotations[click_frame], self.att_map_sigma)

        if show:
            cv2.imshow('heatmap', gaussian)
            cv2.waitKey(0)

        return gaussian

    def render_all(self, click_frame, show=True):
        heatmap = self.attention_map(click_frame)
        point_masks = self.attention_point(click_frame)
        frame = self.video.frame(click_frame)
        frame = self.plot_map_on_frame(heatmap, frame)
        heatmap = 255-cv2.merge((heatmap, heatmap, heatmap))
        frame = np.concatenate((frame, point_masks), axis=1)
        if show:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
        else:
            return frame

# video_points = [[ j ] + click_point for j, click_point in enumerate(annotation)]
# with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
#     p.starmap(gen_gaussians_and_save, [(p, (video.height, video.width), save_dir) for p in video_points])

# print(f'[{i+1} / {ds.size}] Done - Instance name: {vid_instance.instance_name}')
# break


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generate attention maps from clicks logs. To use look to the PATHS class.')
    parser.add_argument('--plot-click', default=-1, type=int, help='Plot the click in a given frame number.')
    parser.add_argument('--all', action='store_true', default=False, help='Plot all clicks in the video.')
    args = parser.parse_args()

    if args.plot_click > 0:
        render = Render()
        render.render_all(args.plot_click)
    else:
        video_attention_maps(args.all)