import os
import argparse
from queue import Queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import yaml
import scipy.stats

from data import VideoIterator, ClickAnnotation, file_loader

class PATHS:

    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../", "config.yaml")

    @classmethod
    def load_config(cls):
        """Load paths from YAML file"""
        with open(cls.config_path, "r") as file:
            config = yaml.safe_load(file)

        cls.video_name = config["video_name"]
        cls.video_path = config["video_path"].format(video_name=cls.video_name)
        cls.annotation_path = config["annotation_path"]
        cls.save_dirs = config["save_dirs"].format(video_name=cls.video_name)

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

PATHS.load_config()

def save_worker(queue: Queue, save_dir: str, num_threads: int):
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 200
    def save_file(data):
        index, heatmap = data
        np.save(os.path.join(save_dir, f'{str(index).zfill(5)}.npy'), heatmap)

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


def filter_outliers(points):
    points = np.array(points)
    x_values = points[:, 0]
    y_values = points[:, 1]

    def iqr_outliers(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data > lower_bound) | (data < upper_bound) # return negative outliers
    def zscore_outliers(data):
        z_scores = scipy.stats.zscore(data)
        return np.abs(z_scores) < 1.5

    x_outliers = zscore_outliers(x_values)
    y_outliers = zscore_outliers(y_values)

    # Mark points as outliers if they are outliers in either dimension
    outlier_mask = x_outliers & y_outliers
    return points[outlier_mask]

def gen_fixation(image_shape, points):
    image = np.zeros(image_shape, dtype='uint8')
    for point in points:
        if np.isnan(point).any():
            continue
        image[min(point[0], image_shape[0])-1, min(point[1], image_shape[1])-1] = 255
    return image

def save_fixation(args):
    
    video_names = PATHS.video_names() if args.all else [PATHS.video_name]
    for video_name in video_names:
        PATHS.set_video_name(PATHS, video_name)
        video = VideoIterator(PATHS.video_path)
        annotations = ClickAnnotation(PATHS.annotation_path, PATHS.video_name, interpolate=True, sequence_length=video.num_frames+1)
        save_dir = os.path.join(PATHS.save_dirs, video_name)
        

        queue = multiprocessing.Queue(maxsize=600)
        save_process = multiprocessing.Process(target=save_worker, args=(queue, save_dir, 10))
        save_process.start()

        pbar = tqdm(total=len(annotations), desc=f'video: {video_name}')
        for i, ann in enumerate(annotations, start=1):
            ann = filter_outliers(ann)
            fixation = gen_fixation((video.height, video.width), ann)
            queue.put((i, fixation))
            pbar.update(1)
        queue.put(None)
        save_process.join()
        pbar.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', default=PATHS.video_path)
    parser.add_argument('--video-name', default=PATHS.video_name)
    parser.add_argument('--annotation-path', default=PATHS.annotation_path)
    parser.add_argument('--save-dirs', default=PATHS.save_dirs)
    parser.add_argument('--all', action='store_true',default=False)
    args = parser.parse_args()

    PATHS.video_path = args.video_path
    PATHS.annotation_path = args.annotation_path
    PATHS.save_dirs = args.save_dirs
    PATHS.video_name = args.video_name
    save_fixation(args)

main()