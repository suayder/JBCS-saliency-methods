"""
Helpers. This file contains the annotation's processing functions and click data loader

"""
import os
from pathlib import Path

import cv2
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d


class ClickAnnotation:

    def __init__(self, annotation_path: str, video_name, interpolate=False, **kwargs):
        """
        Expected the folder structure (n>=1):

        annotation_path/
            user_1/
                video_name/
                    %d.txt
            user_2/
                video_name/
                    %d.txt
            ...
            user_n/
                video_name/
                    %d.txt
        """
        annotation_path = Path(annotation_path)
        self.annotators = {i: d for i, d in enumerate(annotation_path.iterdir()) if d.is_dir()}

        all_clicks = []
        for i, folder in self.annotators.items():
            df_clicks = file_loader(os.path.join(str(annotation_path.resolve()), folder, video_name), ret='dataframe')
            if interpolate:
                sequence_length = kwargs.get('sequence_length', None)
                df_clicks = interpolate_frames(df_clicks, sequence_length)

            df_clicks = df_clicks.astype({'x': 'int', 'y': 'int'})
            df_clicks['user'] = i
            all_clicks.append(df_clicks)

        self.clicks = self.__concat_users(all_clicks)

    def __concat_users(self, all_clicks):
        def rename_columns(df):
            user = df['user'].unique()[0]
            return df.rename(columns={'x': f'x_{user}', 'y': f'y_{user}'}).drop(columns=['user'])

        all_clicks = [rename_columns(df) for df in all_clicks]

        merged = all_clicks[0].copy()
        # Merge each subsequent dataframe
        for df in all_clicks[1:]:
            merged = pd.merge(merged, df, on='frame', how='outer')

        return merged.reset_index().astype({'frame': 'int'}).set_index('frame')

    def __len__(self):
        self.clicks.sort_index(inplace=True)
        return len(self.clicks)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.clicks):
            click = self.clicks.iloc[self.count]
            click = [(click[f'x_{i}'], click[f'y_{i}']) for i in range(len(click) // 2)]
            self.count += 1
            return click
        else:
            raise StopIteration
    
    def __getitem__(self, idx):
        if idx < len(self.clicks) and idx in self.clicks.index:
            click = self.clicks.loc[idx]
            click = [(click[f'x_{i}'], click[f'y_{i}']) for i in range(len(click) // 2)]
            return click
        else:
            raise IndexError(f'Index {idx} not available, try one of these {list(self.clicks.index)}')

    def __repr__(self):
        return f'number of clicks: {len(self.clicks)}\nNumber of annotators {len(self.annotators)}'

class VideoIterator:
    def __init__(self, video_path):
        """
        Loads a video as an interator
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    def __iter__(self):
        if not self.cap.isOpened():
            raise StopIteration
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    @property
    def current_frame(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def num_frames(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))
    
    def frame(self, frame_id):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    
def file_loader(path, ret='dataframe'):
    """
    This is a function that load the annotated text files and load to a DataFrame with columns frame,x,y

    Args:
        ret: type for returned data, can be ['dict','dataframe']
    Return:
        Daframe with columns frame,x,y or a dict like this {frame_number: {'x': x, 'y': y}, ...}
    """
    files = [f for f in os.listdir(path) if f.endswith('.txt')]

    all_clicks = []
    for file in files:
        with open(os.path.join(path, file)) as f:
            line = f.readline().strip().split(',')

        all_clicks.append(file.split('.')[:1] + line)

    df_clicked = pd.DataFrame(all_clicks, columns=['frame', 'x', 'y']).astype(int)
    df_clicked = df_clicked.set_index('frame')
    df_clicked.sort_values('frame', inplace=True)

    assert (ret in ['dataframe', 'dict'])
    if ret == 'dataframe':
        return df_clicked
    else:
        return df_clicked.to_dict(orient='index')

## processing functions

def interpolate_frames(sequence, sequence_lenght=None):
    """
    Interpolates frames given a sequence

    Args:
        sequence: numpy array or dataframe containing three columns (frame, x, y)
        sequence_lenght: the max length to one sequence
    Returns:
         Numpy array with the interpolated sequence
    """

    sequence = sequence.reset_index().to_numpy() if isinstance(sequence, pd.DataFrame) else sequence
    sequence = sequence[sequence[:, 0].argsort()]

    max_frame = sequence_lenght if sequence_lenght else int(sequence[:, 0].max())
    all_frames = set(range(0, max_frame))
    x = sequence[:, 1]
    y = sequence[:, 2]
    z = sequence[:, 0]

    missing_frames = list(all_frames - set(z))

    interpx = interp1d(z, x, kind='cubic', bounds_error=False, fill_value=x.mean())
    interpy = interp1d(z, y, kind='cubic', bounds_error=False, fill_value=y.mean())
    x_interp = interpx(missing_frames)
    y_interp = interpy(missing_frames)

    interp_array = np.column_stack((missing_frames, x_interp, y_interp))
    interpolated_frames = np.vstack((sequence[:, :3], interp_array))
    interpolated_df = pd.DataFrame(interpolated_frames, columns=['frame', 'x', 'y']).set_index('frame')

    interpolated_df = interpolated_df.astype({'x': int, 'y': int})
    interpolated_df = interpolated_df.sort_index()

    return interpolated_df


def gaussian_2d(x, y, mean_x, mean_y, sigma):
    return np.exp(-((x - mean_x) ** 2 + (y - mean_y) ** 2) / (2 * sigma ** 2))


def gen_gaussians(points, img_shape, sigma):
    height, width = img_shape
    heatmap = np.zeros((height, width))

    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    x, y = np.meshgrid(x, y)
    for (mean_x, mean_y) in points:
        heatmap += gaussian_2d(x, y, mean_x, mean_y, sigma)

    heatmap /= np.max(heatmap)  # Normalize to the range [0, 1]
    return (heatmap * 255).astype(np.uint8)