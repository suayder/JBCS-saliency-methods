import os
from argparse import ArgumentParser
import cv2
import numpy as np
from tqdm import tqdm
import scipy.stats
from scipy.signal import fftconvolve

OVERLAY_ALPHA = 0.3
REFRAME_ASPECT_RATIO = (9,16)

all_crops = []

def find_max_sum_position(image, box_shape):

    box_kernel = np.ones(box_shape, dtype=int)
    convolution_result = fftconvolve(image, box_kernel, mode='same')
    # Find the position with the maximum sum
    max_position = np.unravel_index(np.argmax(convolution_result), convolution_result.shape)
    return max_position

def overlay_img(img, saliency_map):
    scaled_map = ((saliency_map - saliency_map.min()) * (1 / (saliency_map.max() - saliency_map.min()) * 255)).astype('uint8')
    heatmap = cv2.applyColorMap(scaled_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1,  heatmap, OVERLAY_ALPHA, 0)
    return overlay


def crop_and_center_image(frame, x0, y0, x1, y1):
    croped_img = np.zeros((frame.shape[0], frame.shape[1], 3)).astype(np.uint8)
    cy, cx = frame.shape[0] // 2, frame.shape[1] // 2
    xc0, yc0 = cx - (x1 - x0) // 2, cy - (y1 - y0) // 2
    xc1, yc1 = cx + (x1 - x0) // 2, cy + (y1 - y0) // 2
    croped_img[yc0:yc1, xc0:xc1] = frame[y0:y1, x0:x1]
    return croped_img


def accumulate_points(frame_img_path, saliency_map_path, reduce_factor=0.8, reframe=False):

    frame = cv2.imread(frame_img_path)
    saliency_map = cv2.imread(saliency_map_path)

    assert frame is not None and saliency_map is not None, "frame or saliency map is None"

    if reframe:
        fx, fy = compute_scaling_factors(frame.shape[:2])
        fx*=reduce_factor
        fy*=reduce_factor
    else:
        fx = fy = reduce_factor

    if frame.shape[:2] != saliency_map.shape[:2]:
        saliency_map = cv2.resize(saliency_map, (frame.shape[1], frame.shape[0]))

    if len(frame.shape) == 3:
        rect_shape = (int(frame.shape[0]*fy), int(frame.shape[1]*fx), frame.shape[2])
    else:
        rect_shape = (int(frame.shape[0]*fy), int(frame.shape[1]*fx))

    optimal_bbox = find_max_sum_position(saliency_map, rect_shape)
    x0, y0 = (np.clip(optimal_bbox[1] - rect_shape[1] // 2, 0, np.inf),
              np.clip(optimal_bbox[0] - rect_shape[0] // 2, 0, np.inf))
    x1, y1 = (x0 + rect_shape[1], y0 + rect_shape[0])
    all_crops.append((x0, y0, x1, y1))

def moving_average_interpolation(coords, image_shape, window_size=5):
    coords = np.array(coords)
    smoothed = np.zeros_like(coords)

    # interpolate center x and center y
    cx_interp = np.convolve(((coords[:, 0] + coords[:, 2]) / 2), np.ones(window_size)/window_size, mode='same')
    cy_interp = np.convolve(((coords[:, 1] + coords[:, 3]) / 2), np.ones(window_size)/window_size, mode='same')

    # calculate width and height of the boxes
    width = scipy.stats.mode(coords[:, 2] - coords[:, 0]).mode
    height = scipy.stats.mode(coords[:, 3] - coords[:, 1]).mode

    # translate the original coords to the new interpolated center positions
    smoothed[:, 0] = np.clip((cx_interp - width / 2), 0, np.inf).astype(int)  # x0
    smoothed[:, 1] = np.clip((cy_interp - height / 2), 0, np.inf).astype(int)  # y0

    # Ensure x1 and y1 are within bounds and adjust x0 and y0 if necessary
    smoothed[:, 2] = np.clip(smoothed[:, 0] + width, 0, image_shape[1]).astype(int)  # x1
    smoothed[:, 3] = np.clip(smoothed[:, 1] + height, 0, image_shape[0]).astype(int)  # y1

    # Adjust x0 and y0 to maintain the same width and height
    smoothed[:, 0] = np.clip(smoothed[:, 2] - width, 0, smoothed[:, 2]).astype(int)  # x0
    smoothed[:, 1] = np.clip(smoothed[:, 3] - height, 0, smoothed[:, 3]).astype(int)  # y0

    return smoothed.tolist()


def crop_frame(frame_img_path, coords, saliency_map_path, reduce_factor=0.8, overlay=False, reframe=False, center_crop=False):

    frame = cv2.imread(frame_img_path)
    saliency_map = cv2.imread(saliency_map_path)

    assert frame is not None and saliency_map is not None, "frame or saliency map is None"

    x0, y0, x1, y1 = map(int, coords)

    if overlay:
        frame = overlay_img(frame, saliency_map)

    if center_crop:
        croped_img = crop_and_center_image(frame, x0, y0, x1, y1)
    else:
        croped_img = frame[y0:y1, x0:x1]

    return croped_img

def compute_scaling_factors(original_shape):

    h, w = original_shape
    r_w, r_h = REFRAME_ASPECT_RATIO
    target_ratio = r_w / r_h

    if (w / h) > target_ratio:
        # Image is wider than the target aspect ratio, limit height
        fy = 1.0
        fx = (h * target_ratio) / w 
    else:
        # Image is taller than the target aspect ratio, limit width
        fx = 1.0
        fy = (w / target_ratio) / h

    return fx, fy

def crop_and_save(f_path, coords, sal_path, save_path, reduce_factor, overlay, reframe, center_crop):

    croped_img = crop_frame(f_path, coords, sal_path, reduce_factor=reduce_factor, overlay=overlay, reframe=reframe, center_crop=center_crop)
    file_name = os.path.basename(f_path)
    cv2.imwrite(os.path.join(save_path, file_name), croped_img)

def crop_video(args):
    frames_path = sorted(os.listdir(args.img_dir), key=lambda n: int(n.split('.')[0]))
    saliency_maps_path = sorted(os.listdir(args.maps_dir), key=lambda n: int(n.split('.')[0]))

    # temp
    frames_path = [p for p in frames_path if p in saliency_maps_path]
    frames_path = [os.path.join(args.img_dir, f) for f in frames_path]
    saliency_maps_path = [os.path.join(args.maps_dir, m) for m in  saliency_maps_path]
    
    assert len(frames_path) == len(saliency_maps_path), "no matching size in saliency maps {0} != {1}".format(len(frames_path), len(saliency_maps_path))

    os.makedirs(args.save_path, exist_ok=True)

    i = 0
    for f_path, sal_path in zip(frames_path, saliency_maps_path):
        i+=1
        accumulate_points(f_path, sal_path, args.reduce_factor, args.reframe)
        if i==100:
            pass

    img_shape = cv2.imread(frames_path[0]).shape[:2]
    interpolated = moving_average_interpolation(all_crops, img_shape, window_size=5)
    for f_path, sal_path, (x0, y0, x1, y1) in zip(frames_path, saliency_maps_path, interpolated):
        crop_and_save(f_path, (x0, y0, x1, y1), sal_path, args.save_path, args.reduce_factor, args.overlay, args.reframe, args.center_crop)
    
    # recortar o frame


if __name__ == "__main__":

    parser = ArgumentParser(description="Crop images based on attention maps")
    parser.add_argument("--img-dir", type=str, default='/home/suayder/Desktop/method2/images/', help="Path to frames of the video")
    parser.add_argument("--maps-dir", type=str, default='/home/suayder/Desktop/method2/maps/', help="Path to attention maps directory (jpg)")
    parser.add_argument("--save-path", type=str, default='/home/suayder/Desktop/method2/crops/', help="output folder")
    parser.add_argument("--reduce-factor", default=0.8, type=float)
    parser.add_argument("--overlay", action="store_true", default=False, help="overlay the saliency map on the frame")
    parser.add_argument("--reframe", action="store_true", default=False)
    parser.add_argument("--center-crop", action="store_true", default=False, help="center the crop in image")
    args = parser.parse_args()

    crop_video(args)
    print('done')