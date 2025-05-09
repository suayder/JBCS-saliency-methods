import os
from argparse import ArgumentParser
import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
from scipy.signal import fftconvolve

OVERLAY_ALPHA = 0.3
REFRAME_ASPECT_RATIO = (9,16)

def find_max_sum_position(image, box_shape):

    box_kernel = np.ones(box_shape, dtype=int)
    convolution_result = fftconvolve(image, box_kernel, mode='same')
    # Find the position with the maximum sum
    max_position = np.unravel_index(np.argmax(convolution_result), convolution_result.shape)

    return max_position

def interpolate_cropping():
    pass

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

def crop_frame(frame_img_path, saliency_map_path, reduce_factor=0.8, overlay=False, reframe=False, center_crop=False):

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
    x0, y0, x1, y1 = (optimal_bbox[1] - rect_shape[1] // 2, optimal_bbox[0] - rect_shape[0] // 2,
                      optimal_bbox[1] + rect_shape[1] // 2, optimal_bbox[0] + rect_shape[0] // 2)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(frame.shape[1], max(frame.shape[1], x1)), min(frame.shape[0], max(frame.shape[0], y1))
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

def crop_and_save(f_path, sal_path, save_path, reduce_factor, overlay, reframe, center_crop):

    croped_img = crop_frame(f_path, sal_path, reduce_factor=reduce_factor, overlay=overlay, reframe=reframe, center_crop=center_crop)
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

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for f_path, sal_path in zip(frames_path, saliency_maps_path):
            futures.append(executor.submit(crop_and_save, f_path, sal_path, args.save_path, args.reduce_factor, args.overlay, args.reframe, args.center_crop))
    
    for _ in tqdm(futures, total=len(frames_path)):
        _.result()  
    
    # frame, groudt, frame_with_saliences, croped_img = add_text_labels(frame, groudt, frame_with_saliences, croped_img, model_name)
    # combine_and_save_output(frame, groudt, frame_with_saliences, croped_img, output_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Crop images based on attention maps")
    parser.add_argument("--img-dir", type=str, default='/data/JBCS_paper/ds1/val/0006/images/', help="Path to frames of the video")
    parser.add_argument("--maps-dir", type=str, default='/data/JBCS_paper/inferences/method_1/humanb_80/0006/', help="Path to attention maps directory (jpg)")
    parser.add_argument("--save-path", type=str, default='/data/JBCS_paper/crop/method1/tmfi-croped-centered', help="output folder")
    parser.add_argument("--num-workers", default=4)
    parser.add_argument("--reduce-factor", default=0.8, type=float)
    parser.add_argument("--overlay", action="store_true", default=False, help="overlay the saliency map on the frame")
    parser.add_argument("--reframe", action="store_true", default=False)
    parser.add_argument("--center-crop", action="store_true", default=False, help="center the crop in image")
    args = parser.parse_args()

    crop_video(args)
    print('done')