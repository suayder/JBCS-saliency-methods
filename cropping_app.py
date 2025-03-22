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

# def crop_frame(frame_path):
#     frame = cv2.imread(os.path.join(PATHS.instance_path, "images", f"{framen:04}.png"))
#     smap = np.load(os.path.join(PATHS.instance_path, "maps", f"{framen}.npy"))
#     np_smap = cv2.imread(os.path.join(PATHS.salience_maps_path, f"{framen:04}.png"))
#     np_smap = cv2.resize(np_smap, (frame.shape[1], frame.shape[0]))
    
#     if (frame is not None) and (np_smap is not None):
#         output_file = os.path.join(PATHS.output_path, 'images', f'{PATHS.video_name}',f'{framen}.png')
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)

#         nv = 4
#         resolution = (int(frame.shape[1]*nv*0.5), int(frame.shape[0]*0.5))  # Adjust the resolution as needed

#         rect_shape = (frame.shape[0], int(frame.shape[0]*9/16), 3)
#         optimal_bbox = find_max_sum_position(np_smap, rect_shape)

#         x0, y0, x1, y1 = (optimal_bbox[1] - rect_shape[1] // 2, optimal_bbox[0] - rect_shape[0] // 2,
#                           optimal_bbox[1] + rect_shape[1] // 2, optimal_bbox[0] + rect_shape[0] // 2)

#         mask = np.ones((frame.shape[0], frame.shape[1], 3)).astype(np.uint8) * 255
#         mask[y0:y1, x0:x1] = frame[y0:y1, x0:x1]

#         scaled_pred_map = ((smap - smap.min()) * (1 / (smap.max() - smap.min()) * 255)).astype('uint8')
#         groudt = cv2.applyColorMap(scaled_pred_map, cv2.COLORMAP_JET)
#         groudt = cv2.addWeighted(groudt, 0.5, frame, 0.5, 0)

#         scaled_pred_map = ((np_smap - np_smap.min()) * (1 / (np_smap.max() - np_smap.min()) * 255)).astype('uint8')
#         heatmap_img = cv2.applyColorMap(scaled_pred_map, cv2.COLORMAP_JET)
#         frame_with_saliences = cv2.addWeighted(heatmap_img, 0.5, frame, 0.5, 0)

#         croped_img = np.zeros((frame.shape[0], frame.shape[1], 3)).astype(np.uint8)
#         cy, cx = frame.shape[0] // 2, frame.shape[1] // 2
#         xc0, yc0 = cx - rect_shape[1] // 2, cy - rect_shape[0] // 2
#         xc1, yc1 = cx + rect_shape[1] // 2, cy + rect_shape[0] // 2
#         # croped_img = cv2.resize(frame[y0:y1, x0:x1], (frame.shape[1], frame.shape[0]))
#         croped_img[yc0:yc1, xc0:xc1] = frame[y0:y1, x0:x1]

#         frame = cv2.putText(frame, "original frame", (frame.shape[0] // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
#         groudt = cv2.putText(groudt, "ground truth", (groudt.shape[0] // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
#         frame_with_saliences = cv2.putText(frame_with_saliences, model_name, (frame_with_saliences.shape[0] // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
#         croped_img = cv2.putText(croped_img, "cropped", (frame.shape[0] // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

#         frame = np.hstack([frame, groudt, frame_with_saliences, croped_img])
#         cv2.imwrite(output_file, frame)
#         print(f"Output image saved at: {output_file}")


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

# def add_text_labels(frame, groudt, frame_with_saliences, croped_img, model_name):
#     frame = cv2.putText(frame, "original frame", (frame.shape[0] // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
#     groudt = cv2.putText(groudt, "ground truth", (groudt.shape[0] // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
#     frame_with_saliences = cv2.putText(frame_with_saliences, model_name, (frame_with_saliences.shape[0] // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
#     croped_img = cv2.putText(croped_img, "cropped", (frame.shape[0] // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
#     return frame, groudt, frame_with_saliences, croped_img

# def combine_and_save_output(frame, groudt, frame_with_saliences, croped_img, output_file):
#     combined_frame = np.hstack([frame, groudt, frame_with_saliences, croped_img])
#     cv2.imwrite(output_file, combined_frame)
#     print(f"Output image saved at: {output_file}")


def crop_frame(frame_img_path, saliency_map_path, reduce_factor=0.8, overlay=False, reframe=False):

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
    
    if overlay:
        frame = overlay_img(frame, saliency_map)

    croped_img = crop_and_center_image(frame, x0, y0, x1, y1)

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

def crop_and_save(f_path, sal_path, save_path, reframe):

    croped_img = crop_frame(f_path, sal_path, overlay=False, reframe=reframe)
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
            futures.append(executor.submit(crop_and_save, f_path, sal_path, args.save_path, args.reframe))
    
    for _ in tqdm(futures, total=len(frames_path)):
        _.result()  
    
    # frame, groudt, frame_with_saliences, croped_img = add_text_labels(frame, groudt, frame_with_saliences, croped_img, model_name)
    # combine_and_save_output(frame, groudt, frame_with_saliences, croped_img, output_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Crop images based on attention maps")
    parser.add_argument("--img-dir", type=str, default='/run/media/suayder/data/JBCS_paper/data/val/0002/images/', help="Path to frames of the video")
    parser.add_argument("--maps-dir", type=str, default='results/stsa-side-map/', help="Path to attention maps directory (jpg)")
    parser.add_argument("--save-path", type=str, default='results/stsa-croped', help="output folder")
    parser.add_argument("--num-workers", default=4)
    parser.add_argument("--reframe", action="store_true", default=False)
    args = parser.parse_args()

    crop_video(args)