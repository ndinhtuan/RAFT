import sys

import cv2.mat_wrapper
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.resize(img, fx=0.5, fy=0.5, dsize=None)
    # print(img, img.shape)
    # cv2.imshow("img", img); cv2.waitKey(0)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def warp_image_with_flow_numpy_only(img, flow, next_img):
    """
    Warps a color image using optical flow (backward warping) with NumPy only (nearest-neighbor).
    
    Parameters:
    - image: np.ndarray of shape (H, W, 3), dtype float32 or uint8
    - flow:  np.ndarray of shape (H, W, 2), where flow[y, x] = (dx, dy)

    Returns:
    - warped: np.ndarray of shape (H, W, 3)
    """
    img = img.copy()
    img = np.array(img, dtype=np.uint8)
    img = np.ascontiguousarray(img)
    img = img[:, :, [2, 1, 0]]

    flow = np.array(flow.copy(), dtype=np.int16)
    # print(flow[100:200,1020:1024,:], np.min(flow), np.max(flow), np.mean(flow)); exit()

    H, W = flow.shape[:2]
    warped = np.zeros_like(img)
    one_threading = False

    import time
    t1 = time.time()

    if one_threading:
        for y in range(H):
            for x in range(W):
                y = H-y-1
                x = W-x-1
                flow_x, flow_y = flow[y, x]
                new_x = x + flow_x 
                new_y = y + flow_y

                if new_x > 0 and new_x < W and new_y > 0 and new_y < H:
                    warped[new_y, new_x, :] = img[y, x, :]
                    # warped[y, x, :] = next_img[new_y, new_x, :]
    else:

        # Create meshgrid of pixel coordinates
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        
        # Compute destination coordinates
        new_x = np.round(x + flow[..., 0]).astype(np.int32)
        new_y = np.round(y + flow[..., 1]).astype(np.int32)

        # Create valid mask
        valid = (0 <= new_x) & (new_x < W) & (0 <= new_y) & (new_y < H)

        # Get flattened indices
        src_yx = np.stack([y[valid], x[valid]], axis=1)
        dst_yx = np.stack([new_y[valid], new_x[valid]], axis=1)

        # Assign values — note: if overlaps happen, later pixels overwrite earlier ones
        warped[dst_yx[:, 0], dst_yx[:, 1]] = img[src_yx[:, 0], src_yx[:, 1]]

    t2 = time.time()
    print("time: ", t2-t1)

    # # Generate grid of coordinates
    # grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    # print("grid: ", grid_x, grid_y)
    # print("flow: ", flow[..., 0], flow[..., 1])

    # # Compute source coordinates
    # src_x = (grid_x - flow[..., 0]).round().astype(int)
    # src_y = (grid_y - flow[..., 1]).round().astype(int)

    # # Mask: valid coordinates within image bounds
    # valid = (src_x >= 0) & (src_x < W) & (src_y >= 0) & (src_y < H)

    # # Flatten arrays to index into 1D arrays
    # tgt_y, tgt_x = grid_y[valid], grid_x[valid]
    # src_y_valid, src_x_valid = src_y[valid], src_x[valid]

    # # Copy pixels from source to destination
    # warped[tgt_y, tgt_x] = img[src_y_valid, src_x_valid]

    cv2.imshow("wraped_img", warped)
    next_img = next_img.copy()
    next_img = np.array(next_img, dtype=np.uint8)
    next_img = next_img[:, :, [2, 1, 0]]
    cv2.imshow("next_img", next_img)

    return warped

def wrap(img, flow, next_img):
    
    img = img.copy()
    img = np.array(img, dtype=np.uint8)
    img = np.ascontiguousarray(img)
    img = img[:, :, [2, 1, 0]]
    img_result = np.zeros_like(img)
    flow = np.array(flow.copy(), dtype=np.int16)
    H, W = flow.shape[:2]
    
    # Create a mesh grid of pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Apply flow to get target coordinates
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    
    # Remap the original image to the new coordinates
    img_result = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow("wraped_img", img_result)
    next_img = next_img.copy()
    next_img = np.array(next_img, dtype=np.uint8)
    next_img = next_img[:, :, [2, 1, 0]]
    cv2.imshow("next_img", next_img)

    return img_result

def viz(img, flo, next_img):
    img = img[0].permute(1,2,0).cpu().numpy()
    next_img = next_img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo_raw = flo.copy()
    flo = flow_viz.flow_to_image(flo)
    warp_image_with_flow_numpy_only(img, flo_raw, next_img)
    img_flo = np.concatenate([img, flo], axis=1)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        print(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, image2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
