import sys
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

from pathlib import Path


DEVICE = 'cuda:0'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack([img, img, img]).transpose(1, 2, 0)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    # cv2.waitKey(1)
    return flo[:, :, [2, 1, 0]]


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.tif')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            rgb_flow = viz(image1, flow_up)

            if args.path_output is not None:
                # save flow as rgb image
                # root_folder_images = os.path.dirname(imfile1)
                # if root_folder_images.startswith("/"):
                #     root_folder_images = root_folder_images[1:]
                # flow_output_path = os.path.join(args.path_output, root_folder_images)
                flow_output_path = args.path_output
                # FIXME: will divide by max magnitude in frame
                flow_rgb_path = os.path.join(flow_output_path, "flow_rgb")
                # flow_rgb_path = f'{args.path_output}_flow_rgb'
                Path(flow_rgb_path).mkdir(parents=True, exist_ok=True)
                image_file_name_1_flow_rgb = f'{os.path.basename(imfile1).split(".")[0]}.jpg'
                cv2.imwrite(os.path.join(flow_rgb_path, image_file_name_1_flow_rgb), rgb_flow)

                flow_up = flow_up.cpu().numpy()
                # save flow as uv vectors. 32bit floats
                flow_uv_path = os.path.join(flow_output_path, "flow_uv")
                # flow_uv_path = f'{args.path_output}_flow_uv'
                image_file_name_1_flow_uv = f'{os.path.basename(imfile1).split(".")[0]}.npy'
                Path(flow_uv_path).mkdir(parents=True, exist_ok=True)

                with open(os.path.join(flow_uv_path, image_file_name_1_flow_uv), 'wb') as f:
                    np.save(f, flow_up)

                # # save flow as sparse uv vectors. ignore pixel shifts of smaller sparse_threshold
                # sparse_threshold = 0.9
                # flow_uv_sparse_path = os.path.join(os.path.dirname(imfile1), "flow_uv_sparse")
                # image_file_name_1_flow_uv = f'{os.path.basename(imfile1).split(".")[0]}.npz'
                # Path(flow_uv_sparse_path).mkdir(parents=True, exist_ok=True)
                #
                # flow_up[flow_up < sparse_threshold] = 0
                # sparse_matrix = scipy.sparse.csc_matrix(flow_up)
                # scipy.sparse.save_npz(os.path.join(flow_uv_sparse_path, image_file_name_1_flow_uv), sparse_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="restore checkpoint")
    parser.add_argument('path', help="dataset for evaluation")
    parser.add_argument('--path_output', help="save flow to")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    demo(args)
