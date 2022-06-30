import os
import glob
import numpy as np
import matplotlib.pylab as plt
import cv2
from core.utils import flow_viz_adapted
from pathlib import Path

root_folder = "/mnt/data/anomaly_detection_datasets_preprocessed/ucsd"

root_folder_data_test_flow = os.path.join(root_folder, "UCSDped2_flow/Test")
root_folder_data_train_flow = os.path.join(root_folder, "UCSDped2_flow/Train")

export_root_folder_data_flow_rgb_recalculated = os.path.join(root_folder, "UCSDped2_flow_rgb_recalculated")


def load_metadata(root_folder, metadata_file_name="rad_max_magnitude_flow_uv.npy"):
    with open(os.path.join(root_folder, metadata_file_name), "rb") as f:
        metadata = np.load(f, allow_pickle=True)
        return metadata.tolist()


def get_dirs_without_files(root_path):
    return [item for item in sorted(os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, item))]


metadata_test = load_metadata(root_folder_data_test_flow)
metadata_train = load_metadata(root_folder_data_train_flow)

rad_max_train = np.asarray(sorted([metadata_train[key]["rad_max"] for key in metadata_train])[::-1])
rad_max_test = np.asarray(sorted([metadata_test[key]["rad_max"] for key in metadata_test])[::-1])

if False:  # if plotting
    plt.figure("rad max")
    line, = plt.plot(rad_max_train, label="train")
    plt.plot(rad_max_test, color=line.get_color(), linestyle=":", label="test")
    plt.legend()
    plt.xlabel("sorted rad max flows descending")
    plt.ylabel("rad max (max flow magnitude) value")
    plt.ylim((0, None))
    # plt.grid()
    plt.show()

for current_root_folder in [root_folder_data_train_flow, root_folder_data_test_flow]:
    all_uv_flow_paths = glob.glob(os.path.join(current_root_folder, "*/flow_uv/*"))
    all_uv_flow_paths = sorted(all_uv_flow_paths)
    # print(all_uv_flow_paths)

    u_v_rad_max = dict()
    current_max = 0
    for idx_, current_uv_flow_path in enumerate(all_uv_flow_paths):
        path_metadata = current_uv_flow_path.split("/")[-4:]  # ['Test', 'Test001', 'flow_uv', '001.npy']
        path_metadata[-2] = "flow_rgb"
        path_metadata[-1] = path_metadata[-1].split(".")[0] + ".jpg"

        image_path = os.path.join(export_root_folder_data_flow_rgb_recalculated, "/".join(path_metadata[:-1]))
        Path(image_path).mkdir(parents=True, exist_ok=True)
        with open(current_uv_flow_path, "rb") as f:
            flow_uv = np.load(f)[0].transpose(1, 2, 0)
            flo = flow_viz_adapted.flow_to_image(flow_uv, rad_max=max(rad_max_test.max(), rad_max_train.max()))
            print("save to:", image_path)
            cv2.imwrite(os.path.join(image_path, path_metadata[-1]), flo[..., ::-1])
            # cv2.imshow('image', flo[:, :, [2, 1, 0]] / 255.0)
            # cv2.waitKey(1)

