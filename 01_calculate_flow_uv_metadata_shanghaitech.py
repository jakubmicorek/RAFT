import os
import glob
import numpy as np

root_folder_shanghaitech = "/mnt/data/anomaly_detection_datasets_preprocessed/shanghaitech/"

root_folder_shanghaitech_data_test_flow = os.path.join(root_folder_shanghaitech, "shanghaitech_flow/testing/")
root_folder_shanghaitech_data_train_flow = os.path.join(root_folder_shanghaitech, "shanghaitech_flow/training/")


def get_dirs_without_files(root_path):
    return [item for item in sorted(os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, item))]


max_mag_per_root = list()
for current_root_folder in [root_folder_shanghaitech_data_train_flow, root_folder_shanghaitech_data_test_flow]:
    all_uv_flow_paths = glob.glob(os.path.join(current_root_folder, "*/*/flow_uv/*"))
    all_uv_flow_paths = sorted(all_uv_flow_paths)
    u_v_rad_max = dict()
    current_max = 0
    for idx_, current_uv_flow_path in enumerate(all_uv_flow_paths):
        with open(current_uv_flow_path, "rb") as f:
            flow_uv = np.load(f)[0]
            u = flow_uv[0]
            v = flow_uv[1]
            rad = np.sqrt(np.square(u) + np.square(v))
            rad_max = np.max(rad)

            current_max = current_max if rad_max < current_max else rad_max

            key_ = "/".join(current_uv_flow_path.split("/")[-3:])
            u_v_rad_max[key_] = dict()
            u_v_rad_max[key_]["u_min"] = u.min()
            u_v_rad_max[key_]["u_max"] = u.max()
            u_v_rad_max[key_]["v_min"] = v.min()
            u_v_rad_max[key_]["v_max"] = v.max()
            u_v_rad_max[key_]["rad_max"] = rad_max

            print(current_max, rad_max, key_, u_v_rad_max[key_])

    max_mag_per_root.append((current_max, current_root_folder))

    save_file = os.path.join(current_root_folder, 'rad_max_magnitude_flow_uv.npy')
    print("save rad max magnitude to:", save_file)
    np.save(save_file, u_v_rad_max)

print("max magnitudes:")
for item_ in max_mag_per_root:
    print(item_)



# export train flow
# print(get_dirs_without_files(root_folder_shanghaitech_data_test_flow))
# for current_video_folder in get_dirs_without_files(root_folder_shanghaitech_data_train_rgb):
#     current_path_rgb = os.path.join(root_folder_shanghaitech_data_train_rgb, current_video_folder)
#     current_path_flow = os.path.join(root_folder_shanghaitech_data_train_flow, current_video_folder)
#     command = f"python demo.py models/raft-things.pth {current_path_rgb} --path_output={current_path_flow}"
#     print(command)
#     # os.system(command)
#
#
# # export test flow
# print(get_dirs_without_files(root_folder_shanghaitech_data_test_rgb))
# for current_video_folder in get_dirs_without_files(root_folder_shanghaitech_data_test_rgb):
#     current_path_rgb = os.path.join(root_folder_shanghaitech_data_test_rgb, current_video_folder)
#     current_path_flow = os.path.join(root_folder_shanghaitech_data_test_flow, current_video_folder)
#     command = f"python demo.py models/raft-things.pth {current_path_rgb} --path_output={current_path_flow}"
#     print(command)
#     # os.system(command)
