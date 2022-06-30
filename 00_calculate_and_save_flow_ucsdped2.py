import os

root_folder_ucsd = "/mnt/data/anomaly_detection_datasets_preprocessed/ucsd"

root_folder_ucsd_ped2_data_test_rgb = os.path.join(root_folder_ucsd, "UCSDped2/Test")
root_folder_ucsd_ped2_data_train_rgb = os.path.join(root_folder_ucsd, "UCSDped2/Train")

root_folder_ucsd_ped2_data_test_flow = os.path.join(root_folder_ucsd, "UCSDped2_flow/Test")
root_folder_ucsd_ped2_data_train_flow = os.path.join(root_folder_ucsd, "UCSDped2_flow/Train")


def get_dirs_without_files(root_path):
    return [item for item in sorted(os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, item)) and not item.endswith("_gt")]


# export train flow
print(get_dirs_without_files(root_folder_ucsd_ped2_data_train_rgb))
for current_video_folder in get_dirs_without_files(root_folder_ucsd_ped2_data_train_rgb):
    current_path_rgb = os.path.join(root_folder_ucsd_ped2_data_train_rgb, current_video_folder)
    current_path_flow = os.path.join(root_folder_ucsd_ped2_data_train_flow, current_video_folder)
    command = f"python demo.py models/raft-things.pth {current_path_rgb} --path_output={current_path_flow}"
    print(command)
    os.system(command)


# export test flow
print(get_dirs_without_files(root_folder_ucsd_ped2_data_test_rgb))
for current_video_folder in get_dirs_without_files(root_folder_ucsd_ped2_data_test_rgb):
    current_path_rgb = os.path.join(root_folder_ucsd_ped2_data_test_rgb, current_video_folder)
    current_path_flow = os.path.join(root_folder_ucsd_ped2_data_test_flow, current_video_folder)
    command = f"python demo.py models/raft-things.pth {current_path_rgb} --path_output={current_path_flow}"
    print(command)
    os.system(command)





