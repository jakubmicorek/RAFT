import os

root_folder_shanghaitech = "/mnt/data/anomaly_detection_datasets_preprocessed/shanghaitech/"

root_folder_shanghaitech_data_test_rgb = os.path.join(root_folder_shanghaitech, "shanghaitech/testing/frames/")
root_folder_shanghaitech_data_train_rgb = os.path.join(root_folder_shanghaitech, "shanghaitech/training/videos/")

root_folder_shanghaitech_data_test_flow = os.path.join(root_folder_shanghaitech, "shanghaitech_flow/testing/frames/")
root_folder_shanghaitech_data_train_flow = os.path.join(root_folder_shanghaitech, "shanghaitech_flow/training/videos/")


def get_dirs_without_files(root_path):
    return [item for item in sorted(os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, item))]


# export train flow
print(get_dirs_without_files(root_folder_shanghaitech_data_train_rgb))
for current_video_folder in get_dirs_without_files(root_folder_shanghaitech_data_train_rgb):
    current_path_rgb = os.path.join(root_folder_shanghaitech_data_train_rgb, current_video_folder)
    current_path_flow = os.path.join(root_folder_shanghaitech_data_train_flow, current_video_folder)
    command = f"python demo.py models/raft-things.pth {current_path_rgb} --path_output={current_path_flow}"
    print(command)
    os.system(command)


# export test flow
print(get_dirs_without_files(root_folder_shanghaitech_data_test_rgb))
for current_video_folder in get_dirs_without_files(root_folder_shanghaitech_data_test_rgb):
    current_path_rgb = os.path.join(root_folder_shanghaitech_data_test_rgb, current_video_folder)
    current_path_flow = os.path.join(root_folder_shanghaitech_data_test_flow, current_video_folder)
    command = f"python demo.py models/raft-things.pth {current_path_rgb} --path_output={current_path_flow}"
    print(command)
    os.system(command)





