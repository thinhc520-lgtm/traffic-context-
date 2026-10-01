"""
Phân chia tập dữ liệu thành Train(70%), Validation(15%), Test(15%)

"""

import os
import random
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def split_list(items, train_ratio=0.7, val_ratio=0.15):
    """
    Phân chia danh sách items thành 3 tập: train, validation, test
    :param items: danh sách các items cần phân chia
    :param train_ratio: tỷ lệ phần trăm cho tập train
    :param valratio: tỷ lệ phần trăm cho tập validation
    :return: 3 danh sách: train_items, val_items, test_items
    """
    random.shuffle(items)
    n = len(items)
    n_train = int(n* train_ratio)
    n_val = int(n * val_ratio)

    train_items = items[:n_train]
    val_items = items[n_train:n_train+n_val]
    test_items = items[n_train+n_val:]

    return train_items, val_items, test_items

def split_bdd100k(image_dir, output_dir):
    """
    Phân chia tập dữ liệu BDD100k thành 3 tập: train, validation, test
    :param image_dir: thư mục chứa ảnh gốc
    :param output_dir: thư mục đầu ra để lưu các danh sách phân chia
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png"}
    all_images=[
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    train_images, val_images, test_images = split_list(all_images)

    #Ghi ra các file txt
    for split_name, image_list in [("bdd_train.txt",train_images),
                                   ("bdd_val.txt",val_images),
                                   ("bdd_test.txt",test_images)]:
        save_path = os.path.join(output_dir, split_name)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(image_list)+"\n")
    print(f"[BDD100K] Tổng số ảnh: {len(all_images)}, Train: {len(train_images)}, Validation: {len(val_images)}, Test: {len(test_images)}")

def split_nuscenes(scene_to_images_dict, output_dir):
    """
    Phân chia tập dữ liệu nuScenes thành 3 tập: train, validation, test
    :param scene_to_images_dict: dictionary mapping scenes to their images
    :param output_dir: thư mục đầu ra để lưu các danh sách phân chia
    """
    os.makedirs(output_dir, exist_ok=True)

    scenes = list(scene_to_images_dict.keys())
    train_scenes, val_scenes, test_scenes = split_list(scenes)

    train_images=[img for scene in train_scenes for img in scene_to_images_dict[scene]]
    val_images=[img for scene in val_scenes for img in scene_to_images_dict[scene]]
    test_images=[img for scene in test_scenes for img in scene_to_images_dict[scene]]

    for split_name, image_list in [("nuscenes_train.txt",train_images),
                                   ("nuscenes_val.txt",val_images),
                                   ("nuscenes_test.txt",test_images)]:
        save_path = os.path.join(output_dir, split_name)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(image_list)+"\n")
            
    total_frames = len(train_images) + len(val_images) + len(test_images)
    print(f"[nuScenes] Tổng số scene: {len(scenes)} ({total_frames} ảnh)")
    print(f" -> Train: {len(train_scenes)} scenes ({len(train_images)} frames)")
    print(f" -> Validation: {len(val_scenes)} scenes ({len(val_images)} frames)")
    print(f" -> Test: {len(test_scenes)} scenes ({len(test_images)} frames)")

def merge_splits(output_split_dir):
    """
    Gộp các file phân chia từ BDD100k và nuScenes thành 1 file tổng hợp
    để mô hình YoLO có thể đọc trực tiếp khi huấn luyện
    """
    for split_name in ["train", "val", "test"]:
        bdd_file = os.path.join(output_split_dir, f"bdd_{split_name}.txt")
        nuscenes_file = os.path.join(output_split_dir, f"nuscenes_{split_name}.txt")
        total_file = os.path.join(output_split_dir, f"{split_name}.txt")

        combined_lines = []
        if os.path.exists(bdd_file):
            with open(bdd_file, "r", encoding="utf-8") as f:
                combined_lines.extend(f.readlines())
        if os.path.exists(nuscenes_file):
            with open(nuscenes_file, "r", encoding="utf-8") as f:
                combined_lines.extend(f.readlines())

        random.shuffle(combined_lines)
        with open(total_file, "w", encoding="utf-8") as f:
            f.writelines(combined_lines)

    print(f"Đã gộp các file phân chia thành 1 file tổng hợp trong thư mục: {output_split_dir}")

if __name__ == "__main__":
    split_output_dir = "path/to/split_output"  
    print(f"Mô-đun split_dataset.py đã được chạy. Các file phân chia sẽ được lưu tại: {split_output_dir}")