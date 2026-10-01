"""
Kiểm tra tính đúng đắn của nhãn và thống kê phân bố lớp:
1. Vẽ bounding box trên ảnh gốc dựa trên nhãn YOLOv5 và hiển thị ảnh để kiểm tra trực quan.
2. Đếm số lượng đối tượng mỗi class để đánh giá mất cân bằng dữ liệu.
"""
import os
import sys
import random
from pathlib import Path
import cv2
import numpy as np

# Thêm thư mục gốc vào path để import các mô-đun từ utils
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.label_mapper import (
    YOLO_DETECTION_CLASS_ID_TO_NAME,
    SEG_COLOR_MAPPING
)

# Màu đại diện Bbox (BGR cho OpenCV)
BOX_COLORS = {
    0: (0, 0, 255),      # Red cho vehicle
    1: (0, 255, 0),      # Green cho pedestrian
    2: (255, 0, 0),      # Blue cho bicycle/motorcycle
}


def draw_yolo_boxes(image, label_path):
    """
    Đọc file nhãn YOLO .txt và vẽ bounding box kèm tên nhãn lên ảnh
    """
    if not os.path.exists(label_path):
        return image

    h, w, _ = image.shape
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x_c = float(parts[1]) * w
            y_c = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h
        except ValueError:
            continue

        # Kẹp tọa độ trong phạm vi ảnh để an toàn
        x1 = max(0, int(x_c - box_w / 2.0))
        y1 = max(0, int(y_c - box_h / 2.0))
        x2 = min(w - 1, int(x_c + box_w / 2.0))
        y2 = min(h - 1, int(y_c + box_h / 2.0))

        color = BOX_COLORS.get(class_id, (0, 255, 255))
        class_name = YOLO_DETECTION_CLASS_ID_TO_NAME.get(class_id, f"ID:{class_id}")

        # Vẽ hộp
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Vẽ nhãn chữ phía trên hộp
        text = f"{class_name}"
        cv2.putText(image, text, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def overlay_mask(image, mask_path, alpha=0.4):
    """
    Vẽ lớp phủ bán trong suốt (alpha blending) của segmentation mask lên ảnh gốc.
    """
    if not os.path.exists(mask_path):
        return image

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return image

    # Resize mask về cùng cỡ với ảnh nếu cần
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for class_id, rgb_color in SEG_COLOR_MAPPING.items():
        if class_id == 0:  # Không phủ màu lên background
            continue
        # Chuyển RGB sang BGR cho OpenCV
        bgr_color = [rgb_color[2], rgb_color[1], rgb_color[0]]
        colored_mask[mask == class_id] = bgr_color

    # Trộn ảnh gốc và mask
    blended = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    return blended


def visualize_random_samples(image_dir, label_dir, mask_dir=None, output_vis_dir="data/sanity_check", num_samples=25):
    """
    Lấy ngẫu nhiên mẫu ảnh, vẽ box/mask rồi lưu vào thư mục kiểm tra.
    """
    os.makedirs(output_vis_dir, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png"}

    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    if not image_files:
        print(f"Không tìm thấy ảnh trong thư mục: {image_dir}")
        return

    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    print(f"Đang xuất {len(sample_files)} ảnh kiểm tra sang: {output_vis_dir}")

    for img_file in sample_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)
        lbl_path = os.path.join(label_dir, f"{base_name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        if mask_dir:
            msk_path = os.path.join(mask_dir, f"{base_name}.png")
            img = overlay_mask(img, msk_path)

        img = draw_yolo_boxes(img, lbl_path)

        save_path = os.path.join(output_vis_dir, f"check_{img_file}")
        cv2.imwrite(save_path, img)

    print("Hoàn tất trực quan hóa kiểm tra nhãn!")


def calculate_class_distribution(label_dir):
    """
    Thống kê tổng số lượng bounding box theo từng lớp nhãn (kể cả class không hợp lệ).
    """
    stats = {class_id: 0 for class_id in YOLO_DETECTION_CLASS_ID_TO_NAME.keys()}
    total_boxes = 0

    if not os.path.exists(label_dir):
        print(f"Thư mục nhãn không tồn tại: {label_dir}")
        return

    txt_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    for txt_file in txt_files:
        with open(os.path.join(label_dir, txt_file), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        continue
                    stats[class_id] = stats.get(class_id, 0) + 1
                    total_boxes += 1

    print("\n" + "=" * 45)
    print("      THỐNG KÊ PHÂN BỐ LỚP DETECTION")
    print("=" * 45)
    print(f"Tổng số file nhãn đã kiểm tra: {len(txt_files)}")
    print(f"Tổng số bounding boxes:        {total_boxes}\n")

    for class_id in sorted(stats.keys()):
        count = stats[class_id]
        class_name = YOLO_DETECTION_CLASS_ID_TO_NAME.get(class_id, "Unknown/Invalid ID")
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f" - [{class_id}] {class_name:<16}: {count:>8} boxes ({percentage:6.2f}%)")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    # Thay đổi các đường dẫn này theo dataset thực tế của bạn
    DATASET_ROOT = project_root / "data"
    IMAGE_DIR = DATASET_ROOT / "images" / "train"
    LABEL_DIR = DATASET_ROOT / "labels" / "train"
    MASK_DIR = DATASET_ROOT / "masks" / "train"

    print("Bắt đầu kiểm tra sanity check...")
    if LABEL_DIR.exists():
        calculate_class_distribution(str(LABEL_DIR))

    if IMAGE_DIR.exists() and LABEL_DIR.exists():
        mask_path = str(MASK_DIR) if MASK_DIR.exists() else None
        visualize_random_samples(
            str(IMAGE_DIR), 
            str(LABEL_DIR), 
            mask_dir=mask_path, 
            output_vis_dir=str(DATASET_ROOT / "sanity_check"), 
            num_samples=10
        )
    else:
        print("Đường dẫn ảnh hoặc nhãn không tồn tại, vui lòng kiểm tra lại cấu hình.")