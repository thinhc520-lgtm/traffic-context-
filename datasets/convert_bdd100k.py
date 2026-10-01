"""
Chuyển đổi nhãn BDD100k từ định dạng JSON sang file txt theo chuẩn YOLOv5
"""
import os
import json
from pathlib import Path
import sys
from tqdm import tqdm #Thư viện hiển thị thanh tiến trình trực quan (% hoàn thành, tốc độ xử lý) trên màn hình dòng lệnh khi lặp qua danh sách dài.

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.label_mapper import map_bdd_detection

def convert_bdd100k_to_yolo(box2d, img_width, img_height):
    """
    Chuyển đổi bounding box từ định dạng BDD100k sang định dạng YOLOv5
    :param box2d: dict chứa thông tin bounding box theo định dạng BDD100k
    :param img_width: chiều rộng ảnh
    :param img_height: chiều cao ảnh
    :return: list [class_id, x_center, y_center, width, height] theo chuẩn YOLOv5
    """
    x1 = float(box2d['x1'])
    y1 = float(box2d['y1'])
    x2 = float(box2d['x2'])
    y2 = float(box2d['y2'])

    # Tính toán tọa độ trung tâm và kích thước của bounding box theo chuẩn YOLOv5
    x_center = (x1 + x2) / 2.0 / img_width #tính tọa độ tâm trục ngang
    y_center = (y1 + y2) / 2.0 / img_height #tính tọa độ tâm trục dọc
    width = (x2 - x1) / img_width #tính chiều rộng tương đối
    height = (y2 - y1) / img_height #tính chiều cao 

    #Giới hạn trong khoảng [0, 1] để tránh lỗi khi bounding box nằm ngoài ảnh
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height

def process_bdd_labels(json_path, output_dir, img_width=1280, img_height=720):
    #Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    print(f"Đang đọc nhãn từ file: {json_path}")
    with open(json_path,"r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Đang chuyển đổi {len(data)} nhãn sang định dạng YOLOv5...")
    converted_count = 0
    box_count = 0
    for item in tqdm(data):
        img_name = item.get("name","")
        if not img_name:
            continue

        base_name = os.path.splitext(img_name)[0]
        label_file = os.path.join(output_dir, f"{base_name}.txt")

        yolo_lines = []
        labels = item.get("labels", [])
        if labels:
            for lbl in labels:
                category = lbl.get("category", "")
                box2d = lbl.get("box2d", None)

                if box2d is None:
                    continue

                class_id = map_bdd_detection(category)
                if class_id is None:
                    continue

                x_center, y_center, width, height = convert_bdd100k_to_yolo(box2d, img_width, img_height)

                if width <= 0 or height <= 0:
                    continue

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                box_count += 1

        with open(label_file, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)

        converted_count += 1
    print(f"Hoàn tất chuyển đổi. Đã xử lý {converted_count} nhãn và tạo {box_count} bounding box.")

if __name__ == "__main__":
    #Ví dụ sử dụng
    json_path = "path/to/bdd100k_labels.json"  # Thay bằng đường dẫn thực tế đến file nhãn BDD100k
    output_dir = "path/to/output_labels"        # Thay bằng đường dẫn thư mục đầu ra cho file txt YOLOv5
    process_bdd_labels(json_path, output_dir)
    print("Chuyển đổi nhãn BDD100k sang định dạng YOLOv5 hoàn tất.")