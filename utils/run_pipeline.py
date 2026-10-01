import os
import sys

# Thêm thư mục utils vào đường dẫn hệ thống
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from label_mapper import map_bdd_detection, YOLO_DETECTION_CLASS_NAMES
from visualizer import draw_boxes

def run_pipeline_check():
    print("==========================================")
    print("🚀 BẮT ĐẦU VERIFY PIPELINE TRAFFIC CONTEXT")
    print("==========================================")

    frames_dir = "test_data/extracted_frames"
    sample_frame = os.path.join(frames_dir, "frame_0000.jpg")
    output_demo = os.path.join(frames_dir, "pipeline_demo.jpg")

    if not os.path.exists(sample_frame):
        print(f"Loi: Khong tim thay tep {sample_frame}")
        return

    # 1. Kiểm tra ánh xạ nhãn BDD100K -> YOLO
    test_categories = ["car", "pedestrian", "bicycle"]
    print("\n[1] Kiem tra Label Mapper:")
    for cat in test_categories:
        class_id = map_bdd_detection(cat)
        class_name = YOLO_DETECTION_CLASS_NAMES[class_id] if class_id is not None else "Unknown"
        print(f"    - Nhan BDD '{cat}' -> YOLO Class ID {class_id} ({class_name})")

    # 2. Kiểm tra Visualizer
    print("\n[2] Kiem tra Visualizer (Ve Bounding Box mau):")
    demo_boxes = [(100, 100, 300, 250), (350, 120, 450, 280)]
    demo_labels = ["Vehicle (0.95)", "Pedestrian (0.88)"]
    demo_colors = [(0, 255, 0), (0, 0, 255)]  # Xanh lá (Vehicle), Đỏ (Pedestrian)

    success = draw_boxes(sample_frame, output_demo, demo_boxes, demo_labels, demo_colors)
    
    if success:
        print("\n==========================================")
        print("✅ TOAN BO HA TANG XU LY (UTILS) DA SAN SANG!")
        print("==========================================")

if __name__ == "__main__":
    run_pipeline_check()