import cv2
import numpy as np
import os
from context_reasoner import ContextReasoner
from visualizer import Visualizer

def run_test():
    # 1. Giả lập 1 Khung hình (Khung đen 640x480)
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 2. Giả lập Dữ liệu BBox từ Thành viên B
    mock_detections = [
        {"bbox": [100, 150, 300, 350], "label": "car", "score": 0.92},
        {"bbox": [280, 200, 340, 380], "label": "pedestrian", "score": 0.88} # Nằm gần xe
    ]

    # 3. Chạy Reasoner & Visualizer
    reasoner = ContextReasoner(proximity_threshold=150.0)
    visualizer = Visualizer()

    reasoning_result = reasoner.analyze(mock_detections)
    output_frame = visualizer.draw(mock_frame, mock_detections, reasoning_result)

    # 4. Xuất kết quả kiểm tra
    os.makedirs("test_data", exist_ok=True)
    output_path = "test_data/mock_output.jpg"
    cv2.imwrite(output_path, output_frame)
    
    print("=== CHẠY DỰ DỤNG PIPELINE THÀNH CÔNG ===")
    print("Kết quả suy luận:", reasoning_result)
    print(f"Ảnh kết quả đã lưu tại: {output_path}")

if __name__ == "__main__":
    run_test()