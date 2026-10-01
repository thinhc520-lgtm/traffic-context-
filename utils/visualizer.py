import cv2
import os

def draw_boxes(image_path, output_path, boxes, labels, colors=None):
    """
    Vẽ bounding box và nhãn nhận diện lên khung ảnh.
    - boxes: danh sách [(x1, y1, x2, y2), ...]
    - labels: danh sách tên nhãn ["Vehicle", "Pedestrian", ...]
    - colors: màu BGR cho từng box (mặc định xanh lá)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Loi: Khong the doc anh tu {image_path}")
        return False

    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)  # Mặc định màu xanh lá

    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = map(int, box)
        # Vẽ khung nhận diện
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # Vẽ nhãn phía trên khung
        cv2.putText(image, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Da luu anh truc quan hoa: {output_path}")
    return True

if __name__ == "__main__":
    # Test thử trực tiếp trên frame_0000.jpg đã trích xuất
    input_frame = "test_data/extracted_frames/frame_0000.jpg"
    output_frame = "test_data/extracted_frames/frame_0000_demo.jpg"
    
    # Tạo các box thử nghiệm (x1, y1, x2, y2)
    demo_boxes = [(100, 150, 350, 300), (400, 180, 480, 320)]
    demo_labels = ["Vehicle", "Pedestrian"]
    demo_colors = [(0, 255, 0), (0, 0, 255)]  # Xanh lá cho xe, đỏ cho người đi bộ
    
    draw_boxes(input_frame, output_frame, demo_boxes, demo_labels, demo_colors)