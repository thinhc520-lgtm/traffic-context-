import cv2
import os

def extract_frames(video_path, output_folder, frame_interval_sec=1):
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file {video_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Lỗi: Không thể đọc FPS của video.")
        return

    interval = int(fps * frame_interval_sec)
    
    count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            saved_count += 1
        count += 1
        
    cap.release()
    print(f"Đã trích xuất {saved_count} ảnh vào thư mục {output_folder}")

if __name__ == "__main__":
    extract_frames("test_data/raw_videos/sample_10s.mp4", "test_data/extracted_frames/")