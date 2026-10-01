import cv2
import os

def cut_video(input_path, output_path, start_sec, end_sec, target_size=(1280, 720)):
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file {input_path}")
        return
        
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    while cap.isOpened():
        ret, frame = cap.read()
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not ret or current_sec > end_sec:
            break
        resized_frame = cv2.resize(frame, target_size)
        out.write(resized_frame)
        
    cap.release()
    out.release()
    print(f"Đã cắt video thành công: {output_path}")

if __name__ == "__main__":
    cut_video("test_data/raw_videos/sample.mp4", "test_data/raw_videos/sample_10s.mp4", start_sec=0, end_sec=10)