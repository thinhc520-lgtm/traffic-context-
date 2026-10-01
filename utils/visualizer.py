import cv2

class Visualizer:
    def __init__(self):
        # Bảng màu BGR cho các đối tượng
        self.color_map = {
            "car": (255, 0, 0),        # Xanh dương
            "pedestrian": (0, 0, 255), # Đỏ
            "person": (0, 0, 255),
            "motorcycle": (0, 255, 255),
            "bus": (255, 255, 0),
            "default": (0, 255, 0)     # Xanh lá
        }

    def draw(self, image, detections, reasoning_result):
        annotated_img = image.copy()

        # 1. Vẽ Bounding Box & Label
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = det.get('label', 'object')
            score = det.get('score', 0.0)
            
            color = self.color_map.get(label.lower(), self.color_map['default'])
            
            # Vẽ BBox
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ Label & Score
            text = f"{label} {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 2. Vẽ Banner trạng thái Suy luận Ngữ cảnh lên đầu hình
        status = reasoning_result.get("status", "NORMAL")
        bg_color = (0, 180, 0) if status == "NORMAL" else (0, 0, 200) # Green / Red
        
        cv2.rectangle(annotated_img, (0, 0), (annotated_img.shape[1], 40), bg_color, -1)
        alert_text = " | ".join(reasoning_result.get("alerts", []))
        cv2.putText(annotated_img, f"STATUS [{status}]: {alert_text}", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_img