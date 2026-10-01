import numpy as np

class ContextReasoner:
    def __init__(self, proximity_threshold=150.0):
        """
        proximity_threshold: Khoảng cách pixel tối đa để cảnh báo nguy cơ va chạm
        """
        self.proximity_threshold = proximity_threshold

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def analyze(self, detections):
        """
        Phân tích mối quan hệ không gian giữa các đối tượng từ danh sách BBox
        """
        alerts = []
        status = "NORMAL"
        
        pedestrians = [d for d in detections if d['label'] in ['person', 'pedestrian']]
        vehicles = [d for d in detections if d['label'] in ['car', 'bus', 'truck', 'motorcycle']]

        # Tính khoảng cách giữa Người đi bộ và Phương tiện
        for ped in pedestrians:
            p_center = self._get_center(ped['bbox'])
            for veh in vehicles:
                v_center = self._get_center(veh['bbox'])
                dist = np.sqrt((p_center[0] - v_center[0])**2 + (p_center[1] - v_center[1])**2)

                if dist < self.proximity_threshold:
                    alerts.append(f"CANH BAO: {ped['label']} sat {veh['label']} ({int(dist)}px)")
                    status = "DANGER"

        if not alerts:
            alerts.append("Giao thong binh thuong")

        return {
            "status": status,
            "alerts": alerts,
            "pedestrian_count": len(pedestrians),
            "vehicle_count": len(vehicles)
        }