import streamlit as st
import cv2
import numpy as np
import os
import sys

# Thêm thư mục utils vào hệ thống để gọi các module
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from context_reasoner import ContextReasoner
from visualizer import Visualizer

# 1. Cấu hình trang Web Demo
st.set_page_config(
    page_title="Traffic Context Reasoning Demo",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Hệ Thống Nhận Diện & Biểu Diễn Ngữ Cảnh Giao Thông")
st.markdown("Đồ án cơ sở ngành KHMT - **Phân tích ngữ cảnh giao thông từ video hành trình**")
st.markdown("---")

# 2. Thanh cấu hình Sidebar (Theo chuẩn yêu cầu đồ án)
st.sidebar.header("⚙️ Cấu Hình Hệ Thống")
dataset_source = st.sidebar.selectbox(
    "Nguồn chuẩn hóa dữ liệu:",
    ["BDD100K", "nuScenes / nuImages"]
)

proximity_thresh = st.sidebar.slider(
    "Ngưỡng khoảng cách cảnh báo (pixels):",
    min_value=50, max_value=300, value=150, step=10
)

st.sidebar.info(f"Đang chọn nguồn dữ liệu: **{dataset_source}**")

# 3. Khu vực Tải tệp thử nghiệm
st.subheader("1. Tải lên ảnh / video hành trình thử nghiệm")
uploaded_file = st.file_uploader(
    "Chọn tệp ảnh (.jpg, .png) để phân tích ngữ cảnh:",
    type=["jpg", "jpeg", "png"]
)

# Khởi tạo mô-đun
reasoner = ContextReasoner(proximity_threshold=proximity_thresh)
visualizer = Visualizer()

# 4. Xử lý và Hiển thị Kết quả
if uploaded_file is not None:
    # Đọc tệp ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("2. Kết quả Phân tích Ngữ cảnh Giao thông")
    col_img, col_info = st.columns([2, 1])

    # Giả lập dữ liệu BBox (Sẵn sàng nối Model thật từ A/B)
    mock_detections = [
        {"bbox": [100, 150, 300, 350], "label": "car", "score": 0.92},
        {"bbox": [280, 200, 340, 380], "label": "pedestrian", "score": 0.88}
    ]

    # Thực hiện Suy luận & Trực quan hóa
    reasoning_result = reasoner.analyze(mock_detections)
    output_frame = visualizer.draw(image, mock_detections, reasoning_result)

    with col_img:
        # Chuyển BGR (OpenCV) sang RGB (Streamlit)
        rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, caption="Khung hình đã xử lý ngữ cảnh", use_container_width=True)

    with col_info:
        st.markdown("### 📊 Dữ liệu Ngữ cảnh (JSON)")
        st.json(reasoning_result)

        status = reasoning_result.get("status", "NORMAL")
        if status == "DANGER":
            st.error(f"⚠️ Trạng thái: **{status}** (Phát hiện nguy cơ va chạm!)")
        else:
            st.success(f"✅ Trạng thái: **{status}** (An toàn)")

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Phương tiện", reasoning_result.get("vehicle_count", 0))
        col_m2.metric("Người đi bộ", reasoning_result.get("pedestrian_count", 0))
else:
    st.info("💡 Vui lòng tải lên một tệp ảnh để xem giao diện phân tích ngữ cảnh.")