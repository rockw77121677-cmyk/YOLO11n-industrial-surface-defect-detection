import time
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO


# 1. 页面设置
st.set_page_config(
    page_title="工业表面缺陷检测系统",
    page_icon="🔍",
    layout="wide"
)

# 2. 模型路径
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "outputs" / "neu_yolo11n_train50" / "weights" / "best.pt"

# 3. 缺陷类别中文名称
CLASS_NAMES_CN = {
    0: "龟裂",
    1: "夹杂",
    2: "斑块",
    3: "点蚀表面",
    4: "轧入氧化皮",
    5: "划痕"
}


# 4. 加载模型
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# 5. 页面标题
st.title("🔍 工业金属表面缺陷检测系统")
st.caption("基于 YOLO11n 的工业表面缺陷目标检测演示")

st.markdown(
    "上传一张工业表面图片，系统将自动识别缺陷位置、类别和置信度。"
)

# 6. 检查模型文件
if not MODEL_PATH.exists():
    st.error(f"没有找到模型文件：{MODEL_PATH}")
    st.stop()

# 7. 加载模型
model = load_model(MODEL_PATH)

# 8. 侧边栏设置
st.sidebar.header("检测设置")

conf_threshold = st.sidebar.slider(
    "置信度阈值",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

st.sidebar.info("模型：YOLO11n")
st.sidebar.info("训练轮数：50 epochs")
st.sidebar.info("缺陷类别：6 类")

# 9. 上传图片
uploaded_file = st.file_uploader(
    "请选择待检测的工业表面图片",
    type=["jpg", "jpeg", "png", "bmp"]
)

# 10. 执行检测
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("检测结果")

    start_time = time.perf_counter()

    result = model.predict(
        source=image,
        imgsz=416,
        conf=conf_threshold,
        iou=0.5,
        verbose=False
    )[0]

    elapsed_time = time.perf_counter() - start_time

    # 绘制检测框
    plotted_image = result.plot()
    plotted_image = plotted_image[:, :, ::-1]
    result_image = Image.fromarray(plotted_image)

    # 显示原图和检测结果
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 原始图片")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("#### 检测结果")
        st.image(result_image, use_container_width=True)

    # 统计检测结果
    detection_count = len(result.boxes)

    st.subheader("检测统计")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("缺陷目标数量", detection_count)
    metric2.metric("检测耗时", f"{elapsed_time:.3f} 秒")
    metric3.metric("置信度阈值", f"{conf_threshold:.2f}")

    # 整理检测结果表格
    records = []

    if detection_count > 0:
        boxes = result.boxes

        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for index in range(detection_count):
            class_id = class_ids[index]

            records.append(
                {
                    "序号": index + 1,
                    "缺陷类别": CLASS_NAMES_CN.get(
                        class_id,
                        model.names.get(class_id, str(class_id))
                    ),
                    "置信度": round(float(confidences[index]), 4),
                    "左上角X": round(float(coordinates[index][0]), 1),
                    "左上角Y": round(float(coordinates[index][1]), 1),
                    "右下角X": round(float(coordinates[index][2]), 1),
                    "右下角Y": round(float(coordinates[index][3]), 1)
                }
            )

        st.subheader("缺陷检测明细")
        dataframe = pd.DataFrame(records)
        st.dataframe(dataframe, use_container_width=True)

        # 缺陷类别统计
        st.subheader("缺陷类别统计")

        class_counts = dataframe["缺陷类别"].value_counts()
        st.bar_chart(class_counts)

        # 下载检测结果
        csv_data = dataframe.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            label="下载检测结果 CSV",
            data=csv_data,
            file_name="defect_detection_results.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "当前置信度阈值下没有检测到缺陷。"
            "你可以尝试适当降低左侧的置信度阈值。"
        )

else:
    st.info("请先上传一张 JPG、JPEG、PNG 或 BMP 格式的图片。")
