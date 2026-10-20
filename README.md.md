**基于 YOLO11n 的工业表面缺陷检测系统**

**1. 项目简介**

**本项目基于 YOLO11n 目标检测模型，实现工业金属表面缺陷的自动识别与定位，并通过 Streamlit 构建可交互的 Web 演示界面。**

**用户可以上传工业表面图片，查看缺陷检测框、缺陷类别、置信度及检测统计信息。**

**2. 项目功能**

**工业表面图片上传**

**基于 YOLO11n 的缺陷目标检测**

**缺陷位置可视化与置信度展示**

**检测目标数量和类别统计**

**检测结果明细表**

**CSV 检测结果导出**

**3. 技术栈**

**Python**

**PyTorch**

**Ultralytics YOLO11n**

**Streamlit**

**Pandas**

**Pillow**

**4. 数据集**

**使用 NEU-DET 工业表面缺陷数据集，包含以下六类缺陷：**

**Crazing（龟裂）**

**Inclusion（夹杂）**

**Patches（斑块）**

**Pitted Surface（点蚀表面）**

**Rolled-in Scale（轧入氧化皮）**

**Scratches（划痕）**

**5. 模型训练与评估**

**模型采用 YOLO11n，训练轮数为 50 epochs。**

**当前基线评估指标：**

**Precision：0.753**

**Recall：0.825**

**mAP@0.5：0.814**

**mAP@0.5:0.95：0.491**

**以上指标为此前记录的基线评估结果。正式发布时，应确保它们对应实际保存的评估报告，并注明评估数据集及相关参数。**

**6. 项目结构**

**IndustrialVision/ ├── app.py ├── README.md ├── data/ │ └── NEU-DET/ └── outputs/ └── neu_yolo11n_train50/ └── weights/ ├── best.pt └── last.pt**

**7. 运行方法**

**在 Windows CMD 中执行：**

**C:\AIProjects\IndustrialVision\vision312\Scripts\python.exe -m streamlit run C:\AIProjects\IndustrialVision\app.py**

**启动后，在浏览器打开：**

**<http://localhost:8501>**

## 系统演示

![工业表面缺陷检测系统演示](assets/demo.png)

**8. 项目成果**

**完成了从工业缺陷数据集准备、YOLO11n 模型训练与评估，到 Streamlit Web 部署及检测结果可视化的完整流程。**

**项目支持图片检测、缺陷定位、类别统计和 CSV 结果导出，可用于工业视觉检测场景的学习与演示。**
