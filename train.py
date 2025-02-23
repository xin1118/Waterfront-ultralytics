import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model=YOLO(r'yolov8-test.yaml') # 模型文件路径

    model.train(data='data.yaml', # yaml数据路径
                imgsz=320,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=4,
                workers=10,
                device='0'
                )
