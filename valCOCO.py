#这是专门的COCO指标计算方法，请参考我的使用方法
#我的这些参数方法是可以调整，请你根据你的项目调试
from ultralytics import YOLOMM

model = YOLOMM('D:/JiQI/MM-experiment/ResTest/YOLOMM/weights/best.pt')
model.cocoval(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
          split='test',device='1',half=True,
        #   modality='x',
          project='D:/JiQI/MM-experiment/ResTest',
          name='Val')