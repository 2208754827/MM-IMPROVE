from ultralytics import RTDETRMM

model = RTDETRMM('D:/JiQI/MM-experiment/ResTest/RTDETR-mid/weights/best.pt')
model.cocoval(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
          split='val',device='0',half=True,
        #   modality='x',
          project='D:/JiQI/MM-experiment/ResTest',
          name='Val')