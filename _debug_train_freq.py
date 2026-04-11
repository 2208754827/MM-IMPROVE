import traceback
from ultralytics import RTDETRMM

model_path = r'ultralytics/cfg/models/rtmm/r18/aifi-dattention-CSP-FreqSpatial.yaml'
print('start')
try:
    m = RTDETRMM(model_path)
    print('BUILD_OK')
    m.train(
        data=r'D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml',
        epochs=1,
        batch=2,
        imgsz=320,
        device=0,
        workers=0,
        amp=False,
        save=False,
        val=False,
        project='D:/JiQI/MM-experiment/ResTest',
        name='debug_freqspatial',
    )
    print('TRAIN_OK')
except Exception:
    print('TRAIN_FAIL')
    traceback.print_exc()
