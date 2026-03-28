#这是提供 参考示例的 YOLOMM推理器使用方法
#按必须按照你的需求来更改而不是盲目去使用


from doctest import debug
from ultralytics import YOLOMM
model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/YOLOMM-LST/weights/best.pt')

# 测试双模态预测（新API - 显式rgb_source/x_source）
print("=== 测试双模态预测 ===")
model.predict(rgb_source='/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',
              x_source='/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png',
              project ='ResTest',
              name='YOLOMM-LST_dual_modal',
              save=True,
              exist_ok=True,
              debug=True,
              )

# 测试单模态RGB预测（使用零填充X模态）
print("\n=== 测试单模态RGB预测 ===")
model.predict(rgb_source='/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',
              x_source=None,
              project ='ResTest',
              name='YOLOMM-LST_single_rgb',
              save=True,
              exist_ok=True,
              debug=True
              )

# 测试单模态红外预测（使用零填充RGB模态）
print("\n=== 测试单模态红外预测 ===")
model.predict(rgb_source=None,
              x_source='/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png',
              project ='ResTest',
              name='YOLOMM-LST_single_ir',
              save=True,
              exist_ok=True,
              debug=True
              )