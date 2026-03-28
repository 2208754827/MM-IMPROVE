from ultralytics import YOLO

w = r"ResTest/RTDETR-mid/weights/best.pt"   # 改成你实际val用的权重
m = YOLO(w)
print("model.nc =", m.model.nc)
print("len(names) =", len(m.names))
print("names head =", list(m.names.items())[:10])
