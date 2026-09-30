#这是提供 参考示例的 YOLOMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM(r'D:\BaiduNetdiskDownload\MutilModel_3398475911\ultralytics\cfg\models\rtmm\r18\baseline_plus_csp.yaml')
    model.train(data=r'D:\BaiduNetdiskDownload\M3FD_split\data.yaml',
                epochs=100,batch=8,
                # modality='X', 模态消融参数 非必要不得开启
                # cache=True,
                exist_ok=True,
                project='D:/JiQI/MM-experiment/ResTest',name='Test/YOLOMM')
