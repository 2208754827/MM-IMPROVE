from ultralytics import YOLOMM
import os


def main():
    # 可选：清除可能冲突的环境变量
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    model = YOLOMM('D:\JiQI\MM-experiment\ResTest\LLVIP-UP\weights\best.pt')
    model.val(
        data='D:\BaiduNetdiskDownload\LLVIP_mm\data.yaml',
        split='test',
        device=0,  # 使用 CPU
        # modality='rgb+ir',

        project='D:/JiQI/MM-experiment/ResTest',
        name='Val'
    )


if __name__ == '__main__':
    main()