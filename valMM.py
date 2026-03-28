from ultralytics import YOLOMM
import os


def main():
    # 可选：清除可能冲突的环境变量
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    model = YOLOMM('ResTest/r18-mm-mid-aifi-msla/weights/best.pt')
    model.val(
        data='D:\\BaiduNetdiskDownload\\M3FD\\M3FD_split\\data.yaml',
        split='val',
        device=0,  # 使用 CPU
        # modality='rgb+ir',

        project='ResTest',
        name='Val'
    )


if __name__ == '__main__':
    main()