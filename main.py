#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO, YOLO_ONNX

if __name__ == "__main__":
    img_path='img/street++.jpg'
    yolo = YOLO()
    img = img_path
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image, crop = False, count=False)
        #r_image.show()
        r_image.save("result.jpg")