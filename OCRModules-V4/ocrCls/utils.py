import cv2
import math
import numpy as np

def resize_norm_img(img):
    imgC, imgH, imgW = [3,48,192]
    limited_max_width = 1048
    limited_min_width = 16
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    imgW = max(min(imgW, limited_max_width),limited_min_width)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH,limited_min_width)
    if ratio_imgH > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if len(img.shape) == 2:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im