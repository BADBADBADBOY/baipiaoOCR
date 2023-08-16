import cv2
import math
import numpy as np
from serviceOCRModule.ocrCls.infer import ppClsOpenvino
from serviceOCRModule.ocrDetect.infer import ppDetectOpenvino
from serviceOCRModule.ocrRecog.infer import ppRecogOpenvino
from serviceOCRModule.service_config import cls_model_file,detect_model_file,recog_model_file,recog_keys_file,detect_params


## 计算欧式距离
def cal_distance(coord1,coord2):
    return math.sqrt((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2)

## 得到文字的长和宽
def cal_width_height(bbox):
    width = cal_distance((bbox[0],bbox[1]),(bbox[2],bbox[3]))
    height = cal_distance((bbox[2],bbox[3]),(bbox[4],bbox[5]))
    return int(width),int(height)

def get_perspective_image(image,bbox):
    width,height = cal_width_height(bbox)
    if height>width:
        pts1 = np.float32([[0,0],[height,0],[height,width],[0,width]])
        pts2 = np.float32(np.array([bbox[2],bbox[3],bbox[4],bbox[5],bbox[6],bbox[7],bbox[0],bbox[1]]).reshape(4,2))
        width,height = height,width
    else:
        pts1 = np.float32([[0,0],[width,0],[width,height],[0,height]])
        pts2 = np.float32(bbox.reshape(4,2))
    M = cv2.getPerspectiveTransform(pts2,pts1)
    dst = cv2.warpPerspective(image,M,(width,height))
    return dst


ppcls_bin = ppClsOpenvino(cls_model_file)
ppdetect_bin = ppDetectOpenvino(detect_model_file,detect_params)
pprecog_bin = ppRecogOpenvino(recog_model_file,recog_keys_file)

img = cv2.imread("./imgtestFile/detect.jpg")
bbox_batch = ppdetect_bin.det_img(img,736)

img_show = img.copy()
for bbox in bbox_batch:
    bbox = bbox.reshape(-1, 2).astype(np.int)
    img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
cv2.imwrite('detect_service.jpg',img_show)

for box in bbox_batch:
    cut_img = get_perspective_image(img,box.reshape(-1))
    cut_img,cls,cls_conf = ppcls_bin.cls_img(cut_img)
    recog_text = pprecog_bin.recog_img(cut_img)
    print(recog_text)



