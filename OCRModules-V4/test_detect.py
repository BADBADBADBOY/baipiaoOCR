import cv2
import time
import numpy as np
from ocrDetect.infer import ppDetect,ppDetectOnnx,ppDetectOpenvino
from config import detect_params

img_file = "./imgtestFile/detect.jpg"


torch_model_path = "./ocrDetect/new_model_dir/detect.pth"
onnx_model_path = "./ocrDetect/new_model_dir/detect.onnx"
openvino_model_path = "./ocrDetect/new_model_dir/openvino_dir/detect.xml"

# torch_cuda_ppdetect = ppDetect(torch_model_path,detect_params,use_cuda=True)
torch_ppdetect = ppDetect(torch_model_path,detect_params)
onnx_ppdetect = ppDetectOnnx(onnx_model_path,detect_params)
openvino_ppdetect = ppDetectOpenvino(openvino_model_path,detect_params)

test_num = 10

img_ori = cv2.imread(img_file) 

torch_time = 0
for i in range(test_num):
    start = time.time()
    bbox_batch = torch_ppdetect.det_img(img_ori,736)
    end = time.time()
    torch_time+=end-start
    img_show = img_ori.copy()
    for bbox in bbox_batch:
        bbox = bbox.reshape(-1, 2).astype(np.int)
        img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
    cv2.imwrite('detect_show_torch.jpg',img_show)
    
# torch_cuda_time = 0
# for i in range(test_num):
#     start = time.time()
#     bbox_batch = torch_cuda_ppdetect.det_img(img_ori,736)
#     end = time.time()
#     torch_cuda_time+=end-start
#     img_show = img_ori.copy()
#     for bbox in bbox_batch:
#         bbox = bbox.reshape(-1, 2).astype(np.int)
#         img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
#     cv2.imwrite('detect_show_torch_cuda.jpg',img_show)
    
onnx_time = 0
for i in range(test_num):
    start = time.time()
    bbox_batch = onnx_ppdetect.det_img(img_ori,736)
    end = time.time()
    onnx_time+=end-start
    img_show = img_ori.copy()
    for bbox in bbox_batch:
        bbox = bbox.reshape(-1, 2).astype(np.int)
        img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
    cv2.imwrite('detect_show_onnx.jpg',img_show)
    
openvino_time = 0
for i in range(test_num):
    start = time.time()
    bbox_batch = openvino_ppdetect.det_img(img_ori,736)
    end = time.time()
    openvino_time+=end-start
    img_show = img_ori.copy()
    for bbox in bbox_batch:
        bbox = bbox.reshape(-1, 2).astype(np.int)
        img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
    cv2.imwrite('detect_show_openvino.jpg',img_show)
    
print("torch_time_avg:{},onnx_time_avg:{},openvino_time_avg:{}".format(torch_time/test_num,onnx_time/test_num,openvino_time/test_num))
# print("torch_time_avg:{},torch_cuda_time_avg:{},onnx_time_avg:{},openvino_time_avg:{}".format(torch_time/test_num,torch_cuda_time/test_num,onnx_time/test_num,openvino_time/test_num))
