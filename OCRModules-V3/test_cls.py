import cv2
import time
from ocrCls.infer import ppCls,ppClsOnnx,ppClsOpenvino


img_file = "./imgtestFile/cls.jpg" 

torch_model_file = "./ocrCls/new_model_dir/cls.pth"
onnx_model_file = "./ocrCls/new_model_dir/cls.onnx"
openvino_model_file = "./ocrCls/new_model_dir/openvino_dir/cls.xml"

# torch_cuda_ppcls_bin = ppCls(torch_model_file,use_cuda=True)
torch_ppcls_bin = ppCls(torch_model_file)
onnx_ppcls_bin = ppClsOnnx(onnx_model_file)
openvino_ppcls_bin = ppClsOpenvino(openvino_model_file)


img = cv2.imread(img_file)
test_num = 10

torch_time = 0
for i in range(test_num):
    start = time.time()
    _,cls,cls_conf = torch_ppcls_bin.cls_img(img)
    end = time.time()
    torch_time+=end-start
print(cls,cls_conf)

# torch_cuda_time = 0
# for i in range(test_num):
#     start = time.time()
#     _,cls,cls_conf = torch_cuda_ppcls_bin.cls_img(img)
#     end = time.time()
#     torch_cuda_time+=end-start
# print(cls,cls_conf)
    
onnx_time = 0
for i in range(test_num):
    start = time.time()
    _,cls,cls_conf = onnx_ppcls_bin.cls_img(img)
    end = time.time()
    onnx_time+=end-start
print(cls,cls_conf)
    
openvino_time = 0
for i in range(test_num):
    start = time.time()
    _,cls,cls_conf = openvino_ppcls_bin.cls_img(img)
    end = time.time()
    openvino_time+=end-start
print(cls,cls_conf)
print("torch_time_avg:{},onnx_time_avg:{},openvino_time_avg:{}".format(torch_time/test_num,onnx_time/test_num,openvino_time/test_num))
# print("torch_time_avg:{},torch_cuda_time_avg:{},onnx_time_avg:{},openvino_time_avg:{}".format(torch_time/test_num,torch_cuda_time/test_num,onnx_time/test_num,openvino_time/test_num))

