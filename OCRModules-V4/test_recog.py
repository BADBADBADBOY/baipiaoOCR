import cv2
import time
from ocrRecog.infer import ppRecog,ppRecogOnnx,ppRecogOpenvino

img_file = "./imgtestFile/recog.jpg"


torch_model_file = "./ocrRecog/new_model_dir/recog.pth"
onnx_model_file = "./ocrRecog/new_model_dir/recog.onnx"
openvino_model_file = "./ocrRecog/new_model_dir/openvino_dir/recog.xml"

keys_file = "./ocrRecog/keyFiles/ppocr_keys_v1.txt"

img = cv2.imread(img_file)

# torch_cuda_pprecog_bin = ppRecog(torch_model_file,keys_file,use_cuda=True)
torch_pprecog_bin = ppRecog(torch_model_file,keys_file)
onnx_pprecog_bin = ppRecogOnnx(onnx_model_file,keys_file)
openvino_pprecog_bin = ppRecogOpenvino(openvino_model_file,keys_file)

test_num = 10

torch_time = 0
for i in range(test_num):
    start = time.time()
    recog_text = torch_pprecog_bin.recog_img(img)
    end = time.time()
    torch_time += end-start
print(recog_text)

# torch_cuda_time = 0
# for i in range(test_num):
#     start = time.time()
#     recog_text = torch_cuda_pprecog_bin.recog_img(img)
#     end = time.time()
#     torch_cuda_time += end-start
# print(recog_text)


onnx_time = 0
for i in range(test_num):
    start = time.time()
    recog_text = onnx_pprecog_bin.recog_img(img)
    end = time.time()
    onnx_time+=end-start
print(recog_text)
    
    
openvino_time = 0
for i in range(test_num):
    start = time.time()
    recog_text = openvino_pprecog_bin.recog_img(img)
    end = time.time()
    openvino_time+=end-start
print(recog_text)
    
print("torch_time_avg:{},onnx_time_avg:{},openvino_time_avg:{}".format(torch_time/test_num,onnx_time/test_num,openvino_time/test_num))
# print("torch_time_avg:{},torch_cuda_time_avg:{},onnx_time_avg:{},openvino_time_avg:{}".format(torch_time/test_num,torch_cuda_time/test_num,onnx_time/test_num,openvino_time/test_num))
