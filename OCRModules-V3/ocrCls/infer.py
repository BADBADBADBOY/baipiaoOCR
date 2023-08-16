import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from ocrCls.backbone import MobileNetV3
from ocrCls.head import ClsHead
from ocrCls.utils import resize_norm_img

class ppOCRv3ClsModel(nn.Module):
    def __init__(self,n_classes=2):
        super(ppOCRv3ClsModel,self).__init__()
        self.ppocrcls_backbone = MobileNetV3()
        self.ppocrcls_head = ClsHead(200,n_classes)
    def forward(self,x):
        x = self.ppocrcls_backbone(x)
        x = self.ppocrcls_head(x)
        return x

class ppCls(object):
    def __init__(self,model_path,use_cuda=False):
        super(ppCls,self).__init__()
        model = ppOCRv3ClsModel(n_classes=2)
        weights = torch.load(model_path,map_location='cpu')
        model.load_state_dict(weights)
        if use_cuda:
            model = model.cuda()
            self.use_cuda = use_cuda
        model.eval()
        self.model = model
        self.angles = ['0','180']
    
    def cls_img(self,img):
        img_ori = img.copy()
        img = resize_norm_img(img)
        img = torch.Tensor(img).unsqueeze(0)
        if hasattr(self,'use_cuda') and self.use_cuda:
            img = img.cuda()
        with torch.no_grad():
            out = self.model(img)
        index = out[0].argmax().item()
        if index==1:
            img_ori = cv2.rotate(img_ori,1)
        return img_ori,self.angles[index],out[0][index].item()
    
    def onnx_cls_img(self,img):
        img_ori = img.copy()
        img = resize_norm_img(img)
        img = torch.Tensor(img).unsqueeze(0)
        with torch.no_grad():
            out = self.model(img)
        return img.numpy(),out.numpy()

class ppClsOnnx(object):
    def __init__(self,model_path):
        super(ppClsOnnx,self).__init__()
        
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(model_path)
        
        self.angles = ['0','180']
    
    def cls_img(self,img):
        img_ori = img.copy()
        img = resize_norm_img(img)
        img = img[np.newaxis, :]
        
        preds = self.session.run(["out"], {'input': img})
        out = preds[0]
        
        index = out[0].argmax().item()
        if index==1:
            img_ori = cv2.rotate(img_ori,1)
        return img_ori,self.angles[index],out[0][index].item()
    
class ppClsOpenvino(object):
    def __init__(self,model_path):
        super(ppClsOpenvino,self).__init__()
        
        from openvino.runtime import Core, AsyncInferQueue
        ie = Core()
        model_ir = ie.read_model(model=model_path)
        compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU", config={"PERFORMANCE_HINT":"LATENCY","CPU_RUNTIME_CACHE_CAPACITY":"0"})
        self.infer_request = compiled_model_ir.create_infer_request()
        
        self.angles = ['0','180']
    
    def cls_img(self,img):
        img_ori = img.copy()
        img = resize_norm_img(img)
        img = img[np.newaxis, :]
        self.infer_request.infer([img])
        out = self.infer_request.get_output_tensor(0).data
        index = out[0].argmax().item()
        if index==1:
            img_ori = cv2.rotate(img_ori,1)
        return img_ori,self.angles[index],out[0][index].item()