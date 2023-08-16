
import torch
import torch.nn as nn
import math
import numpy as np
import torchvision.transforms as transforms

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from ocrDetect.backbone import PPLCNetV3
from ocrDetect.head import DBHead
from ocrDetect.neck import RSEFPN
from ocrDetect.postprocess import DBPostProcess
from ocrDetect.utils import resize_image,post_img

def convert_repmodel(model):
    for module in model.modules():
        if hasattr(module, 'rep'):
            module.rep()
    return model


class ppOCRv4DetectModel(nn.Module):
    def __init__(self,):
        super(ppOCRv4DetectModel,self).__init__()
        self.db_backbone = convert_repmodel(PPLCNetV3(scale=0.75,det=True))
        self.db_neck = RSEFPN([12,18,42,360],96)
        self.db_head = DBHead(96)
        
    def forward(self,x):
        x = self.db_backbone(x)
        x = self.db_neck(x)
        x = self.db_head(x)
        return x

class ppDetect(object):
    def __init__(self,model_file,params,use_cuda=False):
        super(ppDetect,self).__init__()
        model = ppOCRv4DetectModel()
        model.load_state_dict(torch.load(model_file,map_location='cpu'))
        if use_cuda:
            model = model.cuda()
            self.use_cuda = use_cuda
        model.eval()
        self.model = model
        self.dbprocess = DBPostProcess(params)
    
    def det_img(self,img_ori,shortest=736):
        img = post_img(img_ori,shortest)
        if hasattr(self,'use_cuda') and self.use_cuda:
            img = img.cuda()
        with torch.no_grad():
            out = self.model(img)
        scale = (img_ori.shape[1] * 1.0 / out.shape[3], img_ori.shape[0] * 1.0 / out.shape[2])       
        bbox_batch,score_batch = self.dbprocess(out.cpu().numpy(),[scale])
        return bbox_batch[0]
    
    def onnx_det_img(self,img_ori,shortest=736):
        img = post_img(img_ori,shortest)
        with torch.no_grad():
            out = self.model(img)
        return img.numpy(),out.numpy()
    
class ppDetectOnnx(object):
    def __init__(self,model_file,params):
        super(ppDetectOnnx,self).__init__()
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(model_file)
        self.dbprocess = DBPostProcess(params)
    
    def det_img(self,img_ori,shortest=736):
        img = post_img(img_ori,shortest)
        img = img.numpy()
        preds = self.session.run(["out"], {'input': img})
        out = preds[0]
        scale = (img_ori.shape[1] * 1.0 / out.shape[3], img_ori.shape[0] * 1.0 / out.shape[2])       
        bbox_batch,score_batch = self.dbprocess(out,[scale])
        return bbox_batch[0]
    
class ppDetectOpenvino(object):
    def __init__(self,model_file,params):
        super(ppDetectOpenvino,self).__init__()
        from openvino.runtime import Core, AsyncInferQueue
        ie = Core()
        model_ir = ie.read_model(model=model_file)
        compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU", config={"PERFORMANCE_HINT":"LATENCY","CPU_RUNTIME_CACHE_CAPACITY":"0"})
        self.infer_request = compiled_model_ir.create_infer_request()
        
        self.dbprocess = DBPostProcess(params)
    
    def det_img(self,img_ori,shortest=736):
        img = post_img(img_ori,shortest)
        img = img.numpy()
        self.infer_request.infer([img])
        out = self.infer_request.get_output_tensor(0).data
        scale = (img_ori.shape[1] * 1.0 / out.shape[3], img_ori.shape[0] * 1.0 / out.shape[2])       
        bbox_batch,score_batch = self.dbprocess(out,[scale])
        return bbox_batch[0]
    