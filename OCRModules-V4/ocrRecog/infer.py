import torch
import torch.nn as nn
import cv2
import os
import sys
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from ocrRecog.backbone import PPLCNetV3
from ocrRecog.neck import SequenceEncoder
from ocrRecog.head import CTCHead
from ocrRecog.utils import resize_norm_img,CTCLabelDecode


def convert_repmodel(model):
    for module in model.modules():
        if hasattr(module, 'rep'):
            module.rep()
    return model


class ppOCRv4RecogModel(nn.Module):
    def __init__(self,n_classes=6625):
        super(ppOCRv4RecogModel,self).__init__()
        self.ppocrv4_backbone = convert_repmodel(PPLCNetV3(scale = 0.95))
        self.ppocrv4_neck = SequenceEncoder(480)
        self.ppocrv4_head = CTCHead(120,n_classes)
    def forward(self,x):
        x = self.ppocrv4_backbone(x)
        x = self.ppocrv4_neck(x)
        x = self.ppocrv4_head(x)
        return x

class ppRecog(object):
    def __init__(self,model_path,keys_file,use_space_char=False,use_cuda=False):
        super(ppRecog,self).__init__()
        model = ppOCRv4RecogModel(n_classes=6625)
        weights = torch.load(model_path,map_location='cpu')
        model.load_state_dict(weights)
        if use_cuda:
            model = model.cuda()
            self.use_cuda = use_cuda
        model.eval()
        self.model = model
        self.decode_ctc = CTCLabelDecode(keys_file,use_space_char=use_space_char)
        
    def recog_img(self,img):
        max_wh_ratio = img.shape[1]/img.shape[0]
        resized_image = resize_norm_img(img,max_wh_ratio)
        resized_image = torch.from_numpy(resized_image).unsqueeze(0)
        if hasattr(self,'use_cuda') and self.use_cuda:
            resized_image = resized_image.cuda()
        with torch.no_grad():
            out = self.model(resized_image)
        text = self.decode_ctc(out)
        return text[0]
    
    def onnx_recog_img(self,img):
        max_wh_ratio = img.shape[1]/img.shape[0]
        img = resize_norm_img(img,max_wh_ratio)
        img = torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            out = self.model(img)
        return img.numpy(),out.numpy()


class ppRecogOnnx(object):
    def __init__(self,model_path,keys_file,use_space_char=False):
        super(ppRecogOnnx,self).__init__()
        
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(model_path)
        
        self.decode_ctc = CTCLabelDecode(keys_file,use_space_char=use_space_char)
        
    def recog_img(self,img):
        max_wh_ratio = img.shape[1]/img.shape[0]
        img = resize_norm_img(img,max_wh_ratio)
        img = img[np.newaxis, :]
        preds = self.session.run(["out"], {'input': img})
        out = preds[0]
        text = self.decode_ctc(out)
        return text[0]


class ppRecogOpenvino(object):
    def __init__(self,model_path,keys_file,use_space_char=False):
        super(ppRecogOpenvino,self).__init__()
        from openvino.runtime import Core, AsyncInferQueue
        ie = Core()
        model_ir = ie.read_model(model=model_path)
        compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU", config={"PERFORMANCE_HINT":"LATENCY","CPU_RUNTIME_CACHE_CAPACITY":"0"})
        self.infer_request = compiled_model_ir.create_infer_request()
        self.decode_ctc = CTCLabelDecode(keys_file,use_space_char=use_space_char)
        
    def recog_img(self,img):
        max_wh_ratio = img.shape[1]/img.shape[0]
        img = resize_norm_img(img,max_wh_ratio)
        img = img[np.newaxis, :]
        self.infer_request.infer([img])
        out = self.infer_request.get_output_tensor(0).data
        text = self.decode_ctc(out)
        return text[0]