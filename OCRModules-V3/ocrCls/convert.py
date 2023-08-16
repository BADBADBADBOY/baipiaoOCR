import sys
import os
import torch
import torch.nn as nn
import numpy as np

import math
import cv2
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from ocrCls.backbone import MobileNetV3
from ocrCls.head import ClsHead


class ppOCRv3ClsModel(nn.Module):
    def __init__(self,n_classes=2):
        super(ppOCRv3ClsModel,self).__init__()
        self.ppocrcls_backbone = MobileNetV3()
        self.ppocrcls_head = ClsHead(200,n_classes)
    def forward(self,x):
        x = self.ppocrcls_backbone(x)
        x = self.ppocrcls_head(x)
        return x

def weights_paddle2torch(paddle_weights_path,torch_model,new_model_name):
    import paddle
    import torch
    
    paddle_weights = paddle.load(paddle_weights_path)
    torch_weights = {}
    
    backbone_num = 0
    head_num = 0

    for key,value in torch_model.state_dict().items():
        if 'num_batches_tracked' in key:
            continue 
        if 'ppocrcls_backbone.' in key:
            paddle_key = key.replace('ppocrcls_backbone.','backbone.')
            if 'running_mean' in paddle_key:
                paddle_key = paddle_key.replace('.running_mean','._mean')
            if 'running_var' in paddle_key:
                paddle_key = paddle_key.replace('.running_var','._variance')
            if paddle_key in paddle_weights.keys():
                backbone_num+=1
                if 'fc' in paddle_key and 'weight' in paddle_key:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key]).transpose(1,0)
                else:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key])      
        elif 'ppocrcls_head.' in key:
            paddle_key = key.replace('ppocrcls_head.','head.')
            if paddle_key in paddle_weights.keys():
                head_num+=1
            if 'fc' in paddle_key and 'weight' in paddle_key:
                torch_weights[key] = torch.Tensor(paddle_weights[paddle_key]).transpose(1,0)
            else:
                torch_weights[key] = torch.Tensor(paddle_weights[paddle_key])
    print("paddle_weights_num:{},torch_weights_num:{}".format(len(paddle_weights),backbone_num+head_num))
    torch_model.load_state_dict(torch_weights) 
    torch.save(torch_weights,'{}'.format(new_model_name))

def weights_torch2torch(torch_weights_path,torch_model,new_model_name):
    torch_weights = torch.load(torch_weights_path)
    new_dict = {}
    total_num = 0 
    for key in torch_weights.keys():
        if 'backbone' in key:
            key_new = key.replace('backbone.','ppocrcls_backbone.')
            new_dict[key_new] = torch_weights[key]
            total_num+=1
        if 'head.' in key:
            key_new = key.replace('head.','ppocrcls_head.')
            new_dict[key_new] = torch_weights[key]
            total_num+=1
    print("torch_ori_weights_num:{},torch_new_weights_num:{}".format(len(torch_weights),total_num))
    torch_model.load_state_dict(new_dict)
    torch.save(torch_model.state_dict(),'{}'.format(new_model_name))
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_model_file', type=str, default="./ocrCls/origin_model_dir/ch_ptocr_mobile_v2.0_cls_infer.pth", help='Model file that needs to be converted')
    parser.add_argument('--new_model_file', type=str, default="./ocrCls/new_model_dir/cls.pth", help='The location of the model file after conversion')
    parser.add_argument('--n_classes', type=int, default=2 , help='model classes')
    args = parser.parse_args()
    model = ppOCRv3ClsModel(n_classes = args.n_classes)
    if args.origin_model_file.endswith('.pth'):
        weights_torch2torch(args.origin_model_file,model,args.new_model_file)
    else:
        weights_paddle2torch(args.origin_model_file,model,args.new_model_file)
    