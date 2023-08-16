# -*- coding: utf-8 -*-
# @Time : 2023/3/13 9:51
# @Author : fangxuwei
# @Github Name : BADBADBADBOY
# @File : ppOCRv3RecogModel
# @Project : SVTR

import argparse
import torch
import torch.nn as nn
import numpy as np
import math
import cv2
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from ocrRecog.backbone import MobileNetV1Enhance
from ocrRecog.neck import SequenceEncoder
from ocrRecog.head import CTCHead


class ppOCRv3RecogModel(nn.Module):
    def __init__(self,n_classes=6625):
        super(ppOCRv3RecogModel,self).__init__()
        self.ppocrv3_backbone = MobileNetV1Enhance()
        self.ppocrv3_neck = SequenceEncoder(512)
        self.ppocrv3_head = CTCHead(64,n_classes)
    def forward(self,x):
        x = self.ppocrv3_backbone(x)
        x = self.ppocrv3_neck(x)
        x = self.ppocrv3_head(x)
        return x

def weights_paddle2torch(paddle_weights_path,torch_model,new_model_name):
    import paddle
    import torch
    
    paddle_weights = paddle.load(paddle_weights_path)
    torch_weights = {}
    
    backbone_num = 0
    head_num = 0
    neck_num = 0

    for key,value in torch_model.state_dict().items():
        if 'num_batches_tracked' in key:
            continue 
        if 'ppocrv3_backbone.' in key:
            paddle_key = key.replace('ppocrv3_backbone.','backbone.')
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
                
        elif 'ppocrv3_neck.' in key:
            paddle_key = key.replace('ppocrv3_neck.','head.ctc_encoder.')
            
            if 'running_mean' in paddle_key:
                paddle_key = paddle_key.replace('.running_mean','._mean')
            if 'running_var' in paddle_key:
                paddle_key = paddle_key.replace('.running_var','._variance')
            
            if paddle_key in paddle_weights.keys():
                neck_num+=1
                if 'fc' in paddle_key and 'weight' in paddle_key or 'qkv.weight' in paddle_key:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key]).transpose(1,0)
                else:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key])
                
        elif 'ppocrv3_head.' in key:
            paddle_key = key.replace('ppocrv3_head.','head.ctc_head.')
            if paddle_key in paddle_weights.keys():
                head_num+=1
            if 'fc' in paddle_key and 'weight' in paddle_key:
                torch_weights[key] = torch.Tensor(paddle_weights[paddle_key]).transpose(1,0)
            else:
                torch_weights[key] = torch.Tensor(paddle_weights[paddle_key])
    print("paddle_weights_num:{},torch_weights_num:{}".format(len(paddle_weights),backbone_num+head_num+neck_num))
    torch_model.load_state_dict(torch_weights) 
    torch.save(torch_weights,'{}'.format(new_model_name))

def weights_torch2torch(torch_weights_path,torch_model,new_model_name):
    torch_weights = torch.load(torch_weights_path)
    new_dict = {}
    total_num = 0 
    model_keys = torch_model.state_dict().keys()
    
    for key in torch_weights.keys():
        if 'backbone.' in key:
            key_new = key.replace('backbone.','ppocrv3_backbone.')
            if key_new in model_keys:
                new_dict[key_new] = torch_weights[key]
                total_num+=1
        if 'neck.' in key:
            key_new = key.replace('neck.','ppocrv3_neck.')
            if key_new in model_keys:
                new_dict[key_new] = torch_weights[key]
                total_num+=1
        if 'head.' in key:
            key_new = key.replace('head.','ppocrv3_head.')
            if key_new in model_keys:
                new_dict[key_new] = torch_weights[key]
                total_num+=1
    print("torch_ori_weights_num:{},torch_new_weights_num:{}".format(len(torch_weights),total_num))
    torch_model.load_state_dict(new_dict)
    torch.save(torch_model.state_dict(),'{}'.format(new_model_name))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_model_file', type=str, default="", help='Model file that needs to be converted')
    parser.add_argument('--new_model_file', type=str, default="", help='The location of the model file after conversion')
    parser.add_argument('--n_classes', type=int, default=6625 , help='model classes')
    args = parser.parse_args()
    model = ppOCRv3RecogModel(n_classes = args.n_classes)
    if args.origin_model_file.endswith('.pth'):
        weights_torch2torch(args.origin_model_file,model,args.new_model_file)
    else:
        weights_paddle2torch(args.origin_model_file,model,args.new_model_file)



