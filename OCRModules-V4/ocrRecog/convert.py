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

from ocrRecog.backbone import PPLCNetV3
from ocrRecog.neck import SequenceEncoder
from ocrRecog.head import CTCHead


class ppOCRv4RecogModel(nn.Module):
    def __init__(self,n_classes=6625):
        super(ppOCRv4RecogModel,self).__init__()
        self.ppocrv4_backbone = PPLCNetV3(scale = 0.95)
        self.ppocrv4_neck = SequenceEncoder(480)
        self.ppocrv4_head = CTCHead(120,n_classes)
    def forward(self,x):
        x = self.ppocrv4_backbone(x)
        x = self.ppocrv4_neck(x)
        x = self.ppocrv4_head(x)
        return x

def convert_repmodel(model):
    for module in model.modules():
        if hasattr(module, 'rep'):
            module.rep()
    return model
    
def weights_paddle2torch(paddle_weights_path,torch_model,new_model_name):
    
    import torch
#     import paddle
#     paddle_weights = paddle.load(paddle_weights_path)
    
    import pickle
    with open(paddle_weights_path, 'rb') as file:
        paddle_weights = pickle.load(file)
    
    torch_weights = {}
    
    backbone_num = 0
    head_num = 0
    neck_num = 0

    for key,value in torch_model.state_dict().items():
        if 'num_batches_tracked' in key:
            continue 
        if 'ppocrv4_backbone.' in key:
            paddle_key = key.replace('ppocrv4_backbone.','backbone.')
            if 'running_mean' in paddle_key:
                paddle_key = paddle_key.replace('.running_mean','._mean')
            elif 'running_var' in paddle_key:
                paddle_key = paddle_key.replace('.running_var','._variance')
            
            elif '.act.lab.scale' in paddle_key:
                paddle_key = paddle_key.replace('.act.lab.scale','.activation.scale')
            elif '.act.lab.bias' in paddle_key:
                paddle_key = paddle_key.replace('.act.lab.bias','.activation.bias')
                
            elif '.lab.scale' in paddle_key:
                paddle_key = paddle_key.replace('.lab.scale','.w')
            elif '.lab.bias' in paddle_key:
                paddle_key = paddle_key.replace('.lab.bias','.b')
            if paddle_key in paddle_weights.keys():
                backbone_num+=1
                if 'fc' in paddle_key and 'weight' in paddle_key:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key]).transpose(1,0)
                else:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key])
            else:
                print("out {}".format(paddle_key))
                
        elif 'ppocrv4_neck.' in key:
            paddle_key = key.replace('ppocrv4_neck.','head.ctc_encoder.')
            
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
            else:
                print("out {}".format(paddle_key))
                
        elif 'ppocrv4_head.' in key:
            paddle_key = key.replace('ppocrv4_head.','head.ctc_head.')
            if paddle_key in paddle_weights.keys():
                head_num+=1
                if 'fc' in paddle_key and 'weight' in paddle_key:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key]).transpose(1,0)
                else:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key])
            else:
                print("out {}".format(paddle_key))
            
    print("paddle_weights_num:{},torch_weights_num:{}".format(len(paddle_weights),backbone_num+head_num+neck_num))
    torch_model.load_state_dict(torch_weights) 
    torch.save(torch_weights,'{}'.format(new_model_name))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_model_file', type=str, default="", help='Model file that needs to be converted')
    parser.add_argument('--new_model_file', type=str, default="", help='The location of the model file after conversion')
    parser.add_argument('--n_classes', type=int, default=6625 , help='model classes')
    args = parser.parse_args()
    model = ppOCRv4RecogModel(n_classes = args.n_classes)
    model = convert_repmodel(model)
    weights_paddle2torch(args.origin_model_file,model,args.new_model_file)



