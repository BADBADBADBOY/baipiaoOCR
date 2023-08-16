import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import math
import numpy as np
import argparse
from ocrDetect.backbone import MobileNetV3
from ocrDetect.head import DBHead
from ocrDetect.neck import RSEFPN
import torch
import torch.nn as nn
from ocrDetect.postprocess import DBPostProcess


class ppOCRv3DetectModel(nn.Module):
    def __init__(self,):
        super(ppOCRv3DetectModel,self).__init__()
        self.db_backbone = MobileNetV3()
        self.db_neck = RSEFPN([16,24,56,480],96)
        self.db_head = DBHead(96)
        
    def forward(self,x):
        x = self.db_backbone(x)
        x = self.db_neck(x)
        x = self.db_head(x)
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
        if 'db_backbone.' in key:
            paddle_key = key.replace('db_backbone.','backbone.')
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
            else:
                print("out paddle_key:{},torch_key:{}".format(paddle_key,key))
        
        elif 'db_neck.' in key:
            paddle_key = key.replace('db_neck.','neck.')
            
            if 'running_mean' in paddle_key:
                paddle_key = paddle_key.replace('.running_mean','._mean')
            if 'running_var' in paddle_key:
                paddle_key = paddle_key.replace('.running_var','._variance')
            
            if paddle_key in paddle_weights.keys():
                neck_num+=1
                if 'fc' in paddle_key and 'weight' in paddle_key:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key]).transpose(1,0)
                else:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key])
            else:
                print(" out paddle_key:{},torch_key:{}".format(paddle_key,key))
        
        elif 'db_head.' in key:
            paddle_key = key.replace('db_head.','head.')
            
            if 'running_mean' in paddle_key:
                paddle_key = paddle_key.replace('.running_mean','._mean')
            if 'running_var' in paddle_key:
                paddle_key = paddle_key.replace('.running_var','._variance')
            
            if paddle_key in paddle_weights.keys():
                head_num+=1
                if 'fc' in paddle_key and 'weight' in paddle_key:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key]).transpose(1,0)
                else:
                    torch_weights[key] = torch.Tensor(paddle_weights[paddle_key])
            else:
                print("out paddle_key:{},torch_key:{}".format(paddle_key,key))
                
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
            key_new = key.replace('backbone.','db_backbone.')
            key_new = key_new.replace('stages.0.','stage0.').replace('stages.1.','stage1.').replace('stages.2.','stage2.').replace('stages.3.','stage3.')    
            if key_new in model_keys:
                new_dict[key_new] = torch_weights[key]
                total_num+=1
        if 'neck.' in key:
            key_new = key.replace('neck.','db_neck.')
            if key_new in model_keys:
                new_dict[key_new] = torch_weights[key]
                total_num+=1
        if 'head.' in key:
            key_new = key.replace('head.','db_head.')
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
    args = parser.parse_args()
    model = ppOCRv3DetectModel()
    if args.origin_model_file.endswith('.pth'):
        weights_torch2torch(args.origin_model_file,model,args.new_model_file)
    else:
        weights_paddle2torch(args.origin_model_file,model,args.new_model_file)
