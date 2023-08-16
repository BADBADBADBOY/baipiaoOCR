import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import math
import numpy as np
import argparse
from ocrDetect.backbone import PPLCNetV3
from ocrDetect.head import DBHead
from ocrDetect.neck import RSEFPN
import torch
import torch.nn as nn
from ocrDetect.postprocess import DBPostProcess


class ppOCRv3DetectModel(nn.Module):
    def __init__(self,):
        super(ppOCRv3DetectModel,self).__init__()
        self.db_backbone = PPLCNetV3(scale=0.75,det=True)
        self.db_neck = RSEFPN([12,18,42,360],96)
        self.db_head = DBHead(96)
        
    def forward(self,x):
        x = self.db_backbone(x)
        x = self.db_neck(x)
        x = self.db_head(x)
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
    scale_num1 = 0
    scale_num2 = 0
    p_num1 = 0
    p_num2 = 0
    
    for k in paddle_weights.keys():
        if '.pw_conv.w' in k or '.pw_conv.b' in k or '.dw_conv.w' in k or '.dw_conv.b' in k:
            p_num1+=1
        if 'pw_conv.activation.scale' in k or 'pw_conv.activation.bias' in k or 'dw_conv.activation.scale' in k or 'dw_conv.activation.bias' in k:
            p_num2+=1
    for key,value in torch_model.state_dict().items():
        if 'pw_conv.act.lab.bias' in key or 'pw_conv.act.lab.scale' in key or 'dw_conv.act.lab.bias' in key or 'dw_conv.act.lab.scale' in key:
            scale_num1+=1
        if 'pw_conv.lab.bias' in key or 'pw_conv.lab.scale' in key or 'dw_conv.lab.bias' in key or 'dw_conv.lab.scale' in key:
            scale_num2+=1

        if 'num_batches_tracked' in key:
            continue 
        if 'db_backbone.' in key:
            paddle_key = key.replace('db_backbone.','backbone.')
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
#                 print("in paddle_key:{},torch_key:{}".format(paddle_key,key))
            else:
                if 'conv_kxk' in key or 'conv_1x1' in key or 'identity' in key:
                    continue
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
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_model_file', type=str, default="", help='Model file that needs to be converted')
    parser.add_argument('--new_model_file', type=str, default="", help='The location of the model file after conversion')
    args = parser.parse_args()
    model = ppOCRv3DetectModel()
    model = convert_repmodel(model)
    weights_paddle2torch(args.origin_model_file,model,args.new_model_file)
