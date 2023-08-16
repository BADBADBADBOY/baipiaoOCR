
import argparse
import onnx
import torch
import onnxruntime as ort
import cv2
import os
ort.set_default_logger_severity(3)
from openvino.runtime import Core, AsyncInferQueue
from openvino.inference_engine import IECore
from ocrCls.infer import ppCls
from ocrDetect.infer import ppDetect
from ocrRecog.infer import ppRecog
from config import cls_model_file,detect_model_file,recog_model_file,recog_keys_file,detect_params,cls_test_img,detect_test_img,recog_test_img


def torch2onnx(args):
    
    input_names = ["input"]
    output_names = ["out"]
    
    if args.algorithm=='cls':
        output_onnx = "./ocrCls/new_model_dir/cls.onnx"
        ppcls_bin = ppCls(cls_model_file)
        model = ppcls_bin.model
        img = cv2.imread(cls_test_img)
        img,torch_out = ppcls_bin.onnx_cls_img(img)
        if args.dynamic:
            dynamic = {'input': {0:'batch',2: 'height',3:'width'}} 
            dynamic['out'] = {0: 'batch',1: 'channel'}
        else:
            dynamic = None
        
    elif args.algorithm=='detect':
        ppdetect_bin = ppDetect(detect_model_file,detect_params)
        output_onnx = "./ocrDetect/new_model_dir/detect.onnx"
        model = ppdetect_bin.model
        img = cv2.imread(detect_test_img)
        img,torch_out = ppdetect_bin.onnx_det_img(img)
        if args.dynamic:
            dynamic = {'input': {0:'batch',2: 'height',3:'width'}} 
            dynamic['out'] = {0:'batch',1: 'channel',2: 'height',3:'width'}
        else:
            dynamic = None

    elif args.algorithm=='recog':
        pprecog_bin = ppRecog(recog_model_file,recog_keys_file)
        output_onnx = "./ocrRecog/new_model_dir/recog.onnx"
        model = pprecog_bin.model
        img = cv2.imread(recog_test_img)
        img,torch_out = pprecog_bin.onnx_recog_img(img)
        if args.dynamic:
            dynamic = {'input': {0:'batch',2: 'height',3:'width'}} 
            dynamic['out'] = {0:'batch',1: 'time_step',2:'channel'}
        else:
            dynamic = None
    else:
        assert 1==1,print("not support this algorithm !!! ")
    
    torch.onnx._export(model, torch.Tensor(img), output_onnx, opset_version=11,verbose=False,do_constant_folding=True,
                               input_names=input_names, output_names=output_names,dynamic_axes=dynamic)
    if args.simplify:
        if args.algorithm=='cls':
            os.system("python3 -m onnxsim ./ocrCls/new_model_dir/cls.onnx ./ocrCls/new_model_dir/cls_sim.onnx")
        elif args.algorithm=='detect':
            os.system("python3 -m onnxsim ./ocrDetect/new_model_dir/detect.onnx ./ocrDetect/new_model_dir/detect_sim.onnx")
        elif args.algorithm=='recog':
            os.system("python3 -m onnxsim ./ocrRecog/new_model_dir/recog.onnx ./ocrRecog/new_model_dir/recog_sim.onnx")
            
    if args.algorithm=='cls':
        session = ort.InferenceSession("./ocrCls/new_model_dir/cls.onnx")
    elif args.algorithm=='detect':
        session = ort.InferenceSession("./ocrDetect/new_model_dir/detect.onnx")
    elif args.algorithm=='recog':
        session = ort.InferenceSession("./ocrRecog/new_model_dir/recog.onnx")      

    preds = session.run(output_names, {'input': img})
    onnx_output = preds[0]

    print('torch2onnx sum_error_output:{},mean_error_output:{}'.format((onnx_output-torch_out).sum(),(onnx_output-torch_out).mean()))
    
    if args.openvino:
        onnx2openvino(args,img,onnx_output,torch_out)
    
    

def onnx2openvino(args,img,onnx_output,torch_output):
    if args.algorithm=='cls':
        os.system("python {}/site-packages/openvino/tools/mo/mo.py --input_model ./ocrCls/new_model_dir/cls.onnx  --input_shape [-1,3,-1,-1]   --output_dir ./ocrCls/new_model_dir/openvino_dir".format(args.openvino_install_dir))
        openvino_model = "./ocrCls/new_model_dir/openvino_dir/cls.xml"
    elif args.algorithm=='detect':
        os.system("python {}/site-packages/openvino/tools/mo/mo.py --input_model ./ocrDetect/new_model_dir/detect.onnx  --input_shape [-1,3,-1,-1]  --output_dir ./ocrDetect/new_model_dir/openvino_dir".format(args.openvino_install_dir))
        openvino_model = "./ocrDetect/new_model_dir/openvino_dir/detect.xml"
    elif args.algorithm=='recog':
        os.system("python {}/site-packages/openvino/tools/mo/mo.py --input_model ./ocrRecog/new_model_dir/recog.onnx  --input_shape [-1,3,-1,-1]   --output_dir ./ocrRecog/new_model_dir/openvino_dir".format(args.openvino_install_dir))
        openvino_model = "./ocrRecog/new_model_dir/openvino_dir/recog.xml"
    
    ie = Core()
    model_ir = ie.read_model(model=openvino_model)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU", config={"PERFORMANCE_HINT":"LATENCY","CPU_RUNTIME_CACHE_CAPACITY":"0"})

    infer_request = compiled_model_ir.create_infer_request()
    infer_request.infer([img])
    openvino_output = infer_request.get_output_tensor(0).data
    print('onnx2openvino sum_error_output:{},mean_error_output:{}'.format((onnx_output-openvino_output).sum(),(onnx_output-openvino_output).mean()))
    print('torch2openvino sum_error_output:{},mean_error_output:{}'.format((torch_output-openvino_output).sum(),(torch_output-openvino_output).mean()))

    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="", help='algorithm type')
    parser.add_argument('--dynamic', default=False,action='store_true',help='Does the generated onnx model support dynamic dimensions')
    parser.add_argument('--simplify',default=False,action='store_true', help='Simplify the onnx model')
    parser.add_argument('--openvino', default=False,action='store_true',help='Does the generated onnx model support dynamic dimensions')
    parser.add_argument('--openvino_install_dir', required=True,help='Does the generated onnx model support dynamic dimensions')
    args = parser.parse_args()
    torch2onnx(args)