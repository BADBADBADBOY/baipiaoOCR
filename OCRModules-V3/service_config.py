cls_model_file = "./serviceOCRModule/ocrCls/openvino_dir/cls.xml"
detect_model_file = "./serviceOCRModule/ocrDetect/openvino_dir/detect.xml"
recog_model_file = "./serviceOCRModule/ocrRecog/openvino_dir/recog.xml"

recog_keys_file = "./serviceOCRModule/ocrRecog/ppocr_keys_v1.txt"

detect_params = {}
detect_params['thresh'] = 0.3
detect_params['box_thresh'] = 0.6
detect_params['max_candidates'] = 1000 
detect_params['is_poly'] = False
detect_params['unclip_ratio'] = 1.5
detect_params['min_size'] = 5