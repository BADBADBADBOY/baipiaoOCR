cls_model_file = "./ocrCls/new_model_dir/cls.pth"
detect_model_file = "./ocrDetect/new_model_dir/detect.pth"
recog_model_file = "./ocrRecog/new_model_dir/recog.pth"

recog_keys_file = "./ocrRecog/keyFiles/ppocr_keys_v1.txt"


cls_test_img = "./imgtestFile/cls.jpg"
detect_test_img = "./imgtestFile/detect.jpg"
recog_test_img = "./imgtestFile/recog.jpg"


detect_params = {}
detect_params['thresh'] = 0.3
detect_params['box_thresh'] = 0.6
detect_params['max_candidates'] = 1000 
detect_params['is_poly'] = False
detect_params['unclip_ratio'] = 1.5
detect_params['min_size'] = 5