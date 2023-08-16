pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
mkdir -p ocrCls/new_model_dir
mkdir -p ocrDetect/new_model_dir
mkdir -p ocrRecog/new_model_dir
sh scripts/all.sh