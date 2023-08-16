mkdir ./serviceOCRModule
mkdir ./serviceOCRModule/ocrCls
mkdir ./serviceOCRModule/ocrDetect
mkdir ./serviceOCRModule/ocrRecog

mkdir ./serviceOCRModule/ocrCls/openvino_dir
mkdir ./serviceOCRModule/ocrDetect/openvino_dir
mkdir ./serviceOCRModule/ocrRecog/openvino_dir

cp ./ocrCls/*.py ./serviceOCRModule/ocrCls/
cp ./ocrDetect/*.py ./serviceOCRModule/ocrDetect/
cp ./ocrRecog/*.py ./serviceOCRModule/ocrRecog/
cp ./ocrRecog/keyFiles/ppocr_keys_v1.txt  ./serviceOCRModule/ocrRecog/

cp ./ocrCls/new_model_dir/openvino_dir/*  ./serviceOCRModule/ocrCls/openvino_dir
cp ./ocrDetect/new_model_dir/openvino_dir/*  ./serviceOCRModule/ocrDetect/openvino_dir
cp ./ocrRecog/new_model_dir/openvino_dir/* ./serviceOCRModule/ocrRecog/openvino_dir

cp ./service_config.py ./serviceOCRModule
cp ./ocrCls/__init__.py ./serviceOCRModule
