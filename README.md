# baipiaoOCR

将 paddleOCR 转 torchOCR, 支持ppocr-v3,ppocr-v4转torch, onnx, openvino。

### 快速开始，运行下面命令，一键转成torch和onnx和openvino，支持ppOCR-v3和ppOCR-v4。

```
cd OCRModules-V3 or cd OCRModules-V4
sh install.sh
```

### 测试

运行完后会生成一个serviceOCRModule模块，基于openvino的这个模块可以直接用于OCR实际应用开发，测试这个模块：
```
python test_serviceOCRModule.py
```

### 其他测试
```
python test_cls.py
python test_detect.py
python test_recog.py
```


<img src="./doc/show/2.jpg" width=600 height=600 />