### CUDA及CuDnn安装教程：https://blog.csdn.net/jhsignal/article/details/111401628
### 如何检查CUDA及CuDnn是否安装成功：https://blog.csdn.net/jhsignal/article/details/111398427
### TensorRT安装教程：https://blog.csdn.net/a2475865292/article/details/130384976

---

### 1.为什么加载OpenVINO模型时报错?

- 找到“packages”文件夹下“OpenVINO.runtime.win.2024.0.0.2”的dll文件拷贝到Debug根目录下(OpenVINO环境)

### 2.为什么我收到的文件里没有“packages”文件夹？

- 需要打开项目->编译一下项目->自动下载Nuget包就会有packages文件夹，用到的环境都在里头

### 3.为什么yoloV10用不了？

- 由于V10版本推理输出的张量结构与V8 V11这些不同，解析算法不一样，需要针对V10进行修改，目前测试V8、V11可以

### 4.Frameworks环境低于4.7.2为什么不行？

- 由于OpenVINO的库环境需求最低为4.7.2，若不用OpenVINO可以删除库后删除相关代码重新编译

### 5.我的Cuda有安装为什么用不了GPU，一直报错？

- 有可能CUDA版本与OnnxRuntime版本不匹配，可以百度查一下匹配关系，适当降低或者升级OnnxRuntime版本(替换Microsoft.ML.OnnxRuntime.Gpu.Windows与Microsoft.ML.OnnxRuntime.Managed中的库)，参考： https://blog.csdn.net/qq_38308388/article/details/137679214（有测试过的环境搭配：CUDA 12.4、CUDNN 8.9.x 、OnnxRUntime 1.20.0）
  CUDA下载地址：https://developer.nvidia.com/cuda-toolkit-archive   CUDNN下载地址：https://developer.nvidia.com/cudnn-archive

### 6.OpenVINO加载模型时一直报错“OpenVinoSharp.OVException:“Exception from src\inference\src\cpp\core.cpp:92”

- Onnx模型路径不能有中文

### 7.TensorRT安装好后加载不了或者转换不了engine模型，或者加载时报错

- 如果TensorRT完成部署安装后，运行报“找不到TensorRtExtern.dll”时，需要在本机中重新编译TensorRtExtern.dll库

- 克隆guojin-yan大佬的TensorRT-CSharp-API库
```
git clone https://github.com/guojin-yan/TensorRT-CSharp-API.git
```

- 在本机中打开该项目并重新编译，替换重新编译后的库

### 8.固定输入模型与动态输入模型区别

- 固定输入模型：模型的输入大小在模型转换时就已经确定，无法改变。
- 动态输入模型：模型的输入大小在模型转换时不确定，需要在运行时动态指定输入尺寸大小，动态输入尺寸允许导出的模型处理不同的图像尺寸，为不同的使用案例提供灵活性并优化处理效率，支持多张图同时推理。

- python代码中导出模型时添加dynamic = True 表示导出的Onnx模型是否为动态输入模型
```
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", dynamic=True)
```