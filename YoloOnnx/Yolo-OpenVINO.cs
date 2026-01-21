using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenVinoSharp;
using Newtonsoft.Json;
using Onnx;
using static System.Collections.Specialized.BitVector32;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing;
using Newtonsoft.Json.Linq;
using System.Diagnostics;
using System.Collections.Concurrent;
using Google.Protobuf.Compiler;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using static System.Net.Mime.MediaTypeNames;

namespace YoloOnnx
{
    public class Yolo_OpenVINO : IDisposable
    {
        public ModelType modelType;
        Core core;
        CompiledModel compiled_model;
        InferRequest infer_request;
        OpenVinoSharp.Tensor inputTensor;
        ModelProto modelProto;
        ModelInfo modelInfo;

        #region 释放资源
        public void Dispose()
        {
            core?.Dispose();
            compiled_model?.Dispose();
            infer_request?.Dispose();
            inputTensor?.Dispose();
        } 
        #endregion

        #region 加载onnx模型
        public ModelInfo LoadOnnxModel(string path, string device_name)
        {
            core = new Core();
            Model model = core.read_model(path);
            compiled_model = core.compile_model(model, device_name);
            infer_request = compiled_model.create_infer_request();
            inputTensor = infer_request.get_input_tensor();

            //获取模型详细信息
            modelProto = ModelProto.Parser.ParseFromFile(path);
            var infos = modelProto.MetadataProps.ToList();

            modelInfo = new ModelInfo()
            {
                TaskName = infos.Find(x => x.Key == "task").Value,
                Batch = infos.Find(x => x.Key == "batch").Value,
                Imgsz = infos.Find(x => x.Key == "imgsz").Value,
                LabelNames = infos.Find(x => x.Key == "names").Value
            };

            modelType = (ModelType)Enum.Parse(typeof(ModelType), modelInfo.TaskName);

            return modelInfo;
        }

        public ModelInfo LoadOnnxModel(string path, ModelType modelType)
        {
            core = new Core();
            Model model = core.read_model(path);
            compiled_model = core.compile_model(model, "AUTO");
            infer_request = compiled_model.create_infer_request();
            inputTensor = infer_request.get_input_tensor();

            this.modelType = modelType;
            this.modelInfo = null;

            return modelInfo;
        }
        #endregion

        #region 固定输入尺寸模型推理过程

        #region Onnx推理
        public InferenceResult Inference(Bitmap image, float confidence = 0.5f, float iou = 0.5f)
        {
            switch (modelType)
            {
                case ModelType.obb:
                    return InferenceObb(image, confidence, iou);
                case ModelType.detect:
                    return InferenceDetect(image, confidence, iou);
                case ModelType.classify:
                    return InferenceClassify(image, confidence);
                case ModelType.segment:
                    return InferenceSegment(image, confidence, iou);
                case ModelType.pose:
                    return InferencePose(image, confidence, iou);

            }
            return null;
        }
        #endregion

        #region Obb检测
        /*
         张量 output: float32[1, 20, 21504] 的解释
            1: 代表批次大小 batch size，通常在推理过程中为 1。
            20: 代表每个预测框的特征维度，包括定向边界框相关的信息和类别信息。这个 20 维度可能包含如下信息：
                4: 水平边界框的四个坐标值（x, y, w, h，表示中心坐标和宽高）。
                1: 旋转角度（theta），用于表示定向边界框的旋转角度。
                1: 置信度分数（表示这个框是否包含物体的置信度）。
                n: 类别分数，假设有 n 个类别。
                对于 20 维来说，假设模型检测了 15 个类别，那么这个维度可以分解为：

                4: 边界框（x, y, w, h）。
                1: 旋转角度（theta）。
                1: 置信度。
                15: 类别分数。
            21504: 这是网络在特征图上生成的总候选框数量。这个数值通常取决于 YOLOv8 网络的输出特征图尺寸和网格大小，例如，84x256 = 21504，可能来自特征图的宽度和高度。

            OBB 检测输出张量的内容
                对于每个预测框的 20 维特征，通常包含以下几部分：

                1.水平边界框信息:
                    x, y: 边界框中心点的 x 和 y 坐标。
                    w, h: 边界框的宽度和高度。

                2.旋转角度:
                    theta: 定向边界框的旋转角度（通常以弧度表示，范围在 [-π, π]），表示边界框相对于水平轴的倾斜程度。

                3.置信度:
                    表示检测到物体的置信度分数。

                4.类别分数:
                    表示该边界框属于某个类别的概率值。对于 15 个类别，可能会有 15 个分类分数，表示该目标属于每个类别的置信度。

            定向边界框 (OBB) 的输出流程
                1.网格点生成预测框:
                    模型基于特征图的每个网格点生成多个预测框。这些预测框不仅预测了水平边界框，还预测了边界框的旋转角度（theta）。

                2.预测定向边界框 (OBB):
                    每个候选框的预测信息不仅包括物体的中心坐标和宽高，还包括其旋转角度，以便生成更精确的定向边界框。

                3.置信度和类别预测:
                    模型同时为每个候选框预测物体的存在置信度和类别置信度，用于后续的 NMS（非极大值抑制）等处理。

                4.后处理 (NMS):
                    后处理阶段会对所有预测框进行 NMS，去除重叠的框，同时生成最终的 OBB。
        */
        private InferenceResult InferenceObb(Bitmap image, float confidence = 0.5f, float iou = 0.5f)
        {
            try
            {
                long preprocessTime = 0;
                long inferenceTime = 0;
                long postprocessTime = 0;
                Stopwatch time = new Stopwatch();

                time.Restart();
                // 调整图像尺寸以符合ONNX模型的输入要求
                int targetWidth = (int)inputTensor.get_shape()[3];
                int targetHeight = (int)inputTensor.get_shape()[2];
                int ortImgWidth = image.Width;
                int ortImgHeight = image.Height;

                float ratio = Math.Min((float)targetWidth / ortImgWidth, (float)targetHeight / ortImgHeight);

                //Resize图片
                var bitmap = LetterboxResize(image, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImage(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { 1, 3, targetWidth, targetHeight })); // NCHW格式
                //输入预处理数据
                inputTensor.set_data<float>(tensor);
                time.Stop();
                preprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor = infer_request.get_output_tensor();
                var shape = output_tensor.get_shape();
                var output_length = (int)output_tensor.get_size();
                float[] output_data = output_tensor.get_data<float>(output_length);
                var output_data3D = ReshapeTo3D(output_data, (int)shape[0], (int)shape[1], (int)shape[2]);

                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                ConcurrentBag<Boxes> boxes = new ConcurrentBag<Boxes>();

                // 计算填充的像素数
                float padX = (targetWidth - ratio * ortImgWidth) / 2;
                float padY = (targetHeight - ratio * ortImgHeight) / 2;
                Parallel.For(0, shape[1] - 5, j =>
                {

                    for (int i = 0; i < shape[2]; i++)
                    {
                        float score = output_data3D[0, 4 + j, i];
                        if (score > confidence)
                        {
                            float x = output_data3D[0, 0, i];
                            float y = output_data3D[0, 1, i];
                            float w = output_data3D[0, 2, i];
                            float h = output_data3D[0, 3, i];
                            float r = output_data3D[0, shape[1] - 1, i];

                            // 将边界框信息还原到原图中
                            Boxes box = new Boxes()
                            {
                                Confidence = score,
                                LabelId = (int)j,
                                LabelName = labelNames.Count > 0 ? labelNames[(int)j] : "",
                                Angle = r,
                                X = (x - padX) / ratio,
                                Y = (y - padY) / ratio,
                                Width = w / ratio,
                                Height = h / ratio
                            };

                            boxes.Add(box);

                        }
                    }
                });

                //NMS
                List<Boxes> obb_Boxes = NonMaxSuppression(boxes.ToList(), iou);
                time.Stop();
                postprocessTime = time.ElapsedMilliseconds;

                bitmap?.Dispose();

                return new InferenceResult()
                {
                    Boxes = obb_Boxes,
                    ModelType = modelType,
                    ModelInfo = modelInfo,
                    OriginalImage = new Bitmap(image),
                    PreprocessTime = preprocessTime,
                    InferenceTime = inferenceTime,
                    PostprocessTime = postprocessTime
                };
            }
            catch
            {
                return null;
            }

        }
        #endregion

        #region ObjectDetect检测
        /*
         张量 output: float32[1, 84, 8400] 的结构说明

            1: 批次大小 batch size，通常在推理过程中为 1。

            84: 每个候选框的预测特征维度，总共有 84 个数值。它的组成如下：
                4: 边界框的坐标信息（x, y, w, h），表示中心点的 x 和 y 以及边界框的宽度 w 和高度 h。
                1: 置信度分数，表示该候选框包含物体的概率。
                80: 类别概率分布（假设检测的类别有 80 个），每个类别都有一个概率值。
                因此，84 维特征可以分解为：4 (边界框坐标) + 1 (置信度) + 80 (类别分数)。

            8400: 表示网络输出的总候选框数量。这个值通常由网络输出特征图的尺寸决定，表示 YOLOv8 在特征图上的每个位置生成了 8400 个候选框。这可以通过输出特征图的尺寸相乘得出，如 80x105 = 8400。

            解释各个部分的意义
            边界框坐标 (x, y, w, h):

            x: 边界框的中心点在图像中的 x 坐标。
            y: 边界框的中心点在图像中的 y 坐标。
            w: 边界框的宽度。
            h: 边界框的高度。
            YOLO 模型的边界框是通过网格的坐标来计算的，这意味着预测的是相对坐标，需要通过后处理（如乘以输入图像的宽高）才能得到实际像素值。

            置信度:
                置信度表示该边界框内是否包含物体的概率，值在 0 到 1 之间。如果置信度较高（例如 > 0.5），则认为该候选框中可能存在物体。

            类别分数:
                80 个类别分数对应于每个类别的概率值。模型会根据类别分数选择概率最高的类别作为该边界框内物体的类别。

            目标检测流程
                候选框生成:
                    YOLOv8 在输出特征图的每个网格点上生成多个候选框。通过 8400 个候选框的预测，每个候选框都有 84 维特征信息。

                边界框解析:
                    模型输出的 x, y, w, h 是归一化的相对坐标，需要通过乘以图像的宽度和高度，得到实际的边界框尺寸。

                置信度筛选:
                    根据置信度值，选择置信度高于某个阈值的候选框。如果置信度低于阈值，则认为该候选框不包含有效物体。

                类别分数解析:
                    对于每个候选框，选择类别分数最高的类别作为该候选框的最终类别预测。
                非极大值抑制（NMS）:
                    YOLO 输出的边界框可能有重叠，使用 NMS 去除重叠度高的候选框，保留置信度最高的框。
         */
        private InferenceResult InferenceDetect(Bitmap image, float confidence = 0.5f, float iou = 0.5f)
        {
            try
            {
                long preprocessTime = 0;
                long inferenceTime = 0;
                long postprocessTime = 0;
                Stopwatch time = new Stopwatch();

                time.Restart();
                // 调整图像尺寸以符合ONNX模型的输入要求
                int targetWidth = (int)inputTensor.get_shape()[3];
                int targetHeight = (int)inputTensor.get_shape()[2];
                int ortImgWidth = image.Width;
                int ortImgHeight = image.Height;

                float ratio = Math.Min((float)targetWidth / ortImgWidth, (float)targetHeight / ortImgHeight);

                //Resize图片
                var bitmap = LetterboxResize(image, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImage(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { 1, 3, targetWidth, targetHeight })); // NCHW格式
                //输入预处理数据
                inputTensor.set_data<float>(tensor);
                time.Stop();
                preprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor = infer_request.get_output_tensor();
                var shape = output_tensor.get_shape();
                var output_length = (int)output_tensor.get_size();
                float[] output_data = output_tensor.get_data<float>(output_length);
                var output_data3D = ReshapeTo3D(output_data, (int)shape[0], (int)shape[1], (int)shape[2]);

                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                ConcurrentBag<Boxes> boxes = new ConcurrentBag<Boxes>();

                // 计算填充的像素数
                float padX = (targetWidth - ratio * ortImgWidth) / 2;
                float padY = (targetHeight - ratio * ortImgHeight) / 2;
                Parallel.For(0, shape[1] - 4, j =>
                {

                    for (int i = 0; i < shape[2]; i++)
                    {
                        float score = output_data3D[0, 4 + j, i];
                        if (score > confidence)
                        {
                            float x = output_data3D[0, 0, i];
                            float y = output_data3D[0, 1, i];
                            float w = output_data3D[0, 2, i];
                            float h = output_data3D[0, 3, i];
                            float r = 0;


                            // 将边界框信息还原到原图中
                            Boxes box = new Boxes()
                            {
                                Confidence = score,
                                LabelId = (int)j,
                                LabelName = labelNames.Count > 0 ? labelNames[(int)j] : "",
                                Angle = r,
                                X = (x - padX) / ratio,
                                Y = (y - padY) / ratio,
                                Width = w / ratio,
                                Height = h / ratio
                            };

                            boxes.Add(box);

                        }
                    }

                });

                //NMS
                List<Boxes> boxes1 = NonMaxSuppression(boxes.ToList(), iou);
                time.Stop();
                postprocessTime = time.ElapsedMilliseconds;

                bitmap?.Dispose();

                return new InferenceResult()
                {
                    Boxes = boxes1,
                    ModelType = modelType,
                    ModelInfo = modelInfo,
                    OriginalImage = new Bitmap(image),
                    PreprocessTime = preprocessTime,
                    InferenceTime = inferenceTime,
                    PostprocessTime = postprocessTime,
                };
            }
            catch
            {
                return null;
            }

        }
        #endregion

        #region Classify检测
        /*
         张量 output: float32[1, 1000] 的结构说明
            1: 这是批次大小 batch size，通常表示模型一次处理的图片数量。在推理阶段，通常会设置为 1，表示处理单张图片。
            1000: 这是模型输出的类别数。每个数字代表对应类别的置信度分数，数值范围通常在 0 到 1 之间。1000 维的输出意味着模型被训练来识别 1000 个不同的类别。
            每个输出值代表模型对于某个类别的预测分数。通常，最高分数对应的类别就是模型对于输入图片的最终分类结果。

            分类输出张量的处理步骤
                类别置信度分数:
                    每个元素表示模型预测图片属于某个类别的置信度。值越大，模型越确定输入图像属于该类别。

                选取最高置信度的类别:
                    通常的做法是找到置信度最高的那个类别作为预测结果。这个过程可以通过找出最大值及其对应的索引来实现。

                类别映射:
                    YOLOv8 分类模型通常有预定义的 1000 个类别映射表。通过索引，可以将预测的索引值转换为相应的类别名称。
         */
        private InferenceResult InferenceClassify(Bitmap image, float confidence = 0.5f)
        {
            try
            {
                long preprocessTime = 0;
                long inferenceTime = 0;
                long postprocessTime = 0;
                Stopwatch time = new Stopwatch();

                time.Restart();
                // 调整图像尺寸以符合ONNX模型的输入要求
                int targetWidth = (int)inputTensor.get_shape()[3];
                int targetHeight = (int)inputTensor.get_shape()[2];


                //Resize图片
                var bitmap = LetterboxResize(image, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImage(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { 1, 3, targetWidth, targetHeight })); // NCHW格式
                //输入预处理数据
                inputTensor.set_data<float>(tensor);
                time.Stop();
                preprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor = infer_request.get_output_tensor();
                var shape = output_tensor.get_shape();
                var output_length = (int)output_tensor.get_size();
                float[] output_data = output_tensor.get_data<float>(output_length);

                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                var maxElement = output_data.Select((maxConfidence, index) => new { MaxConfidence = maxConfidence, Index = index }).OrderByDescending(x => x.MaxConfidence).First();
                var selectArrar = output_data.Select((confidences, indexes) => new { Confidences = confidences, Indexes = indexes }).Where(x => x.Confidences >= confidence).ToArray();

                ClassifyRes classifyRes = new ClassifyRes()
                {
                    AllClassifyResultIndex = selectArrar.Select(x => x.Indexes).ToArray(),
                    AllConfidence = selectArrar.Select(x => x.Confidences).ToArray(),
                    MaxClassifyResultIndex = maxElement.Index,
                    MaxClassifyResultLabelName = labelNames.Count > 0 ? labelNames[maxElement.Index] : "",
                    MaxConfidence = maxElement.MaxConfidence
                };

                classifyRes.AllClassifyResultLabelName = new string[classifyRes.AllClassifyResultIndex.Count()];
                Parallel.For(0, classifyRes.AllClassifyResultIndex.Count(), i =>
                {
                    classifyRes.AllClassifyResultLabelName[i] = labelNames.Count > 0 ? labelNames[classifyRes.AllClassifyResultIndex[i]] : "";
                });
                time.Stop();
                postprocessTime = time.ElapsedMilliseconds;

                bitmap?.Dispose();
                return new InferenceResult()
                {
                    ClassifyRes = classifyRes,
                    ModelType = modelType,
                    ModelInfo = modelInfo,
                    OriginalImage = new Bitmap(image),
                    PreprocessTime = preprocessTime,
                    InferenceTime = inferenceTime,
                    PostprocessTime = postprocessTime,
                };
            }
            catch
            {
                return null;
            }

        }
        #endregion

        #region Segment检测
        /*
            1.output0: float32[1, 116, 8400]

                这个张量包含 YOLOv8 的目标检测输出，其每个部分都存储了与目标相关的信息。维度为 [1, 116, 8400]，可以分为以下几个部分：

                1: 代表批次大小 batch size，在推理时通常是 1。
                116: 代表每个预测框的总信息量，具体分解如下：
                    4: 边界框坐标（x, y, 宽度, 高度）。
                    80: 分类置信度（假设有 80 个类别，每个类别都有一个置信度值）。
                    32: 与分割（segmentation）相关的特征向量，后面用于生成分割掩码。
                8400: 代表网格上的预测点数量，即模型在每个网格点上预测了 8400 个可能的检测框。通常这是模型特征图的宽高乘积决定的，比如 80x105，这与 YOLO 网络的结构相关。
                
                总结：
                    output0 包含了每个候选目标的边界框、类别置信度以及用于分割的特征向量。

            2. output1: float32[1, 32, 160, 160]
                这个张量包含 YOLOv8 的分割掩码（segmentation mask）相关信息。维度为 [1, 32, 160, 160]：

                1: 代表批次大小 batch size，在推理时通常是 1。
                32: 代表通道数，也就是分割掩码的特征维度（segmentation features），这些特征向量将用于生成最终的分割掩码。
                160x160: 代表特征图的高和宽（通常为下采样后的尺寸）。这些大小取决于网络结构，将与目标框的特征向量组合生成掩码。

                总结：
                    output1 提供了每个通道的特征图，用于生成目标的掩码，结合 output0 中的分割特征，通过矩阵乘法和 Sigmoid 操作，生成目标的最终分割掩码。

            分割掩码生成的详细过程
                检测输出的分割特征 (output0)：
                    从 output0 中的每个候选框提取最后的 32 维特征，这些特征与 output1 的分割特征图进行结合，用于生成掩码。
                    掩码特征图 (output1)：
                        output1 包含了 32 个通道，每个通道都是 160x160 的分割特征图。这些特征图通过与检测框的 32 维分割特征（从 output0 中提取的最后 32 维）进行矩阵乘法，得到一个新的 160x160 的掩码图。
                
                生成掩码：
                    将 output0 中提取的分割特征与 output1 进行矩阵乘法，生成 N x 160 x 160 的分割掩码。
                    然后通过应用 Sigmoid 函数，将结果转换为 0 到 1 之间的值。
                    最终，将掩码值大于 0.5 的部分标记为 1，得到二值化的分割掩码。

                完整过程
                    目标检测：output0 给出每个检测框的位置信息和分类置信度。
                    分割特征提取：从 output0 中提取最后 32 维分割特征，并与 output1 中的特征图进行矩阵乘法，生成掩码。
                    Sigmoid：对结果应用 Sigmoid，将数值映射到 [0, 1] 区间。
                    二值化掩码：对掩码进行二值化，生成最终的分割图。

                总结
                    output0 主要用于目标检测，包括边界框、类别置信度和分割特征。
                    output1 用于分割掩码生成，提供了用于生成分割掩码的特征图。
         */
        private InferenceResult InferenceSegment(Bitmap image, float confidence = 0.5f, float iou = 0.5f)
        {
            try
            {
                long preprocessTime = 0;
                long inferenceTime = 0;
                long postprocessTime = 0;
                Stopwatch time = new Stopwatch();

                time.Restart();
                // 调整图像尺寸以符合ONNX模型的输入要求
                int targetWidth = (int)inputTensor.get_shape()[3];
                int targetHeight = (int)inputTensor.get_shape()[2];
                int ortImgWidth = image.Width;
                int ortImgHeight = image.Height;

                float ratio = Math.Min((float)targetWidth / ortImgWidth, (float)targetHeight / ortImgHeight);

                //Resize图片
                var bitmap = LetterboxResize(image, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImage(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { 1, 3, targetWidth, targetHeight })); // NCHW格式
                //输入预处理数据
                inputTensor.set_data<float>(tensor);
                time.Stop();
                preprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor0 = infer_request.get_output_tensor(0);
                var shape0 = output_tensor0.get_shape();
                var output_length0 = (int)output_tensor0.get_size();
                float[] output_data0 = output_tensor0.get_data<float>(output_length0);
                var output_data3D0 = ReshapeTo3D(output_data0, (int)shape0[0], (int)shape0[1], (int)shape0[2]);

                var output_tensor1 = infer_request.get_output_tensor(1);
                Shape maskShape = output_tensor1.get_shape();
                var output_length1 = (int)output_tensor1.get_size();
                float[] output_data1 = output_tensor1.get_data<float>(output_length1);
                var mask_data4D = ReshapeTo4D(output_data1, (int)maskShape[0], (int)maskShape[1], (int)maskShape[2], (int)maskShape[3]);


                //获取标签
                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                ConcurrentBag<Boxes> boxes = new ConcurrentBag<Boxes>();

                // 计算填充的像素数
                float padX = (targetWidth - ratio * ortImgWidth) / 2;
                float padY = (targetHeight - ratio * ortImgHeight) / 2;
                Parallel.For(0, shape0[1] - 4 - maskShape[1], j =>
                {
                    for (int i = 0; i < shape0[2]; i++)
                    {
                        float score = output_data3D0[0, 4 + j, i];
                        if (score > confidence)
                        {
                            float x = output_data3D0[0, 0, i];
                            float y = output_data3D0[0, 1, i];
                            float w = output_data3D0[0, 2, i];
                            float h = output_data3D0[0, 3, i];
                            float r = 0;

                            // 将边界框信息还原到原图中
                            Boxes box = new Boxes()
                            {
                                Confidence = score,
                                LabelId = (int)j,
                                LabelName = labelNames.Count > 0 ? labelNames[(int)j] : "",
                                Angle = r,
                                X = (x - padX) / ratio,
                                Y = (y - padY) / ratio,
                                Width = w / ratio,
                                Height = h / ratio,
                                BoxIndex = i
                            };

                            boxes.Add(box);

                        }
                    }
                });

                //NMS
                List<Boxes> boxes1 = NonMaxSuppression(boxes.ToList(), iou);

                // 处理分割掩码
                foreach (var item in boxes1)
                {
                    //计算mask权重值
                    var maskWeights = ExtractMaskWeights(output_data3D0, item.BoxIndex, (int)maskShape[1], (int)shape0[1] - (int)maskShape[1]);

                    //将按原图box值转成mask的尺寸并进行权重值计算得到mask(缩小遍历次数，每次只获取box范围内的mask值)
                    float x = (item.X * ratio + padX) / (targetWidth / maskShape[2]);
                    float y = (item.Y * ratio + padY) / (targetWidth / maskShape[2]);
                    float w = item.Width * ratio / (targetWidth / maskShape[2]);
                    float h = item.Height * ratio / (targetWidth / maskShape[2]);
                    int boxX = Math.Max(0, (int)(x - w / 2) - 1);
                    int boxY = Math.Max(0, (int)(y - h / 2) - 1);
                    int maxX = Math.Min(boxX + (int)w + 2, (int)maskShape[2]);
                    int maxY = Math.Min(boxY + (int)h + 2, (int)maskShape[2]);
                    float[,] mask = ProcessMask(maskShape, mask_data4D, maskWeights, boxX, boxY, maxX, maxY);

                    //将mask缩放至原图尺寸大小
                    item.Mask = ResizeMaskAfterLetterbox(mask, item, image.Width, image.Height);
                }
                time.Stop();
                postprocessTime = time.ElapsedMilliseconds;

                bitmap?.Dispose();

                return new InferenceResult()
                {
                    Boxes = boxes1,
                    ModelType = modelType,
                    ModelInfo = modelInfo,
                    OriginalImage = new Bitmap(image),
                    PreprocessTime = preprocessTime,
                    InferenceTime = inferenceTime,
                    PostprocessTime = postprocessTime,
                };
            }
            catch
            {
                //image?.Dispose();
                return null;
            }

        }

        #endregion

        #region Pose检测
        /*
         张量 output: float32[1, 56,8400] 的结构说明
         1: 批次大小 batch size，固定尺寸模型中推理过程中为 1。
         56: 4 个坐标值（bbox）  1 个置信度（confidence）  17 个关键点 × 3 个值（x, y, 可见性）
         8400: 预测框数量 模型在特征图上生成的初始锚框数量
         */
        private InferenceResult InferencePose(Bitmap image, float confidence = 0.5f, float iou = 0.5f)
        {
            try
            {
                long preprocessTime = 0;
                long inferenceTime = 0;
                long postprocessTime = 0;
                Stopwatch time = new Stopwatch();

                time.Restart();
                // 调整图像尺寸以符合ONNX模型的输入要求
                int targetWidth = (int)inputTensor.get_shape()[3];
                int targetHeight = (int)inputTensor.get_shape()[2];
                int ortImgWidth = image.Width;
                int ortImgHeight = image.Height;

                float ratio = Math.Min((float)targetWidth / ortImgWidth, (float)targetHeight / ortImgHeight);

                //Resize图片
                var bitmap = LetterboxResize(image, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImage(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { 1, 3, targetWidth, targetHeight })); // NCHW格式
                //输入预处理数据
                inputTensor.set_data<float>(tensor);
                time.Stop();
                preprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor = infer_request.get_output_tensor();
                var shape = output_tensor.get_shape();
                var output_length = (int)output_tensor.get_size();
                float[] output_data = output_tensor.get_data<float>(output_length);
                var output_data3D = ReshapeTo3D(output_data, (int)shape[0], (int)shape[1], (int)shape[2]);

                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                ConcurrentBag<Boxes> boxes = new ConcurrentBag<Boxes>();

                // 计算填充的像素数
                float padX = (targetWidth - ratio * ortImgWidth) / 2;
                float padY = (targetHeight - ratio * ortImgHeight) / 2;
                Parallel.For(0, shape[2], i =>
                {
                    float score = output_data3D[0, 4, i];
                    if (score > confidence)
                    {
                        float x = output_data3D[0, 0, i];
                        float y = output_data3D[0, 1, i];
                        float w = output_data3D[0, 2, i];
                        float h = output_data3D[0, 3, i];
                        float r = 0;

                        var keypoints = new KeyPoint[17]; // 17个关键点，每个关键点有x和y坐标
                        //获取关键点坐标信息
                        for (int j = 0; j < 17; j++)
                        {
                            int index = 5 + j * 3;
                            float keypointX = output_data3D[0, index, i];
                            float keypointY = output_data3D[0, index + 1, i];
                            float keypointVisibility = output_data3D[0, index + 2, i];
                            string keypointName = KeyPoint.KeypointNames[j];
                            keypoints[j] = new KeyPoint()
                            {
                                X = (keypointX - padX) / ratio,
                                Y = (keypointY - padY) / ratio,
                                Visibility = keypointVisibility,
                                Name = keypointName
                            };
                        }

                        // 将边界框信息还原到原图中
                        Boxes box = new Boxes()
                        {
                            Confidence = score,
                            LabelId = 0,
                            LabelName = labelNames[0],
                            Angle = r,
                            X = (x - padX) / ratio,
                            Y = (y - padY) / ratio,
                            Width = w / ratio,
                            Height = h / ratio,
                            KeyPoints = keypoints
                        };
                        boxes.Add(box);

                    }

                });

                //NMS
                List<Boxes> boxes1 = NonMaxSuppression(boxes.ToList(), iou);
                time.Stop();
                postprocessTime = time.ElapsedMilliseconds;

                bitmap?.Dispose();

                return new InferenceResult()
                {
                    Boxes = boxes1,
                    ModelType = modelType,
                    ModelInfo = modelInfo,
                    OriginalImage = new Bitmap(image),
                    PreprocessTime = preprocessTime,
                    InferenceTime = inferenceTime,
                    PostprocessTime = postprocessTime
                };

            }
            catch
            {
                return null;
            }
        }
        #endregion

        #endregion

        #region 动态输入尺寸模型推理过程
        #region Onnx推理
        public InferenceResult[] Inference_Dyamics(List<Bitmap> images, out InferTimeInfo inferTimeInfo, int targetWidth = 640, int targetHeight = 640, float confidence = 0.5f, float iou = 0.5f)
        {
            inferTimeInfo = new InferTimeInfo();
            switch (modelType)
            {
                case ModelType.obb:
                    return InferenceObb_Dyamics(images, out inferTimeInfo, targetWidth, targetHeight, confidence, iou);
                case ModelType.detect:
                    return InferenceDetect_Dyamics(images, out inferTimeInfo, targetWidth, targetHeight, confidence, iou);
                case ModelType.classify:
                    return InferenceClassify_Dyamics(images, out inferTimeInfo, targetWidth, targetHeight, confidence, iou);
                case ModelType.segment:
                    return InferenceSegment_Dyamics(images,out inferTimeInfo , targetWidth, targetHeight,confidence, iou);
                case ModelType.pose:
                    return InferencePose_Dyamics(images, out inferTimeInfo, targetWidth, targetHeight, confidence, iou);

            }
            return null;
        }
        #endregion

        #region Obb检测
        /*
         张量 output: float32[1, 20, 21504] 的解释
            1: 代表批次大小 batch size，通常在推理过程中为 1。
            20: 代表每个预测框的特征维度，包括定向边界框相关的信息和类别信息。这个 20 维度可能包含如下信息：
                4: 水平边界框的四个坐标值（x, y, w, h，表示中心坐标和宽高）。
                1: 旋转角度（theta），用于表示定向边界框的旋转角度。
                1: 置信度分数（表示这个框是否包含物体的置信度）。
                n: 类别分数，假设有 n 个类别。
                对于 20 维来说，假设模型检测了 15 个类别，那么这个维度可以分解为：

                4: 边界框（x, y, w, h）。
                1: 旋转角度（theta）。
                1: 置信度。
                15: 类别分数。
            21504: 这是网络在特征图上生成的总候选框数量。这个数值通常取决于 YOLOv8 网络的输出特征图尺寸和网格大小，例如，84x256 = 21504，可能来自特征图的宽度和高度。

            OBB 检测输出张量的内容
                对于每个预测框的 20 维特征，通常包含以下几部分：

                1.水平边界框信息:
                    x, y: 边界框中心点的 x 和 y 坐标。
                    w, h: 边界框的宽度和高度。

                2.旋转角度:
                    theta: 定向边界框的旋转角度（通常以弧度表示，范围在 [-π, π]），表示边界框相对于水平轴的倾斜程度。

                3.置信度:
                    表示检测到物体的置信度分数。

                4.类别分数:
                    表示该边界框属于某个类别的概率值。对于 15 个类别，可能会有 15 个分类分数，表示该目标属于每个类别的置信度。

            定向边界框 (OBB) 的输出流程
                1.网格点生成预测框:
                    模型基于特征图的每个网格点生成多个预测框。这些预测框不仅预测了水平边界框，还预测了边界框的旋转角度（theta）。

                2.预测定向边界框 (OBB):
                    每个候选框的预测信息不仅包括物体的中心坐标和宽高，还包括其旋转角度，以便生成更精确的定向边界框。

                3.置信度和类别预测:
                    模型同时为每个候选框预测物体的存在置信度和类别置信度，用于后续的 NMS（非极大值抑制）等处理。

                4.后处理 (NMS):
                    后处理阶段会对所有预测框进行 NMS，去除重叠的框，同时生成最终的 OBB。
        */
        private InferenceResult[] InferenceObb_Dyamics(List<Bitmap> images, out InferTimeInfo inferTimeInfo, int targetWidth = 640, int targetHeight = 640, float confidence = 0.5f, float iou = 0.5f)
        {
            try
            {
                inferTimeInfo = new InferTimeInfo();
                Stopwatch time = new Stopwatch();

                time.Restart();
                //Resize图片
                var bitmap = LetterboxResize_Dyamics(images, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImages_Dyamics(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { images.Count, 3, targetWidth, targetHeight })); // NCHW格式
                //输入预处理数据
                inputTensor.set_data<float>(tensor);

                time.Stop();
                inferTimeInfo.PreprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferTimeInfo.InferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor = infer_request.get_output_tensor();
                var shape = output_tensor.get_shape();
                var output_length = (int)output_tensor.get_size();
                float[] output_data = output_tensor.get_data<float>(output_length);
                var output_data3D = ReshapeTo3D(output_data, (int)shape[0], (int)shape[1], (int)shape[2]);

                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                List<InferenceResult> InferenceResults = new List<InferenceResult>();

                for (int l = 0; l < images.Count; l++)
                {
                    ConcurrentBag<Boxes> boxes = new ConcurrentBag<Boxes>();
                    Bitmap image = images[l];

                    int ortImgWidth = image.Width;
                    int ortImgHeight = image.Height;
                    float ratio = Math.Min((float)targetWidth / ortImgWidth, (float)targetHeight / ortImgHeight);

                    // 计算填充的像素数
                    float padX = (targetWidth - ratio * ortImgWidth) / 2;
                    float padY = (targetHeight - ratio * ortImgHeight) / 2;

                    Parallel.For(0, shape[1] - 5, j =>
                    {

                        for (int i = 0; i < shape[2]; i++)
                        {
                            float score = output_data3D[l, 4 + j, i];
                            if (score > confidence)
                            {
                                float x = output_data3D[l, 0, i];
                                float y = output_data3D[l, 1, i];
                                float w = output_data3D[l, 2, i];
                                float h = output_data3D[l, 3, i];
                                float r = output_data3D[l, shape[1] - 1, i];

                                // 将边界框信息还原到原图中
                                Boxes box = new Boxes()
                                {
                                    Confidence = score,
                                    LabelId = (int)j,
                                    LabelName = labelNames.Count > 0 ? labelNames[(int)j] : "",
                                    Angle = r,
                                    X = (x - padX) / ratio,
                                    Y = (y - padY) / ratio,
                                    Width = w / ratio,
                                    Height = h / ratio
                                };

                                boxes.Add(box);

                            }
                        }
                    });

                    //NMS
                    List<Boxes> obb_Boxes = NonMaxSuppression(boxes.ToList(), iou);

                    InferenceResults.Add(new InferenceResult()
                    {
                        Boxes = obb_Boxes,
                        ModelType = modelType,
                        ModelInfo = modelInfo,
                        OriginalImage = new Bitmap(image)
                    });

                    image?.Dispose();
                }

                return InferenceResults.ToArray();
            }
            catch
            {
                inferTimeInfo = new InferTimeInfo();
                return null;
            }

        }
        #endregion

        #region ObjectDetect检测
        /*
         张量 output: float32[1, 84, 8400] 的结构说明

            1: 批次大小 batch size，通常在推理过程中为 1。

            84: 每个候选框的预测特征维度，总共有 84 个数值。它的组成如下：
                4: 边界框的坐标信息（x, y, w, h），表示中心点的 x 和 y 以及边界框的宽度 w 和高度 h。
                1: 置信度分数，表示该候选框包含物体的概率。
                80: 类别概率分布（假设检测的类别有 80 个），每个类别都有一个概率值。
                因此，84 维特征可以分解为：4 (边界框坐标) + 1 (置信度) + 80 (类别分数)。

            8400: 表示网络输出的总候选框数量。这个值通常由网络输出特征图的尺寸决定，表示 YOLOv8 在特征图上的每个位置生成了 8400 个候选框。这可以通过输出特征图的尺寸相乘得出，如 80x105 = 8400。

            解释各个部分的意义
            边界框坐标 (x, y, w, h):

            x: 边界框的中心点在图像中的 x 坐标。
            y: 边界框的中心点在图像中的 y 坐标。
            w: 边界框的宽度。
            h: 边界框的高度。
            YOLO 模型的边界框是通过网格的坐标来计算的，这意味着预测的是相对坐标，需要通过后处理（如乘以输入图像的宽高）才能得到实际像素值。

            置信度:
                置信度表示该边界框内是否包含物体的概率，值在 0 到 1 之间。如果置信度较高（例如 > 0.5），则认为该候选框中可能存在物体。

            类别分数:
                80 个类别分数对应于每个类别的概率值。模型会根据类别分数选择概率最高的类别作为该边界框内物体的类别。

            目标检测流程
                候选框生成:
                    YOLOv8 在输出特征图的每个网格点上生成多个候选框。通过 8400 个候选框的预测，每个候选框都有 84 维特征信息。

                边界框解析:
                    模型输出的 x, y, w, h 是归一化的相对坐标，需要通过乘以图像的宽度和高度，得到实际的边界框尺寸。

                置信度筛选:
                    根据置信度值，选择置信度高于某个阈值的候选框。如果置信度低于阈值，则认为该候选框不包含有效物体。

                类别分数解析:
                    对于每个候选框，选择类别分数最高的类别作为该候选框的最终类别预测。
                非极大值抑制（NMS）:
                    YOLO 输出的边界框可能有重叠，使用 NMS 去除重叠度高的候选框，保留置信度最高的框。
         */
        private InferenceResult[] InferenceDetect_Dyamics(List<Bitmap> images, out InferTimeInfo inferTimeInfo, int targetWidth = 640, int targetHeight = 640, float confidence = 0.5f, float iou = 0.5f)
        {
            try
            {
                inferTimeInfo = new InferTimeInfo();
                Stopwatch time = new Stopwatch();

                time.Restart();
                //Resize图片
                var bitmap = LetterboxResize_Dyamics(images, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImages_Dyamics(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { images.Count, 3, targetWidth, targetHeight })); // NCHW格式
                //输入预处理数据
                inputTensor.set_data<float>(tensor);

                time.Stop();
                inferTimeInfo.PreprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferTimeInfo.InferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor = infer_request.get_output_tensor();
                var shape = output_tensor.get_shape();
                var output_length = (int)output_tensor.get_size();
                float[] output_data = output_tensor.get_data<float>(output_length);
                var output_data3D = ReshapeTo3D(output_data, (int)shape[0], (int)shape[1], (int)shape[2]);

                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                List<InferenceResult> InferenceResults = new List<InferenceResult>();

                for (int l = 0; l < images.Count; l++)
                {
                    ConcurrentBag<Boxes> boxes = new ConcurrentBag<Boxes>();
                    Bitmap image = images[l];

                    int ortImgWidth = image.Width;
                    int ortImgHeight = image.Height;
                    float ratio = Math.Min((float)targetWidth / ortImgWidth, (float)targetHeight / ortImgHeight);

                    // 计算填充的像素数
                    float padX = (targetWidth - ratio * ortImgWidth) / 2;
                    float padY = (targetHeight - ratio * ortImgHeight) / 2;

                    Parallel.For(0, shape[1] - 4, j =>
                    {

                        for (int i = 0; i < shape[2]; i++)
                        {
                            float score = output_data3D[l, 4 + j, i];
                            if (score > confidence)
                            {
                                float x = output_data3D[l, 0, i];
                                float y = output_data3D[l, 1, i];
                                float w = output_data3D[l, 2, i];
                                float h = output_data3D[l, 3, i];
                                float r = 0;


                                // 将边界框信息还原到原图中
                                Boxes box = new Boxes()
                                {
                                    Confidence = score,
                                    LabelId = (int)j,
                                    LabelName = labelNames.Count > 0 ? labelNames[(int)j] : "",
                                    Angle = r,
                                    X = (x - padX) / ratio,
                                    Y = (y - padY) / ratio,
                                    Width = w / ratio,
                                    Height = h / ratio
                                };

                                boxes.Add(box);

                            }
                        }

                    });

                    //NMS
                    List<Boxes> boxes1 = NonMaxSuppression(boxes.ToList(), iou);

                    InferenceResults.Add(new InferenceResult()
                    {
                        Boxes = boxes1,
                        ModelType = modelType,
                        ModelInfo = modelInfo,
                        OriginalImage = new Bitmap(image),
                    });

                    image?.Dispose();
                }

                return InferenceResults.ToArray();
            }
            catch
            {
                inferTimeInfo = new InferTimeInfo();
                return null;
            }

        }
        #endregion

        #region Classify检测
        /*
         张量 output: float32[1, 1000] 的结构说明
            1: 这是批次大小 batch size，通常表示模型一次处理的图片数量。在推理阶段，通常会设置为 1，表示处理单张图片。
            1000: 这是模型输出的类别数。每个数字代表对应类别的置信度分数，数值范围通常在 0 到 1 之间。1000 维的输出意味着模型被训练来识别 1000 个不同的类别。
            每个输出值代表模型对于某个类别的预测分数。通常，最高分数对应的类别就是模型对于输入图片的最终分类结果。

            分类输出张量的处理步骤
                类别置信度分数:
                    每个元素表示模型预测图片属于某个类别的置信度。值越大，模型越确定输入图像属于该类别。

                选取最高置信度的类别:
                    通常的做法是找到置信度最高的那个类别作为预测结果。这个过程可以通过找出最大值及其对应的索引来实现。

                类别映射:
                    YOLOv8 分类模型通常有预定义的 1000 个类别映射表。通过索引，可以将预测的索引值转换为相应的类别名称。
         */
        private InferenceResult[] InferenceClassify_Dyamics(List<Bitmap> images, out InferTimeInfo inferTimeInfo, int targetWidth = 640, int targetHeight = 640, float confidence = 0.5f, float iou = 0.5f)
        {
            try
            {
                inferTimeInfo = new InferTimeInfo();
                Stopwatch time = new Stopwatch();

                time.Restart();
                //Resize图片
                var bitmap = LetterboxResize_Dyamics(images, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImages_Dyamics(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { images.Count, 3, targetWidth, targetHeight })); // NCHW格式
                //输入预处理数据
                inputTensor.set_data<float>(tensor);

                time.Stop();
                inferTimeInfo.PreprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferTimeInfo.InferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor = infer_request.get_output_tensor();
                var shape = output_tensor.get_shape();
                var output_length = (int)output_tensor.get_size();
                float[] output_data = output_tensor.get_data<float>(output_length);

                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                List<InferenceResult> InferenceResults = new List<InferenceResult>();

                for (int l = 0; l < images.Count; l++)
                {
                    Bitmap image = new Bitmap(images[l]);

                    float[] tempFloat = new float[shape[1]];
                    int startIndex = l * (int)shape[1];
                    Array.Copy(output_data, startIndex, tempFloat, 0, (int)shape[1]);

                    var maxElement = tempFloat.Select((maxConfidence, index) => new { MaxConfidence = maxConfidence, Index = index }).OrderByDescending(x => x.MaxConfidence).First();
                    var selectArrar = tempFloat.Select((confidences, indexes) => new { Confidences = confidences, Indexes = indexes }).Where(x => x.Confidences >= confidence).ToArray();

                    ClassifyRes classifyRes = new ClassifyRes()
                    {
                        AllClassifyResultIndex = selectArrar.Select(x => x.Indexes).ToArray(),
                        AllConfidence = selectArrar.Select(x => x.Confidences).ToArray(),
                        MaxClassifyResultIndex = maxElement.Index,
                        MaxClassifyResultLabelName = labelNames.Count > 0 ? labelNames[maxElement.Index] : "",
                        MaxConfidence = maxElement.MaxConfidence
                    };

                    classifyRes.AllClassifyResultLabelName = new string[classifyRes.AllClassifyResultIndex.Count()];
                    Parallel.For(0, classifyRes.AllClassifyResultIndex.Count(), i =>
                    {
                        classifyRes.AllClassifyResultLabelName[i] = labelNames.Count > 0 ? labelNames[classifyRes.AllClassifyResultIndex[i]] : "";
                    });

                    InferenceResults.Add(new InferenceResult()
                    {
                        ClassifyRes = classifyRes,
                        ModelType = modelType,
                        ModelInfo = modelInfo,
                        OriginalImage = new Bitmap(image),
                    });

                    image?.Dispose();

                }
                return InferenceResults.ToArray();
            }
            catch
            {
                inferTimeInfo = new InferTimeInfo();
                return null;
            }

        }
        #endregion

        #region Segment检测
        /*
            1.output0: float32[1, 116, 8400]

                这个张量包含 YOLOv8 的目标检测输出，其每个部分都存储了与目标相关的信息。维度为 [1, 116, 8400]，可以分为以下几个部分：

                1: 代表批次大小 batch size，在推理时通常是 1。
                116: 代表每个预测框的总信息量，具体分解如下：
                    4: 边界框坐标（x, y, 宽度, 高度）。
                    80: 分类置信度（假设有 80 个类别，每个类别都有一个置信度值）。
                    32: 与分割（segmentation）相关的特征向量，后面用于生成分割掩码。
                8400: 代表网格上的预测点数量，即模型在每个网格点上预测了 8400 个可能的检测框。通常这是模型特征图的宽高乘积决定的，比如 80x105，这与 YOLO 网络的结构相关。
                
                总结：
                    output0 包含了每个候选目标的边界框、类别置信度以及用于分割的特征向量。

            2. output1: float32[1, 32, 160, 160]
                这个张量包含 YOLOv8 的分割掩码（segmentation mask）相关信息。维度为 [1, 32, 160, 160]：

                1: 代表批次大小 batch size，在推理时通常是 1。
                32: 代表通道数，也就是分割掩码的特征维度（segmentation features），这些特征向量将用于生成最终的分割掩码。
                160x160: 代表特征图的高和宽（通常为下采样后的尺寸）。这些大小取决于网络结构，将与目标框的特征向量组合生成掩码。

                总结：
                    output1 提供了每个通道的特征图，用于生成目标的掩码，结合 output0 中的分割特征，通过矩阵乘法和 Sigmoid 操作，生成目标的最终分割掩码。

            分割掩码生成的详细过程
                检测输出的分割特征 (output0)：
                    从 output0 中的每个候选框提取最后的 32 维特征，这些特征与 output1 的分割特征图进行结合，用于生成掩码。
                    掩码特征图 (output1)：
                        output1 包含了 32 个通道，每个通道都是 160x160 的分割特征图。这些特征图通过与检测框的 32 维分割特征（从 output0 中提取的最后 32 维）进行矩阵乘法，得到一个新的 160x160 的掩码图。
                
                生成掩码：
                    将 output0 中提取的分割特征与 output1 进行矩阵乘法，生成 N x 160 x 160 的分割掩码。
                    然后通过应用 Sigmoid 函数，将结果转换为 0 到 1 之间的值。
                    最终，将掩码值大于 0.5 的部分标记为 1，得到二值化的分割掩码。

                完整过程
                    目标检测：output0 给出每个检测框的位置信息和分类置信度。
                    分割特征提取：从 output0 中提取最后 32 维分割特征，并与 output1 中的特征图进行矩阵乘法，生成掩码。
                    Sigmoid：对结果应用 Sigmoid，将数值映射到 [0, 1] 区间。
                    二值化掩码：对掩码进行二值化，生成最终的分割图。

                总结
                    output0 主要用于目标检测，包括边界框、类别置信度和分割特征。
                    output1 用于分割掩码生成，提供了用于生成分割掩码的特征图。
         */
        private InferenceResult[] InferenceSegment_Dyamics(List<Bitmap> images, out InferTimeInfo inferTimeInfo, int targetWidth = 640, int targetHeight = 640, float confidence = 0.5f, float iou = 0.5f)
        {
            try
            {
                inferTimeInfo = new InferTimeInfo();
                Stopwatch time = new Stopwatch();

                time.Restart();
                //Resize图片
                var bitmap = LetterboxResize_Dyamics(images, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImages_Dyamics(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { images.Count, 3, targetWidth, targetHeight })); // NCHW格式
                                                                                                              //输入预处理数据

                inputTensor.set_data<float>(tensor);

                time.Stop();
                inferTimeInfo.PreprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferTimeInfo.InferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor0 = infer_request.get_output_tensor(0);
                var shape0 = output_tensor0.get_shape();
                var output_length0 = (int)output_tensor0.get_size();
                float[] output_data0 = output_tensor0.get_data<float>(output_length0);
                var output_data3D0 = ReshapeTo3D(output_data0, (int)shape0[0], (int)shape0[1], (int)shape0[2]);

                var output_tensor1 = infer_request.get_output_tensor(1);
                Shape maskShape = output_tensor1.get_shape();
                var output_length1 = (int)output_tensor1.get_size();
                float[] output_data1 = output_tensor1.get_data<float>(output_length1);
                var mask_data4D = ReshapeTo4D(output_data1, (int)maskShape[0], (int)maskShape[1], (int)maskShape[2], (int)maskShape[3]);


                //获取标签
                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                List<InferenceResult> InferenceResults = new List<InferenceResult>();
                for (int l = 0; l < images.Count; l++)
                {
                    ConcurrentBag<Boxes> boxes = new ConcurrentBag<Boxes>();
                    Bitmap image = images[l];

                    int ortImgWidth = image.Width;
                    int ortImgHeight = image.Height;
                    float ratio = Math.Min((float)targetWidth / ortImgWidth, (float)targetHeight / ortImgHeight);

                    // 计算填充的像素数
                    float padX = (targetWidth - ratio * ortImgWidth) / 2;
                    float padY = (targetHeight - ratio * ortImgHeight) / 2;

                    Parallel.For(0, shape0[1] - 4 - maskShape[1], j =>
                    {
                        for (int i = 0; i < shape0[2]; i++)
                        {
                            float score = output_data3D0[l, 4 + j, i];
                            if (score > confidence)
                            {
                                float x = output_data3D0[l, 0, i];
                                float y = output_data3D0[l, 1, i];
                                float w = output_data3D0[l, 2, i];
                                float h = output_data3D0[l, 3, i];
                                float r = 0;

                                // 将边界框信息还原到原图中
                                Boxes box = new Boxes()
                                {
                                    Confidence = score,
                                    LabelId = (int)j,
                                    LabelName = labelNames.Count > 0 ? labelNames[(int)j] : "",
                                    Angle = r,
                                    X = (x - padX) / ratio,
                                    Y = (y - padY) / ratio,
                                    Width = w / ratio,
                                    Height = h / ratio,
                                    BoxIndex = i
                                };

                                boxes.Add(box);

                            }
                        }
                    });

                    //NMS
                    List<Boxes> boxes1 = NonMaxSuppression(boxes.ToList(), iou);

                    // 处理分割掩码
                    foreach (var item in boxes1)
                    {
                        //计算mask权重值
                        var maskWeights = ExtractMaskWeights_Dyamics(output_data3D0, l, item.BoxIndex, (int)maskShape[1], (int)shape0[1] - (int)maskShape[1]);

                        //将按原图box值转成mask的尺寸并进行权重值计算得到mask(缩小遍历次数，每次只获取box范围内的mask值)
                        float x = (item.X * ratio + padX) / (targetWidth / maskShape[3]);
                        float y = (item.Y * ratio + padY) / (targetHeight / maskShape[2]);
                        float w = item.Width * ratio / (targetWidth / maskShape[3]);
                        float h = item.Height * ratio / (targetHeight / maskShape[2]);
                        int boxX = Math.Max(0, (int)(x - w / 2) - 1);
                        int boxY = Math.Max(0, (int)(y - h / 2) - 1);
                        int maxX = Math.Min(boxX + (int)w + 2, (int)maskShape[3]);
                        int maxY = Math.Min(boxY + (int)h + 2, (int)maskShape[2]);
                        float[,] mask = ProcessMask_Dyamics(maskShape, l, mask_data4D, maskWeights, boxX, boxY, maxX, maxY);

                        //将mask缩放至原图尺寸大小
                        item.Mask = ResizeMaskAfterLetterbox(mask, item, image.Width, image.Height);
                    }

                    InferenceResults.Add(new InferenceResult()
                    {
                        Boxes = boxes1,
                        ModelType = modelType,
                        ModelInfo = modelInfo,
                        OriginalImage = new Bitmap(image)
                    });

                    image?.Dispose();
                }
                time.Stop();
                inferTimeInfo.PostprocessTime = time.ElapsedMilliseconds;
                return InferenceResults.ToArray();
            }
            catch
            {
                inferTimeInfo = new InferTimeInfo();
                return null;
            }

        }
        #endregion

        #region Pose检测
        /*
         张量 output: float32[1, 56,8400] 的结构说明
         1: 批次大小 batch size，固定尺寸模型中推理过程中为 1。
         56: 4 个坐标值（bbox）  1 个置信度（confidence）  17 个关键点 × 3 个值（x, y, 可见性）
         8400: 预测框数量 模型在特征图上生成的初始锚框数量
         */
        private InferenceResult[] InferencePose_Dyamics(List<Bitmap> images, out InferTimeInfo inferTimeInfo, int targetWidth = 640, int targetHeight = 640, float confidence = 0.5f, float iou = 0.5f)
        {
            try
            {
                inferTimeInfo = new InferTimeInfo();
                Stopwatch time = new Stopwatch();

                time.Restart();
                //Resize图片
                var bitmap = LetterboxResize_Dyamics(images, targetWidth, targetHeight, Color.FromArgb(114, 114, 114));
                //图片归一化操作
                var tensor = NormalizeImages_Dyamics(bitmap);

                //设置输入张量形状
                inputTensor.set_shape(new Shape(new Int32[] { images.Count, 3, targetWidth, targetHeight })); // NCHW格式
                //输入预处理数据
                inputTensor.set_data<float>(tensor);

                time.Stop();
                inferTimeInfo.PreprocessTime = time.ElapsedMilliseconds;

                time.Restart();
                // 推理
                infer_request.infer();
                time.Stop();
                inferTimeInfo.InferenceTime = time.ElapsedMilliseconds;

                time.Restart();
                // 获取输出张量
                var output_tensor = infer_request.get_output_tensor();
                var shape = output_tensor.get_shape();
                var output_length = (int)output_tensor.get_size();
                float[] output_data = output_tensor.get_data<float>(output_length);
                var output_data3D = ReshapeTo3D(output_data, (int)shape[0], (int)shape[1], (int)shape[2]);

                Dictionary<int, string> labelNames = new Dictionary<int, string>();
                if (this.modelInfo != null)
                {
                    labelNames = JsonConvert.DeserializeObject<Dictionary<int, string>>(modelInfo.LabelNames);
                }

                List<InferenceResult> InferenceResults = new List<InferenceResult>();
                for (int l = 0; l < images.Count; l++)
                {
                    ConcurrentBag<Boxes> boxes = new ConcurrentBag<Boxes>();
                    Bitmap image = images[l];

                    int ortImgWidth = image.Width;
                    int ortImgHeight = image.Height;
                    float ratio = Math.Min((float)targetWidth / ortImgWidth, (float)targetHeight / ortImgHeight);

                    // 计算填充的像素数
                    float padX = (targetWidth - ratio * ortImgWidth) / 2;
                    float padY = (targetHeight - ratio * ortImgHeight) / 2;

                    Parallel.For(0, shape[2], i =>
                    {
                        float score = output_data3D[l, 4, i];
                        if (score > confidence)
                        {
                            float x = output_data3D[l, 0, i];
                            float y = output_data3D[l, 1, i];
                            float w = output_data3D[l, 2, i];
                            float h = output_data3D[l, 3, i];
                            float r = 0;

                            var keypoints = new KeyPoint[17]; // 17个关键点，每个关键点有x和y坐标
                            //获取关键点坐标信息
                            for (int j = 0; j < 17; j++)
                            {
                                int index = 5 + j * 3;
                                float keypointX = output_data3D[l, index, i];
                                float keypointY = output_data3D[l, index + 1, i];
                                float keypointVisibility = output_data3D[l, index + 2, i];
                                string keypointName = KeyPoint.KeypointNames[j];
                                keypoints[j] = new KeyPoint()
                                {
                                    X = (keypointX - padX) / ratio,
                                    Y = (keypointY - padY) / ratio,
                                    Visibility = keypointVisibility,
                                    Name = keypointName
                                };
                            }

                            // 将边界框信息还原到原图中
                            Boxes box = new Boxes()
                            {
                                Confidence = score,
                                LabelId = 0,
                                LabelName = labelNames[0],
                                Angle = r,
                                X = (x - padX) / ratio,
                                Y = (y - padY) / ratio,
                                Width = w / ratio,
                                Height = h / ratio,
                                KeyPoints = keypoints
                            };
                            boxes.Add(box);

                        }

                    });

                    //NMS
                    List<Boxes> boxes1 = NonMaxSuppression(boxes.ToList(), iou);

                    InferenceResults.Add(new InferenceResult()
                    {
                        Boxes = boxes1,
                        ModelType = modelType,
                        ModelInfo = modelInfo,
                        OriginalImage = new Bitmap(image)
                    });

                    image?.Dispose();
                }
                return InferenceResults.ToArray();

            }
            catch
            {
                inferTimeInfo = new InferTimeInfo();
                return null;
            }
        }
        #endregion

        #endregion

        #region 推理过程使用到的工具

        #region 分割推理计算掩码权重

        #region 固定尺寸模型
        /// <summary>
        /// 计算Mask权重值
        /// </summary>
        /// <param name="output"></param>
        /// <param name="boxIndex"></param>
        /// <param name="maskChannelCount"></param>
        /// <param name="maskWeightsOffset"></param>
        /// <returns></returns>
        private static float[] ExtractMaskWeights(float[,,] output, int boxIndex, int maskChannelCount, int maskWeightsOffset)
        {
            var maskWeights = new float[maskChannelCount];

            Parallel.For(0, maskChannelCount, i =>
            {
                maskWeights[i] = output[0, maskWeightsOffset + i, boxIndex];
            });

            return maskWeights;
        }

        /// <summary>
        /// 计算掩码160*160
        /// </summary>
        /// <param name="maskTensor"></param>
        /// <param name="maskWeights"></param>
        /// <returns></returns>
        private float[,] ProcessMask(Shape shape, float[,,,] maskTensor, float[] maskWeights, int X, int Y, int maxX, int maxY)
        {
            int w = (int)shape[3]; // 掩码尺寸160x160
            int h = (int)shape[2]; // 掩码尺寸160x160
            float[,] mask = new float[h, w];

            Parallel.For(Y, maxY, y =>
            {
                for (int x = X; x < maxX; x++)
                {
                    var value = 0f;
                    // 提取对应类别的掩码
                    for (int z = 0; z < shape[1]; z++)
                    {
                        value += maskTensor[0, z, y, x] * maskWeights[z];
                    }

                    mask[y, x] = Sigmoid(value);
                }
            });

            return mask;
        }

       

        /// <summary>
        /// 缩放掩码到任意尺寸
        /// </summary>
        /// <param name="mask"></param>
        /// <param name="box"></param>
        /// <param name="targetWidth"></param>
        /// <param name="targetHeight"></param>
        /// <returns></returns>
        public static byte[,] ResizeMaskAfterLetterbox(float[,] mask, Boxes box, int originalWidth, int originalHeight)
        {
            int maskWidth = mask.GetLength(1);  // 通常为 160
            int maskHeight = mask.GetLength(0); // 通常为 160

            // 计算 Letterbox 缩放比例和填充
            float scale = Math.Min((float)maskWidth / (float)originalWidth, (float)maskHeight / (float)originalHeight);
            float newWidth = (float)(originalWidth * scale);
            float newHeight = (float)(originalHeight * scale);

            float padX = (maskWidth - newWidth) / 2f;
            float padY = (maskHeight - newHeight) / 2f;

            // 计算边界框的宽度和高度
            int boxWidth = (int)box.Width;
            int boxHeight = (int)box.Height;

            // 边界框的起始位置 (左上角)
            int boxX = Math.Max(0, (int)(box.X - boxWidth / 2));
            int boxY = Math.Max(0, (int)(box.Y - boxHeight / 2));


            // 确保掩码和边界框在原图中不越界
            int maxX = Math.Min(boxX + boxWidth, originalWidth);
            int maxY = Math.Min(boxY + boxHeight, originalHeight);

            // 创建目标数组
            byte[,] resizedMask = new byte[originalHeight, originalWidth];

            // 遍历原图的像素
            Parallel.For(boxY, maxY, y =>
            {
                for (int x = boxX; x < maxX; x++)
                {
                    // 将原图中的点映射到缩放后的掩码坐标
                    float srcX = (x * scale) + padX - 0.5f;
                    float srcY = (y * scale) + padY - 0.5f;

                    if (srcX >= 0 && srcX < maskWidth && srcY >= 0 && srcY < maskHeight)
                    {
                        // 获取周围的像素索引
                        int x0 = Math.Max((int)Math.Floor(srcX), 0);
                        int x1 = Math.Min(x0 + 1, maskWidth - 1);
                        int y0 = Math.Max((int)Math.Floor(srcY), 0);
                        int y1 = Math.Min(y0 + 1, maskHeight - 1);

                        // 获取插值权重
                        float wx = srcX - x0;
                        float wy = srcY - y0;

                        // 双线性插值
                        float top = mask[y0, x0] * (1 - wx) + mask[y0, x1] * wx;
                        float bottom = mask[y1, x0] * (1 - wx) + mask[y1, x1] * wx;
                        resizedMask[y, x] = (top * (1 - wy) + bottom * wy) > 0.5 ? (byte)1 : (byte)0;
                    }
                    else
                    {
                        // 填充区域外的值
                        resizedMask[y, x] = 0; // 默认背景
                    }
                }
            });

            return resizedMask;
        }
        #endregion

        #region 动态尺寸模型
        /// <summary>
        /// 计算Mask权重值
        /// </summary>
        /// <param name="output"></param>
        /// <param name="boxIndex"></param>
        /// <param name="maskChannelCount"></param>
        /// <param name="maskWeightsOffset"></param>
        /// <returns></returns>
        private static float[] ExtractMaskWeights_Dyamics(float[,,] output, int index, int boxIndex, int maskChannelCount, int maskWeightsOffset)
        {
            var maskWeights = new float[maskChannelCount];

            Parallel.For(0, maskChannelCount, i =>
            {
                maskWeights[i] = output[index, maskWeightsOffset + i, boxIndex];
            });

            return maskWeights;
        }

        /// <summary>
        /// 计算掩码160*160
        /// </summary>
        /// <param name="maskTensor"></param>
        /// <param name="maskWeights"></param>
        /// <returns></returns>
        private float[,] ProcessMask_Dyamics(Shape shape, int index, float[,,,] maskTensor, float[] maskWeights, int X, int Y, int maxX, int maxY)
        {
            int w = (int)shape[3]; // 掩码尺寸160x160
            int h = (int)shape[2]; // 掩码尺寸160x160
            float[,] mask = new float[h, w];

            Parallel.For(Y, maxY, y =>
            {
                for (int x = X; x < maxX; x++)
                {
                    var value = 0f;
                    // 提取对应类别的掩码
                    for (int z = 0; z < shape[1]; z++)
                    {
                        value += maskTensor[index, z, y, x] * maskWeights[z];
                    }

                    mask[y, x] = Sigmoid(value);
                }
            });

            return mask;
        }
        #endregion
        private static float Sigmoid(float value)
        {
            var k = Math.Exp(value);
            return Convert.ToSingle(k / (1.0f + k));
        }
        #endregion

        #region 图片按大小格式化

        #region 固定输入尺寸模型
        /// <summary>
        /// 图像居中resize
        /// </summary>
        /// <param name="image"></param>
        /// <param name="targetWidth"></param>
        /// <param name="targetHeight"></param>
        /// <param name="paddingColor"></param>
        /// <returns></returns>
        public Bitmap LetterboxResize(Bitmap image, int targetWidth, int targetHeight, Color paddingColor)
        {
            int originalWidth = image.Width;
            int originalHeight = image.Height;

            // 计算缩放比
            float ratio = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

            // 缩放后的图像尺寸
            int newWidth = (int)(originalWidth * ratio);
            int newHeight = (int)(originalHeight * ratio);

            // 计算需要的填充
            int padX = (targetWidth - newWidth) / 2;
            int padY = (targetHeight - newHeight) / 2;

            // 创建目标图像并设置背景颜色（填充颜色）
            Bitmap result = new Bitmap(targetWidth, targetHeight, PixelFormat.Format32bppArgb);
            using (Graphics graphics = Graphics.FromImage(result))
            {
                // 填充背景
                graphics.Clear(paddingColor);

                // 设置高质量插值模式，保证缩放效果
                //graphics.InterpolationMode = InterpolationMode.HighQualityBilinear;
                //graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                // 将缩放后的图像绘制到居中的位置
                graphics.DrawImage(image, padX, padY, newWidth, newHeight);
            }

            return result;
        }
        #endregion

        #region 动态输入尺寸模型
        public List<Bitmap> LetterboxResize_Dyamics(List<Bitmap> images, int targetWidth, int targetHeight, Color paddingColor)
        {
            List<Bitmap> resizedImages = new List<Bitmap>(images.Count);

            foreach (var item in images)
            {
                resizedImages.Add(LetterboxResize(item, targetWidth, targetHeight, paddingColor));
            }
            return resizedImages;
        }
        #endregion

        #endregion

        #region 图片归一化

        #region 固定尺寸模型
        public float[] NormalizeImage(Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            int pixelCount = width * height;
            var rectangle = new Rectangle(0, 0, width, height);
            float[] normalizedData = new float[3 * pixelCount]; // 3 通道展开为一维数组
            Span<byte> data;

            BitmapData bitmapData;
            if (bitmap.PixelFormat == PixelFormat.Format24bppRgb && width % 4 == 0)
            {
                bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

                unsafe
                {
                    data = new Span<byte>((void*)bitmapData.Scan0, bitmapData.Height * bitmapData.Stride);
                }

                ExtractPixelsRgb(normalizedData, data, pixelCount);
            }
            else
            {
                // 强制转换为 32 位 PArgb 格式
                bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);

                unsafe
                {
                    data = new Span<byte>((void*)bitmapData.Scan0, bitmapData.Height * bitmapData.Stride);
                }

                ExtractPixelsArgb(normalizedData, data, pixelCount);
            }

            bitmap.UnlockBits(bitmapData);

            return normalizedData;
        }

        public void ExtractPixelsArgb(float[] normalizedData, Span<byte> data, int pixelCount)
        {
            int sidx = 0; // 源数据索引
            int didx = 0; // 目标数组索引
            for (int i = 0; i < pixelCount; i++)
            {
                normalizedData[didx] = data[sidx + 2] / 255.0F; // R 通道
                normalizedData[didx + pixelCount] = data[sidx + 1] / 255.0F; // G 通道
                normalizedData[didx + 2 * pixelCount] = data[sidx] / 255.0F; // B 通道
                didx++;
                sidx += 4;
            }
        }

        public void ExtractPixelsRgb(float[] normalizedData, Span<byte> data, int pixelCount)
        {
            int sidx = 0; // 源数据索引
            int didx = 0; // 目标数组索引
            for (int i = 0; i < pixelCount; i++)
            {
                normalizedData[didx] = data[sidx + 2] / 255.0F; // R 通道
                normalizedData[didx + pixelCount] = data[sidx + 1] / 255.0F; // G 通道
                normalizedData[didx + 2 * pixelCount] = data[sidx] / 255.0F; // B 通道
                didx++;
                sidx += 3;
            }
        }

        #endregion

        #region 动态尺寸模型
        public float[] NormalizeImages_Dyamics(List<Bitmap> bitmaps)
        {
            List<float[]> floats = new List<float[]>(bitmaps.Count);

            foreach (var item in bitmaps)
            {
                floats.Add(NormalizeImage(item));
            }
            
            return ConcatenateFloatArrays(floats); 
        }

        public  float[] ConcatenateFloatArrays(List<float[]> arrays)
        {
            // 计算总长度
            int totalLength = 0;
            foreach (var array in arrays)
            {
                totalLength += array.Length;
            }

            // 创建目标数组
            float[] result = new float[totalLength];
            int offset = 0;

            // 逐个拷贝数据
            foreach (var array in arrays)
            {
                array.CopyTo(result, offset);
                offset += array.Length;
            }

            return result;
        }
        #endregion

        #endregion

        #region NMS
        private float ComputeIoU(Boxes rect1, Boxes rect2)
        {
            // 这里仅提供非旋转矩形的IoU计算方法
            // 对于旋转矩形，您可能需要使用更复杂的几何算法或库

            float left = Math.Max(rect1.X - rect1.Width / 2, rect2.X - rect2.Width / 2);
            float right = Math.Min(rect1.X + rect1.Width / 2, rect2.X + rect2.Width / 2);
            float top = Math.Max(rect1.Y - rect1.Height / 2, rect2.Y - rect2.Height / 2);
            float bottom = Math.Min(rect1.Y + rect1.Height / 2, rect2.Y + rect2.Height / 2);

            float width = right - left;
            float height = bottom - top;

            if (width <= 0 || height <= 0)
                return 0;

            float intersectionArea = width * height;
            float rect1Area = rect1.Width * rect1.Height;
            float rect2Area = rect2.Width * rect2.Height;

            return intersectionArea / (rect1Area + rect2Area - intersectionArea);
        }

        private List<Boxes> NonMaxSuppression(List<Boxes> rectangles, float iouThreshold)
        {
            List<Boxes> pickedBoxes = new List<Boxes>();

            // 根据置信度对边界框进行排序
            rectangles = rectangles.OrderByDescending(box => box.Confidence).ToList();

            while (rectangles.Count > 0)
            {
                // 选择具有最高置信度的边界框
                Boxes topBox = rectangles[0];
                pickedBoxes.Add(topBox);
                rectangles.RemoveAt(0);

                // 删除与所选框重叠面积大于阈值的其他框
                List<Boxes> overlappingBoxes = new List<Boxes>();
                foreach (Boxes box in rectangles)
                {
                    if (ComputeIoU(topBox, box) > iouThreshold)
                    {
                        overlappingBoxes.Add(box);
                    }
                }
                foreach (Boxes box in overlappingBoxes)
                {
                    rectangles.Remove(box);
                }
            }

            return pickedBoxes;
        }
        #endregion

        #region 一维数组转成多维数组(模拟张量reshape操作)
        private float[,,] ReshapeTo3D(float[] inputArray, int dim1, int dim2, int dim3)
        {
            // 检查输入数组的长度是否与目标形状匹配
            if (inputArray.Length != dim1 * dim2 * dim3)
            {
                throw new ArgumentException("输入数组的长度与目标形状不匹配。");
            }

            // 创建一个目标三维数组
            float[,,] outputArray = new float[dim1, dim2, dim3];

            // 遍历输入数组并将值填充到三维数组中
            int index = 0;
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    for (int k = 0; k < dim3; k++)
                    {
                        outputArray[i, j, k] = inputArray[index++];
                    }
                }
            }

            return outputArray;
        }

        private float[,,,] ReshapeTo4D(float[] inputArray, int dim1, int dim2, int dim3, int dim4)
        {
            // 检查输入数组的长度是否与目标形状匹配
            if (inputArray.Length != dim1 * dim2 * dim3 * dim4)
            {
                throw new ArgumentException("输入数组的长度与目标形状不匹配。");
            }

            // 创建一个目标四维数组
            float[,,,] outputArray = new float[dim1, dim2, dim3, dim4];

            // 遍历输入数组并将值填充到四维数组中
            int index = 0;
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    for (int k = 0; k < dim3; k++)
                    {
                        for (int l = 0; l < dim4; l++)
                        {
                            outputArray[i, j, k, l] = inputArray[index++];
                        }
                    }
                }
            }

            return outputArray;
        }

        #endregion 
        #endregion

    }
}
