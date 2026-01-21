using ByteTrackBase;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using YoloOnnx;

namespace YoloOnnxByteTracker
{
    public class OpenVinoTracker:IDisposable
    {
        Yolo_OpenVINO yolo = new Yolo_OpenVINO();
        ModelInfo modelInfo;
        VideoCapture vc;
        public event Action<FrameResult> GetFrameEvent;
        public event Action StartEvent;
        public event Action StopEvent;

        public OpenVinoTracker()
        {

        }

        public ModelInfo LoadModel(string path, string device_name)
        {
            modelInfo = yolo.LoadOnnxModel(path, device_name);
            return modelInfo;
        }

        public bool startFlag = false;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="videoPath">视频路径</param>
        /// <param name="trackBuffer">跟踪缓冲区的大小 表示跟踪器在丢失目标后，仍然保留目标信息的时间窗口。这个值越大，跟踪器对目标的“记忆”时间越长，但可能会增加计算复杂度</param>
        /// <param name="trackThresh">跟踪阈值 用于决定一个检测结果是否足够可靠，可以被初始化为一个新的跟踪目标。通常是一个置信度阈值，只有置信度高于此阈值的检测结果才会被考虑</param>
        /// <param name="highThresh">高置信度阈值 用于区分高置信度和低置信度的检测结果。高置信度的检测结果通常会被优先处理，而低置信度的检测结果可能会被进一步验证或忽略</param>
        /// <param name="matchThresh">匹配阈值 用于决定两个检测结果是否足够相似，可以被认为是同一个目标。通常是一个相似度阈值，只有相似度高于此阈值的检测结果才会被认为是同一个目标</param>
        public void Start(string videoPath, int trackBuffer = 30, float trackThresh = 0.5f, float highThresh = 0.6f, float matchThresh = 0.8f, float detectionThresh = 0.5f, float iou = 0.5f)
        {
            if (startFlag)
            {
                return;
            }
            startFlag = true;
            Thread thread = new Thread(() =>
            {
                StartEvent?.Invoke();
                using (vc = new VideoCapture(videoPath))
                {
                    var tracker = new ByteTracker((int)vc.Fps, trackBuffer, trackThresh, highThresh, matchThresh);
                    Stopwatch time = new Stopwatch();
                    while (vc.Grab())
                    {
                        time.Restart();
                        var src = vc.RetrieveMat();
                        if (src.Empty()) continue;
                        // 确保 Mat 数据是连续的
                        if (!src.IsContinuous()) continue;
                        var img = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(src);

                        var results = yolo.Inference(img, detectionThresh, iou);
                        DetectionResult[] detectionResults = new DetectionResult[results.Boxes.Count];

                        int index = 0;
                        foreach (var item in results.Boxes)
                        {
                            DetectionResult detectionResult = new DetectionResult
                               (
                               new RectBox(item.X - item.Width / 2, item.Y - item.Height / 2, item.Width, item.Height),
                               item.LabelName,
                               item.LabelId,
                               item.Confidence);

                            detectionResults[index] = detectionResult;
                            index++;
                        }
                        var trackOutputs = tracker.Update(detectionResults);
                        foreach (var t in trackOutputs)
                        {
                            OpenCvSharp.Rect rect = new OpenCvSharp.Rect((int)t.RectBox.X, (int)t.RectBox.Y, (int)t.RectBox.Width, (int)t.RectBox.Height);
                            OpenCvSharp.Point point = new OpenCvSharp.Point();
                            point.X = rect.TopLeft.X;
                            point.Y = rect.TopLeft.Y - 6;
                            // 计算文字大小
                            var text = $"ID:{t.TrackId},{t["name"]},{t.Score:P0}";
                            var textSize = Cv2.GetTextSize(text, HersheyFonts.Italic, 0.5, 1, out var baseline);

                            // 计算背景矩形位置（稍微扩大一些）
                            var backgroundRectTopLeft = new OpenCvSharp.Point(point.X, point.Y - textSize.Height);
                            var backgroundRectBottomRight = new OpenCvSharp.Point(point.X + textSize.Width, point.Y + baseline);

                            // 绘制半透明背景（BGR 颜色）
                            Cv2.Rectangle(src, backgroundRectTopLeft, backgroundRectBottomRight, new Scalar(100, 200, 0), -1); // -1 表示填充

                            // 绘制文字（现在文字会有背景）
                            Cv2.PutText(src, text, point, HersheyFonts.Italic, 0.5, Scalar.Blue, 1);

                            Cv2.Rectangle(src, rect, Scalar.LimeGreen, thickness: 2);
                        }

                        var outputImg = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(src);
                        time.Stop();
                        int fps = (int)(1000 / time.ElapsedMilliseconds);
                        FrameResult frameResult = new FrameResult()
                        {
                            Image = outputImg,
                            FPS = fps,
                        };
                        GetFrameEvent?.Invoke(frameResult);

                        if (!startFlag)
                        {
                            break;
                        }
                        Thread.Sleep(1);
                    }
                    StopEvent?.Invoke();
                    startFlag = false;
                }
            });
            thread.IsBackground = true;
            thread.Start();


        }

        public void Dispose()
        {
            yolo.Dispose();
        }
    }
}
