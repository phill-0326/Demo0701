using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net.Mail;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace YoloOnnx
{
    public class InferenceResult
    {
        /// <summary>
        /// 推理类型
        /// </summary>
        public ModelType ModelType { get; set; }
        /// <summary>
        /// 边界框
        /// </summary>
        public List<Boxes> Boxes { get; set; }

        /// <summary>
        /// 分类检测结果
        /// </summary>
        public ClassifyRes ClassifyRes { get; set; }

        /// <summary>
        /// 原图
        /// </summary>
        public Bitmap OriginalImage { get; set; }

        /// <summary>
        /// 模型信息
        /// </summary>
        public ModelInfo ModelInfo { get; set; }

        /// <summary>
        /// 预处理耗时(ms)
        /// </summary>
        public long PreprocessTime { get; set; }

        /// <summary>
        /// 推理耗时(ms)
        /// </summary>
        public long InferenceTime { get; set; }

        /// <summary>
        /// 后处理耗时(ms)
        /// </summary>
        public long PostprocessTime { get; set; }

        public Bitmap DrawReg(Color labelBoxColor, bool isShowLabel = true)
        {
            Bitmap img = null;
            switch (this.ModelType)
            {
                case ModelType.detect:
                    img = DrawRects(OriginalImage, ModelInfo, Boxes, labelBoxColor, isShowLabel);
                    break;
                case ModelType.obb:
                    img = DrawRects(OriginalImage, ModelInfo, Boxes, labelBoxColor, isShowLabel);
                    break;
                case ModelType.classify:
                    img = DrawClassifyRes(OriginalImage, ModelInfo, ClassifyRes, labelBoxColor, isShowLabel);
                    break;
                case ModelType.segment:
                    img = DrawRectsAndMasks(OriginalImage, ModelInfo, Boxes, labelBoxColor, isShowLabel);
                    break;
                case ModelType.pose:
                    img = DrawRectsAndPose(OriginalImage, ModelInfo, Boxes, labelBoxColor, isShowLabel);
                    break;
                    
            }

            return img;
        }
        #region 绘制结果

        #region 绘制边界框
        private Bitmap DrawRects(Bitmap img, ModelInfo modelInfo, List<Boxes> boxes, Color color, bool showLabel = true)
        {
            Pen pen = new Pen(color, 2f);
            int resize = 640;
            if (modelInfo != null)
            {
                resize = Convert.ToInt16(modelInfo.Imgsz.Replace("[", "").Replace("]", "").Split(',')[0]);
            }
            int fontSize = Math.Max(img.Width, img.Height) / resize * 15;
            fontSize = fontSize <= 15 ? 15 : fontSize;
            Font font = new Font("Arial", fontSize, FontStyle.Bold | FontStyle.Italic);

            Bitmap newImage = new Bitmap(img.Width, img.Height, PixelFormat.Format32bppArgb);
            SolidBrush solidBrush = new SolidBrush(color);
            using (Graphics g = Graphics.FromImage(newImage))
            {
                g.Clear(Color.Transparent);
                g.DrawImage(img, 0, 0, img.Width, img.Height);
                foreach (var item in boxes)
                {

                    float tempDegress = (float)(item.Angle * 180 / Math.PI);
                    var s = g.Save();

                    // 将画布中心移动到矩形中心点
                    g.TranslateTransform(item.X, item.Y);
                    g.RotateTransform(tempDegress);

                    if (showLabel)
                    {
                        string text = "";
                        if (modelInfo != null)
                        {

                            if (!string.IsNullOrEmpty(item.LabelName))
                            {
                                text = $"{item.LabelName},{(item.Confidence * 100).ToString("f1")}%";
                            }
                            else
                            {
                                text = $"{item.LabelId},{(item.Confidence * 100).ToString("f1")}%";
                            }
                        }
                        else
                        {
                            text = $"{item.LabelId},{(item.Confidence * 100).ToString("f1")}%";
                        }
                        float strX = -item.Width / 2;
                        float strY = -item.Height / 2;
                        // 测量文本的尺寸
                        SizeF textSize = g.MeasureString(text, font);
                        strY = strY - textSize.Height;
                        // 创建一个带有透明度的填充刷 (半透明黑色背景)
                        Color backgroundColor = Color.FromArgb(200, Color.LimeGreen); // Alpha 128 表示 50% 透明
                        using (Brush backgroundBrush = new SolidBrush(backgroundColor))
                        {
                            g.FillRectangle(backgroundBrush, strX, strY, textSize.Width, textSize.Height);
                        }
                        // 绘制文本
                        g.DrawString(text, font, solidBrush, strX, strY);
                    }

                    g.DrawRectangle(pen, -item.Width / 2, -item.Height / 2, item.Width, item.Height);
                    g.Restore(s);

                }
                img?.Dispose();
                return newImage;
            }

        }


        #endregion

        #region 分割绘制

        #region 绘制边界框和掩码
        private Bitmap DrawRectsAndMasks(Bitmap img, ModelInfo modelInfo, List<Boxes> boxes, Color color, bool showLabel = true)
        {
            Pen pen = new Pen(color, 2f);
            int resize = 640;
            if (modelInfo != null)
            {
                resize = Convert.ToInt16(modelInfo.Imgsz.Replace("[", "").Replace("]", "").Split(',')[0]);
            }
            int fontSize = Math.Max(img.Width, img.Height) / resize * 15;
            fontSize = fontSize <= 15 ? 15 : fontSize;
            Font font = new Font("Arial", fontSize, FontStyle.Bold | FontStyle.Italic);
            Bitmap newImage = new Bitmap(img.Width, img.Height, PixelFormat.Format32bppArgb);
            SolidBrush solidBrush = new SolidBrush(Color.Blue);

            // 创建画布
            using (Graphics g = Graphics.FromImage(newImage))
            {
                g.Clear(Color.Transparent);
                g.DrawImage(img, 0, 0, img.Width, img.Height);  // 绘制原始图像
              
                foreach (var item in boxes)
                {
                    // 绘制分割掩码
                    if (item.Mask != null)
                    {
                        unsafe
                        {
                            BitmapData bitmapData = newImage.LockBits(
                                new Rectangle(0, 0, newImage.Width, newImage.Height),
                                ImageLockMode.ReadWrite,
                                PixelFormat.Format32bppArgb // 确保使用 32 位像素格式
                            );

                            int maskWidth = item.Mask.GetLength(1);
                            int maskHeight = item.Mask.GetLength(0);

                            byte* ptr = (byte*)bitmapData.Scan0; // 指向图像数据的指针
                            int stride = bitmapData.Stride;     // 每行的字节数（可能包含填充）

                            Parallel.For(0, maskHeight, y =>
                            {
                                for (int x = 0; x < maskWidth; x++)
                                {
                                    if (item.Mask[y, x] == 1) // 掩码值大于0表示前景区域
                                    {
                                        // 计算当前像素在图像数据中的位置
                                        byte* pixel = ptr + y * stride + x * 4;

                                        // 获取原始颜色
                                        byte originalB = pixel[0];
                                        byte originalG = pixel[1];
                                        byte originalR = pixel[2];
                                        byte originalA = pixel[3];

                                        // 半透明绿色遮罩
                                        byte maskA = 200;
                                        byte maskR = color.R;
                                        byte maskG = color.G;
                                        byte maskB = color.B;

                                        // 混合颜色
                                        pixel[0] = (byte)((maskB + originalB) / 2); // 蓝色通道
                                        pixel[1] = (byte)((maskG + originalG) / 2); // 绿色通道
                                        pixel[2] = (byte)((maskR + originalR) / 2); // 红色通道
                                        pixel[3] = (byte)((maskA + originalA) / 2); // Alpha 通道
                                    }
                                }
                            });

                            newImage.UnlockBits(bitmapData); // 解锁图像数据
                        }

                    }

                    // 绘制边界框
                    float tempDegress = (float)(item.Angle * 180 / Math.PI);
                    var s = g.Save();

                    // 将画布中心移动到矩形中心点
                    g.TranslateTransform(item.X, item.Y);

                    if (showLabel)
                    {
                        string text = "";
                        if (modelInfo != null)
                        {
                            if (!string.IsNullOrEmpty(item.LabelName))
                            {
                                text = $"{item.LabelName},{(item.Confidence * 100).ToString("f1")}%";
                            }
                            else
                            {
                                text = $"{item.LabelId},{(item.Confidence * 100).ToString("f1")}%";
                            }
                        }
                        else
                        {
                            text = $"{item.LabelId},{(item.Confidence * 100).ToString("f1")}%";
                        }
                        float strX = -item.Width / 2;
                        float strY = -item.Height / 2;
                        // 测量文本的尺寸
                        SizeF textSize = g.MeasureString(text, font);
                        strY = strY - textSize.Height;
                        // 创建一个带有透明度的填充刷 (半透明黑色背景)
                        Color backgroundColor = Color.FromArgb(200, Color.LimeGreen); // Alpha 128 表示 50% 透明
                        using (Brush backgroundBrush = new SolidBrush(backgroundColor))
                        {
                            g.FillRectangle(backgroundBrush, strX, strY, textSize.Width, textSize.Height);
                        }
                        // 绘制文本
                        g.DrawString(text, font, solidBrush, strX, strY);
                    }

                    g.RotateTransform(tempDegress);
                    g.DrawRectangle(pen, -item.Width / 2, -item.Height / 2, item.Width, item.Height);
                    g.Restore(s);
                }
                img?.Dispose();
                return newImage;
            }

        }


        #endregion

        #region Mask byte[,]转成bitmap图
        /// <summary>
        /// Mask byte[,]转成bitmap图
        /// </summary>
        /// <param name="mask"></param>
        /// <returns></returns>
        public static Bitmap MaskToImage(byte[,] mask)
        {
            int maskWidth = mask.GetLength(1);
            int maskHeight = mask.GetLength(0);

            // 创建一个位图
            Bitmap newImage = new Bitmap(maskWidth, maskHeight, PixelFormat.Format32bppArgb);

            unsafe
            {
                BitmapData bitmapData = newImage.LockBits(
                    new Rectangle(0, 0, newImage.Width, newImage.Height),
                    ImageLockMode.ReadWrite,
                    PixelFormat.Format32bppArgb // 确保使用 32 位像素格式
                );

                byte* ptr = (byte*)bitmapData.Scan0; // 指向图像数据的指针
                int stride = bitmapData.Stride;     // 每行的字节数（可能包含填充）

                Parallel.For(0, maskHeight, y =>
                {
                    for (int x = 0; x < maskWidth; x++)
                    {
                        // 计算当前像素在图像数据中的位置
                        byte* pixel = ptr + y * stride + x * 4;

                        if (mask[y, x] == 1) // 掩码值为1表示前景区域
                        {
                            // 设置为白色
                            pixel[0] = 255; // 蓝色通道
                            pixel[1] = 255; // 绿色通道
                            pixel[2] = 255; // 红色通道
                            pixel[3] = 255; // Alpha通道（完全不透明）
                        }
                        else
                        {
                            // 设置为黑色
                            pixel[0] = 0; // 蓝色通道
                            pixel[1] = 0; // 绿色通道
                            pixel[2] = 0; // 红色通道
                            pixel[3] = 255; // Alpha通道（完全不透明）
                        }
                    }
                });

                newImage.UnlockBits(bitmapData); // 解锁图像数据
            }
            return newImage;
        }

        /// <summary>
        /// 获取每个边界框中的独立Mask图
        /// </summary>
        /// <returns></returns>
        public Bitmap[] GetMaskImages()
        {
            int maskWidth = OriginalImage.Width;
            int maskHeight = OriginalImage.Height;

            List<Bitmap> images = new List<Bitmap>();

            foreach (var item in Boxes)
            {
                unsafe
                {


                    // 创建一个位图
                    Bitmap newImage = new Bitmap(maskWidth, maskHeight, PixelFormat.Format32bppArgb);

                    BitmapData bitmapData = newImage.LockBits(new Rectangle(0, 0, newImage.Width, newImage.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

                    byte* ptr = (byte*)bitmapData.Scan0; // 指向图像数据的指针
                    int stride = bitmapData.Stride;     // 每行的字节数（可能包含填充）
                    Parallel.For(0, maskHeight, y =>
                    {
                        for (int x = 0; x < maskWidth; x++)
                        {
                            // 计算当前像素在图像数据中的位置
                            byte* pixel = ptr + y * stride + x * 4;

                            if (item.Mask[y, x] == 1) // 掩码值为1表示前景区域
                            {
                                // 设置为白色
                                pixel[0] = 255; // 蓝色通道
                                pixel[1] = 255; // 绿色通道
                                pixel[2] = 255; // 红色通道
                                pixel[3] = 255; // Alpha通道（完全不透明）
                            }
                        }
                    });

                    newImage.UnlockBits(bitmapData); // 解锁图像数据
                    images.Add(newImage);
                }

            }
            return images.ToArray();
        }

        /// <summary>
        /// 获取所有边界框的Mask图合并成一张图输出
        /// </summary>
        /// <returns></returns>
        public Bitmap GetCombineMaskImage()
        {
            int maskWidth = OriginalImage.Width;
            int maskHeight = OriginalImage.Height;
            // 创建一个位图
            Bitmap newImage = new Bitmap(maskWidth, maskHeight, PixelFormat.Format32bppArgb);
            foreach (var item in Boxes)
            {
                unsafe
                {
                    BitmapData bitmapData = newImage.LockBits(new Rectangle(0, 0, newImage.Width, newImage.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

                    byte* ptr = (byte*)bitmapData.Scan0; // 指向图像数据的指针
                    int stride = bitmapData.Stride;     // 每行的字节数（可能包含填充）


                    Parallel.For(0, maskHeight, y =>
                    {
                        for (int x = 0; x < maskWidth; x++)
                        {
                            // 计算当前像素在图像数据中的位置
                            byte* pixel = ptr + y * stride + x * 4;

                            if (item.Mask[y, x] == 1) // 掩码值为1表示前景区域
                            {
                                // 设置为白色
                                pixel[0] = 255; // 蓝色通道
                                pixel[1] = 255; // 绿色通道
                                pixel[2] = 255; // 红色通道
                                pixel[3] = 255; // Alpha通道（完全不透明）
                            }
                        }
                    });

                    newImage.UnlockBits(bitmapData); // 解锁图像数据
                }

            }
            return newImage;
        }
        #endregion

        #endregion

        #region 显示分类结果
        private Bitmap DrawClassifyRes(Bitmap img, ModelInfo modelInfo, ClassifyRes classifyRes, Color color, bool showLabel = true)
        {
            Pen pen = new Pen(color, 2f);
            int resize = 640;
            if (modelInfo != null)
            {
                resize = Convert.ToInt16(modelInfo.Imgsz.Replace("[", "").Replace("]", "").Split(',')[0]);
            }
            int fontSize = Math.Max(img.Width, img.Height) / resize * 5;
            fontSize = fontSize <= 5 ? 5 : fontSize;
            Font font = new Font("Arial", fontSize);
            Bitmap newImage = new Bitmap(img.Width, img.Height, PixelFormat.Format32bppArgb);
            SolidBrush solidBrush = new SolidBrush(color);
            using (Graphics g = Graphics.FromImage(newImage))
            {
                g.Clear(Color.Transparent);
                g.DrawImage(img, 0, 0, img.Width, img.Height);

                // 将画布中心移动到矩形20,20
                g.TranslateTransform(20, 20);

                if (showLabel)
                {
                    if (classifyRes.AllClassifyResultIndex.Count() > 0)
                    {
                        string text = "";
                        if (modelInfo != null)
                        {
                            if (!string.IsNullOrEmpty(classifyRes.MaxClassifyResultLabelName))
                            {
                                text = $"最大置信度:{classifyRes.MaxClassifyResultLabelName},{(classifyRes.MaxConfidence * 100).ToString("f1")}%\n";
                                for (int i = 0; i < classifyRes.AllClassifyResultIndex.Count(); i++)
                                {
                                    text += $"{i + 1}: {classifyRes.AllClassifyResultLabelName[i]},{(classifyRes.AllConfidence[i] * 100).ToString("f1")}%\n";
                                }
                            }
                            else
                            {
                                text = $"最大置信度:{classifyRes.MaxClassifyResultIndex},{(classifyRes.MaxConfidence * 100).ToString("f1")}%\n";
                                for (int i = 0; i < classifyRes.AllClassifyResultIndex.Count(); i++)
                                {
                                    text += $"{i + 1}: {classifyRes.AllClassifyResultIndex[i]},{(classifyRes.AllConfidence[i] * 100).ToString("f1")}%\n";
                                }
                            }
                        }
                        else
                        {
                            text = $"最大置信度:{classifyRes.MaxClassifyResultIndex},{(classifyRes.MaxConfidence * 100).ToString("f1")}%\n";
                            for (int i = 0; i < classifyRes.AllClassifyResultIndex.Count(); i++)
                            {
                                text += $"{i + 1}: {classifyRes.AllClassifyResultIndex[i]},{(classifyRes.AllConfidence[i] * 100).ToString("f1")}%\n";
                            }
                        }

                        float strX = 0;
                        float strY = 0;
                        // 测量文本的尺寸
                        SizeF textSize = g.MeasureString(text, font);
                        // 创建一个带有透明度的填充刷 (半透明黑色背景)
                        Color backgroundColor = Color.FromArgb(200, Color.LimeGreen); // Alpha 128 表示 50% 透明
                        using (Brush backgroundBrush = new SolidBrush(backgroundColor))
                        {
                            g.FillRectangle(backgroundBrush, strX, strY, textSize.Width, textSize.Height);
                        }
                        // 绘制文本
                        g.DrawString(text, font, solidBrush, strX, strY);
                    }
                }
                img?.Dispose();

                return newImage;
            }
        }
        #endregion

        #region 绘制姿态检测结果
        private Bitmap DrawRectsAndPose(Bitmap img, ModelInfo modelInfo, List<Boxes> boxes, Color color, bool showLabel = true)
        {
            Pen pen = new Pen(color, 2f);
            int resize = 640;
            if (modelInfo != null)
            {
                resize = Convert.ToInt16(modelInfo.Imgsz.Replace("[", "").Replace("]", "").Split(',')[0]);
            }
            int fontSize = Math.Max(img.Width, img.Height) / resize * 15;
            fontSize = fontSize <= 15 ? 15 : fontSize;
            Font font = new Font("Arial", fontSize, FontStyle.Bold | FontStyle.Italic);

            Bitmap newImage = new Bitmap(img.Width, img.Height, PixelFormat.Format32bppArgb);
            SolidBrush solidBrush = new SolidBrush(color);
            using (Graphics g = Graphics.FromImage(newImage))
            {
                g.Clear(Color.Transparent);
                g.DrawImage(img, 0, 0, img.Width, img.Height);
                foreach (var item in boxes)
                {

                    float tempDegress = (float)(item.Angle * 180 / Math.PI);
                    var s = g.Save();

                   
                    // 绘制骨骼连接线
                    using (var skeletonPen = new Pen(Color.Blue, 3))
                    {
                        foreach (var (startIdx, endIdx) in SkeletonConnections)
                        {
                            var kp1 = item.KeyPoints[startIdx];
                            var kp2 = item.KeyPoints[endIdx];

                            // 只绘制可见的关键点之间的连线
                            if (kp1.Visibility > 0.5f && kp2.Visibility > 0.5f)
                            {
                                g.DrawLine(skeletonPen,
                                    new PointF (kp1.X, kp1.Y),
                                    new PointF(kp2.X, kp2.Y));
                            }
                        }
                    }

                    // 绘制关键点
                    for (int i = 0; i < item.KeyPoints.Length; i++)
                    {
                        var kp = item.KeyPoints[i];
                        if (kp.Visibility <= 0.5f) continue;

                        float pointSize = i == 0 ? 10  : 8 ; // 鼻子画大一点
                        using (var brush = new SolidBrush(KeypointColors[i]))
                        {
                            g.FillEllipse(brush,
                                kp.X  - pointSize / 2,
                                kp.Y  - pointSize / 2,
                                pointSize, pointSize);
                        }

                    }

                    // 将画布中心移动到矩形中心点
                    g.TranslateTransform(item.X, item.Y);
                    g.RotateTransform(tempDegress);



                    if (showLabel)
                    {
                        string text = $"{item.LabelName},{(item.Confidence * 100).ToString("f1")}%";

                        float strX = -item.Width / 2;
                        float strY = -item.Height / 2;
                        // 测量文本的尺寸
                        SizeF textSize = g.MeasureString(text, font);
                        strY = strY - textSize.Height;
                        // 创建一个带有透明度的填充刷 (半透明黑色背景)
                        Color backgroundColor = Color.FromArgb(200, Color.LimeGreen); // Alpha 128 表示 50% 透明
                        using (Brush backgroundBrush = new SolidBrush(backgroundColor))
                        {
                            g.FillRectangle(backgroundBrush, strX, strY, textSize.Width, textSize.Height);
                        }
                        // 绘制文本
                        g.DrawString(text, font, solidBrush, strX, strY);
                    }


                    g.DrawRectangle(pen, -item.Width / 2, -item.Height / 2, item.Width, item.Height);
                    g.Restore(s);

                }
                img?.Dispose();
                return newImage;
            }

        }


        // 骨骼连接线定义 (COCO 17关键点连接方式)
        private static readonly (int, int)[] SkeletonConnections = new[]
        {
            // 头部
            (0, 1), (0, 2), (1, 3), (2, 4),
            // 躯干
            (5, 6), (5, 11), (6, 12), (11, 12),
            // 左臂
            (5, 7), (7, 9),
            // 右臂
            (6, 8), (8, 10),
            // 左腿
            (11, 13), (13, 15),
            // 右腿
            (12, 14), (14, 16)
        };

        // 关键点颜色（按部位分组）
        private static readonly Color[] KeypointColors = new[]
        {
            Color.Red,     // 鼻子
            Color.Blue,    // 左眼
            Color.Blue,     // 右眼
            Color.Green,    // 左耳
            Color.Green,    // 右耳
            Color.Magenta, // 左肩
            Color.Magenta,  // 右肩
            Color.Cyan,     // 左肘
            Color.Cyan,     // 右肘
            Color.Yellow,   // 左手腕
            Color.Yellow,   // 右手腕
            Color.Orange,   // 左髋
            Color.Orange,   // 右髋
            Color.Purple,   // 左膝
            Color.Purple,    // 右膝
            Color.Pink,     // 左踝
            Color.Pink       // 右踝
        };

        

        #endregion

        #endregion
    }

}
