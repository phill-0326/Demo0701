using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnx
{
    public class Boxes
    {
        /// <summary>
        /// 置信度
        /// </summary>
        public float Confidence { get; set; }

        /// <summary>
        /// 原图边界框X中心坐标
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// 原图边界框Y中心坐标
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// 原图边界框宽
        /// </summary>
        public float Width { get; set; }

        /// <summary>
        /// 原图边界框高
        /// </summary>
        public float Height { get; set; }

        /// <summary>
        /// 原图边界框角度(弧度)
        /// </summary>
        public float Angle { get; set; }

        /// <summary>
        /// 分类索引ID
        /// </summary>
        public int LabelId { get; set; }

        public int BoxIndex { get; set; }

        /// <summary>
        /// 所属分类名称
        /// </summary>
        public string LabelName { get; set; }

        /// <summary>
        /// 分割掩码
        /// </summary>
        public byte[,] Mask { get; set; }

        /// <summary>
        /// 姿态检测关键点信息
        /// </summary>
        public KeyPoint[] KeyPoints { get; set; }

    }

    /// <summary>
    /// 姿态检测关键点信息
    /// </summary>
    public class KeyPoint
    {
        // 关键点名称 (COCO 17个关键点)
        public static readonly string[] KeypointNames = new[]
        {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
        };

        /// <summary>
        /// 关键点X坐标
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// 关键点Y坐标
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// 关键点置信度
        /// </summary>
        public float Visibility { get; set; }

        /// <summary>
        /// 关键点名称
        /// </summary>
        public string Name { get; set; }
    }
}
