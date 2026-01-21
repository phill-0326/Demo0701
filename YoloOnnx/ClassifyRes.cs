using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnx
{
    public class ClassifyRes
    {
        /// <summary>
        /// 大于设置置信度标签名
        /// </summary>
        public string[] AllClassifyResultLabelName { get; set; }

        /// <summary>
        /// 大于设置置信度标签索引
        /// </summary>
        public int[] AllClassifyResultIndex { get; set; }

        /// <summary>
        /// 大于设置置信度置信度
        /// </summary>
        public float[] AllConfidence { get; set; }

        /// <summary>
        /// 最大置信度标签名，不考虑设置的置信度
        /// </summary>
        public string MaxClassifyResultLabelName { get; set; }

        /// <summary>
        /// 最大置信度
        /// </summary>
        public float MaxConfidence { get; set; }

        /// <summary>
        /// 最大置信度的标签索引
        /// </summary>
        public int MaxClassifyResultIndex;
    }
}
