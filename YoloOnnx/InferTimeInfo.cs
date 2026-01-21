using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnx
{
    public class InferTimeInfo
    {
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
    }
}
