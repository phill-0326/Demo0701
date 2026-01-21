using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnx
{
    public class ModelInfo
    {
        public string TaskName { get; set; }

        public string Batch { get; set; }

        public string Imgsz { get; set; }

        public string LabelNames { get; set; }
    }
}
