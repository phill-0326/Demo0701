using ByteTrackBase;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxByteTracker
{
    public class DetectionResult : IObject
    {
        RectBox _box;
        int _label;
        float _prob;
        string _name;

        public RectBox RectBox => _box;

        public int Label => _label;

        public float Prob => _prob;

        public string Name => _name;

        public DetectionResult(RectBox box, string name, int label, float prob)
        {
            _box = box;
            _label = label;
            _prob = prob;
            _name = name;
        }

        public Track ToTrack()
        {
            return new Track(_box, _prob, ("label", _label), ("name", _name));
        }
    }
}
