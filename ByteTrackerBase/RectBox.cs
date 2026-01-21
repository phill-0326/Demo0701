using System;

namespace ByteTrackBase
{

    public struct RectBox
    {
        public float X { get; set; }

        public float Y { get; set; }

        public float Width { get; set; }

        public float Height { get; set; }

        public RectBox(float x, float y, float width, float height)
        {
            X = x;
            Y = y;
            Width = width;
            Height = height;
        }

        public float[] ToXYAH()
        {
            return new float[] { X + Width / 2, Y + Height / 2, Width / Height, Height };
        }

        public float CalcIoU(RectBox other)
        {
            var boxArea = (other.Width + 1) * (other.Height + 1);
            var iw = Math.Min(X + Width, other.X + other.Width) - Math.Max(X, other.X) + 1;
            var iou = 0f;
            if (iw > 0)
            {
                var ih = Math.Min(Y + Height, other.Y + other.Height) - Math.Max(Y, other.Y) + 1;
                if (ih > 0)
                {
                    var ua = (Width + 1) * (Height + 1) + boxArea - iw * ih;
                    iou = iw * ih / ua;
                }
            }
            return iou;
        }

    }
}
