namespace ByteTrackBase
{

    public interface IObject
    {
        RectBox RectBox { get; }

        int Label { get; }

        float Prob { get; }

        Track ToTrack();
    }
}
