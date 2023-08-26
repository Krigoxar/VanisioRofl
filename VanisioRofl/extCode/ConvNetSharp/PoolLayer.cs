using System;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace VanisioRofl.extCode.ConvNetSharp
{
    [DataContract]
    public class PoolLayer : LayerBase
    {
        [DataMember]
        private int[] switchx;
        [DataMember]
        private int[] switchy;

        public PoolLayer(int width, int height)
        {
            Width = width;
            Height = height;
            Stride = 2;
            Pad = 0;
        }

        [DataMember]
        public int Pad { get; set; }

        [DataMember]
        public int Stride { get; set; }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            InputActivation = input;

            var outputActivation = new Volume(OutputWidth, OutputHeight, OutputDepth, 0.0);

#if PARALLEL
            Parallel.For(0, this.OutputDepth, depth =>
#else
            for (var depth = 0; depth < OutputDepth; depth++)
#endif
            {
                var n = depth * OutputWidth * OutputHeight; // a counter for switches

                var x = -Pad;
                var y = -Pad;
                for (var ax = 0; ax < OutputWidth; x += Stride, ax++)
                {
                    y = -Pad;
                    for (var ay = 0; ay < OutputHeight; y += Stride, ay++)
                    {
                        // convolve centered at this particular location
                        var a = double.MinValue;
                        int winx = -1, winy = -1;

                        for (var fx = 0; fx < Width; fx++)
                        {
                            for (var fy = 0; fy < Height; fy++)
                            {
                                var oy = y + fy;
                                var ox = x + fx;
                                if (oy >= 0 && oy < input.Height && ox >= 0 && ox < input.Width)
                                {
                                    var v = input.Get(ox, oy, depth);
                                    // perform max pooling and store pointers to where
                                    // the max came from. This will speed up backprop 
                                    // and can help make nice visualizations in future
                                    if (v > a)
                                    {
                                        a = v;
                                        winx = ox;
                                        winy = oy;
                                    }
                                }
                            }
                        }

                        switchx[n] = winx;
                        switchy[n] = winy;
                        n++;
                        outputActivation.Set(ax, ay, depth, a);
                    }
                }
            }
#if PARALLEL
);
#endif

            OutputActivation = outputActivation;
            return OutputActivation;
        }

        public override void Backward()
        {
            // pooling layers have no parameters, so simply compute 
            // gradient wrt data here
            var volume = InputActivation;
            volume.WeightGradients = new double[volume.Weights.Length]; // zero out gradient wrt data

#if PARALLEL
            Parallel.For(0, this.OutputDepth, depth =>
#else
            for (var depth = 0; depth < OutputDepth; depth++)
#endif
            {
                var n = depth * OutputWidth * OutputHeight;

                var x = -Pad;
                var y = -Pad;
                for (var ax = 0; ax < OutputWidth; x += Stride, ax++)
                {
                    y = -Pad;
                    for (var ay = 0; ay < OutputHeight; y += Stride, ay++)
                    {
                        var chainGradient = OutputActivation.GetGradient(ax, ay, depth);
                        volume.AddGradient(switchx[n], switchy[n], depth, chainGradient);
                        n++;
                    }
                }
            }
#if PARALLEL
);
#endif
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            // computed
            OutputDepth = InputDepth;
            OutputWidth = (int)Math.Floor((InputWidth + Pad * 2 - Width) / (double)Stride + 1);
            OutputHeight = (int)Math.Floor((InputHeight + Pad * 2 - Height) / (double)Stride + 1);

            // store switches for x,y coordinates for where the max comes from, for each output neuron
            switchx = new int[OutputWidth * OutputHeight * OutputDepth];
            switchy = new int[OutputWidth * OutputHeight * OutputDepth];
        }
    }
}