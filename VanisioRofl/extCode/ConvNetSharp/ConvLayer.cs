using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace VanisioRofl.extCode.ConvNetSharp
{
    [DataContract]
    public class ConvLayer : LayerBase, IDotProductLayer
    {
        public ConvLayer(int width, int height, int filterCount)
        {
            GroupSize = 2;
            L1DecayMul = 0.0;
            L2DecayMul = 1.0;
            Stride = 1;
            Pad = 0;

            FilterCount = filterCount;
            Width = width;
            Height = height;
        }

        [DataMember]
        public Volume Biases { get; private set; }

        [DataMember]
        public List<Volume> Filters { get; private set; }

        [DataMember]
        public int FilterCount { get; private set; }

        [DataMember]
        public double L1DecayMul { get; set; }

        [DataMember]
        public double L2DecayMul { get; set; }

        [DataMember]
        public int Stride { get; set; }

        [DataMember]
        public int Pad { get; set; }

        [DataMember]
        public double BiasPref { get; set; }

        [DataMember]
        public Activation Activation { get; set; }

        [DataMember]
        public int GroupSize { get; private set; }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            // optimized code by @mdda that achieves 2x speedup over previous version

            InputActivation = input;
            var outputActivation = new Volume(OutputWidth, OutputHeight, OutputDepth, 0.0);

            var volumeWidth = input.Width;
            var volumeHeight = input.Height;
            var xyStride = Stride;

#if PARALLEL
            Parallel.For(0, this.OutputDepth, depth =>
#else
            for (var depth = 0; depth < OutputDepth; depth++)
#endif
            {
                var filter = Filters[depth];
                var y = -Pad;

                for (var ay = 0; ay < OutputHeight; y += xyStride, ay++)
                {
                    // xyStride
                    var x = -Pad;
                    for (var ax = 0; ax < OutputWidth; x += xyStride, ax++)
                    {
                        // xyStride

                        // convolve centered at this particular location
                        var a = 0.0;
                        for (var fy = 0; fy < filter.Height; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < filter.Width; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < volumeHeight && ox >= 0 && ox < volumeWidth)
                                {
                                    for (var fd = 0; fd < filter.Depth; fd++)
                                    {
                                        // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                        a += filter.Weights[(filter.Width * fy + fx) * filter.Depth + fd] *
                                             input.Weights[(volumeWidth * oy + ox) * input.Depth + fd];
                                    }
                                }
                            }
                        }

                        a += Biases.Weights[depth];
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
            var volume = InputActivation;
            volume.WeightGradients = new double[volume.Weights.Length]; // zero out gradient wrt bottom data, we're about to fill it

            var volumeWidth = volume.Width;
            var volumeHeight = volume.Height;
            var volumeDepth = volume.Depth;
            var xyStride = Stride;

#if PARALLEL
                var locker = new object();
                Parallel.For(0, this.OutputDepth, () => new Volume(volumeWidth, volumeHeight, volumeDepth, 0), (depth, state, temp) =>
#else
            var temp = volume;
            for (var depth = 0; depth < OutputDepth; depth++)
#endif
            {
                var filter = Filters[depth];
                var y = -Pad;
                for (var ay = 0; ay < OutputHeight; y += xyStride, ay++)
                {
                    // xyStride
                    var x = -Pad;
                    for (var ax = 0; ax < OutputWidth; x += xyStride, ax++)
                    {
                        // xyStride

                        // convolve centered at this particular location
                        var chainGradient = OutputActivation.GetGradient(ax, ay, depth);
                        // gradient from above, from chain rule
                        for (var fy = 0; fy < filter.Height; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < filter.Width; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < volumeHeight && ox >= 0 && ox < volumeWidth)
                                {
                                    for (var fd = 0; fd < filter.Depth; fd++)
                                    {
                                        filter.AddGradient(fx, fy, fd, volume.Get(ox, oy, fd) * chainGradient);
                                        temp.AddGradient(ox, oy, fd, filter.Get(fx, fy, fd) * chainGradient);
                                    }
                                }
                            }
                        }

                        Biases.WeightGradients[depth] += chainGradient;
                    }
                }

#if !PARALLEL
            }
#else
                    return temp;
                }
                    ,
                    result =>
                    {
                        lock (locker)
                        {
                            volume.AddGradientFrom(result);
                        }
                    });
#endif
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            // required
            OutputDepth = FilterCount;

            // computed
            // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
            // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
            // final application.
            OutputWidth = (int)Math.Floor((InputWidth + Pad * 2 - Width) / (double)Stride + 1);
            OutputHeight = (int)Math.Floor((InputHeight + Pad * 2 - Height) / (double)Stride + 1);

            // initializations
            var bias = BiasPref;
            Filters = new List<Volume>();

            for (var i = 0; i < OutputDepth; i++)
            {
                Filters.Add(new Volume(Width, Height, InputDepth));
            }

            Biases = new Volume(1, 1, OutputDepth, bias);
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();
            for (var i = 0; i < OutputDepth; i++)
            {
                response.Add(new ParametersAndGradients
                {
                    Parameters = Filters[i].Weights,
                    Gradients = Filters[i].WeightGradients,
                    L2DecayMul = L2DecayMul,
                    L1DecayMul = L1DecayMul
                });
            }

            response.Add(new ParametersAndGradients
            {
                Parameters = Biases.Weights,
                Gradients = Biases.WeightGradients,
                L1DecayMul = 0.0,
                L2DecayMul = 0.0
            });

            return response;
        }
    }
}