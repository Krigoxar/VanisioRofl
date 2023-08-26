using System;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace VanisioRofl.extCode.ConvNetSharp
{
    [DataContract]
    public class TanhLayer : LayerBase
    {
        public override Volume Forward(Volume input, bool isTraining = false)
        {
            InputActivation = input;
            var outputActivation = input.CloneAndZero();
            var length = input.Weights.Length;

#if PARALLEL
            Parallel.For(0, length, i =>
#else
            for (var i = 0; i < length; i++)
#endif
            { outputActivation.Weights[i] = Math.Tanh(input.Weights[i]); }
#if PARALLEL
);
#endif
            OutputActivation = outputActivation;
            return OutputActivation;
        }

        public override void Backward()
        {
            var volume = InputActivation; // we need to set dw of this
            var volume2 = OutputActivation;
            var length = volume.Weights.Length;
            volume.WeightGradients = new double[length]; // zero out gradient wrt data

#if PARALLEL
            Parallel.For(0, length, i =>
#else
            for (var i = 0; i < length; i++)
#endif
            {
                var v2wi = volume2.Weights[i];
                volume.WeightGradients[i] = (1.0 - v2wi * v2wi) * volume2.WeightGradients[i];
            }
#if PARALLEL
);
#endif
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            OutputDepth = inputDepth;
            OutputWidth = inputWidth;
            OutputHeight = inputHeight;
        }
    }
}