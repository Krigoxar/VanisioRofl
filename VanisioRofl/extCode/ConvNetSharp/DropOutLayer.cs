using System;
using System.Runtime.Serialization;

namespace VanisioRofl.extCode.ConvNetSharp
{
    [DataContract]
    public class DropOutLayer : LayerBase
    {
        private static readonly Random Random = new Random(RandomUtilities.Seed);
        [DataMember]
        private bool[] dropped;

        public DropOutLayer(double dropProb = 0.5)
        {
            DropProb = dropProb;
        }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            InputActivation = input;
            var output = input.Clone();
            var length = input.Weights.Length;

            if (isTraining)
            {
                // do dropout
                for (var i = 0; i < length; i++)
                {
                    if (Random.NextDouble() < DropProb.Value)
                    {
                        output.Weights[i] = 0;
                        dropped[i] = true;
                    } // drop!
                    else
                    {
                        dropped[i] = false;
                    }
                }
            }
            else
            {
                // scale the activations during prediction
                for (var i = 0; i < length; i++)
                {
                    output.Weights[i] *= DropProb.Value;
                }
            }

            OutputActivation = output;
            return OutputActivation; // dummy identity function for now
        }

        public override void Backward()
        {
            var volume = InputActivation; // we need to set dw of this
            var chainGradient = OutputActivation;
            var length = volume.Weights.Length;
            volume.WeightGradients = new double[length]; // zero out gradient wrt data

            for (var i = 0; i < length; i++)
            {
                if (!dropped[i])
                {
                    volume.WeightGradients[i] = chainGradient.WeightGradients[i]; // copy over the gradient
                }
            }
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            // computed
            OutputWidth = inputWidth;
            OutputHeight = inputHeight;
            OutputDepth = inputDepth;

            dropped = new bool[OutputWidth * OutputHeight * OutputDepth];
        }
    }
}