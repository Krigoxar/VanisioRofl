using System;
using System.Runtime.Serialization;

namespace VanisioRofl.extCode.ConvNetSharp
{
    /// <summary>
    ///     implements an L2 regression cost layer,
    ///     so penalizes \sum_i(||x_i - y_i||^2), where x is its input
    ///     and y is the user-provided array of "correct" values.
    /// </summary>
    [DataContract]
    public class RegressionLayer : LayerBase, ILastLayer
    {
        public RegressionLayer(int neuronCount)
        {
            NeuronCount = neuronCount;
        }

        [DataMember]
        public int NeuronCount { get; private set; }

        public double Backward(double y)
        {
            // compute and accumulate gradient wrt weights and bias of this layer
            var x = InputActivation;
            x.WeightGradients = new double[x.Weights.Length]; // zero out the gradient of input Vol
            var loss = 0.0;

            // lets hope that only one number is being regressed
            var dy = x.Weights[0] - y;
            x.WeightGradients[0] = dy;
            loss += 0.5 * dy * dy;

            return loss;
        }

        public double Backward(double[] y)
        {
            // compute and accumulate gradient wrt weights and bias of this layer
            var x = InputActivation;
            x.WeightGradients = new double[x.Weights.Length]; // zero out the gradient of input Vol
            var loss = 0.0;

            for (var i = 0; i < OutputDepth; i++)
            {
                var dy = x.Weights[i] - y[i];
                x.WeightGradients[i] = dy;
                loss += 0.5 * dy * dy;
            }

            return loss;
        }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            InputActivation = input;
            OutputActivation = input;
            return input; // identity function
        }

        public override void Backward()
        {
            throw new NotImplementedException();
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            var inputCount = inputWidth * inputHeight * inputDepth;
            OutputDepth = inputCount;
            OutputWidth = 1;
            OutputHeight = 1;
        }
    }
}