﻿using System;
using System.Runtime.Serialization;

namespace VanisioRofl.extCode.ConvNetSharp
{
    /// <summary>
    ///     Implements Sigmoid nnonlinearity elementwise
    ///     x -> 1/(1+e^(-x))
    ///     so the output is between 0 and 1.
    /// </summary>
    [DataContract]
    public class SigmoidLayer : LayerBase
    {
        public override Volume Forward(Volume input, bool isTraining = false)
        {
            InputActivation = input;
            var volume2 = input.CloneAndZero();
            var length = input.Weights.Length;
            double[] v2w = volume2.Weights;
            double[] vw = input.Weights;

            for (var i = 0; i < length; i++)
            {
                v2w[i] = 1.0 / (1.0 + Math.Exp(-vw[i]));
            }

            OutputActivation = volume2;
            return OutputActivation;
        }

        public override void Backward()
        {
            var volume = InputActivation; // we need to set dw of this
            var volume2 = OutputActivation;
            var length = volume.Weights.Length;
            volume.WeightGradients = new double[length]; // zero out gradient wrt data

            for (var i = 0; i < length; i++)
            {
                var v2wi = volume2.Weights[i];
                volume.WeightGradients[i] = v2wi * (1.0 - v2wi) * volume2.WeightGradients[i];
            }
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