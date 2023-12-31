﻿using System;
using System.Runtime.Serialization;

namespace VanisioRofl.extCode.ConvNetSharp
{
    [DataContract]
    public class SvmLayer : LayerBase, ILastLayer, IClassificationLayer
    {
        [DataMember]
        public int ClassCount { get; set; }

        public double Backward(double yd)
        {
            var y = (int)yd;
            // compute and accumulate gradient wrt weights and bias of this layer
            var x = InputActivation;
            x.WeightGradients = new double[x.Weights.Length]; // zero out the gradient of input Vol

            // we're using structured loss here, which means that the score
            // of the ground truth should be higher than the score of any other 
            // class, by a margin
            var yscore = x.Weights[y]; // score of ground truth
            const double margin = 1.0;
            var loss = 0.0;
            for (var i = 0; i < OutputDepth; i++)
            {
                if (y == i)
                {
                    continue;
                }
                var ydiff = -yscore + x.Weights[i] + margin;
                if (ydiff > 0)
                {
                    // violating dimension, apply loss
                    x.WeightGradients[i] += 1;
                    x.WeightGradients[y] -= 1;
                    loss += ydiff;
                }
            }

            return loss;
        }

        public double Backward(double[] y)
        {
            throw new NotImplementedException();
        }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            InputActivation = input;
            OutputActivation = input; // nothing to do, output raw scores
            return input;
        }

        public override void Backward()
        {
            throw new NotImplementedException();
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            // computed
            OutputDepth = inputWidth * inputHeight * inputDepth;
            OutputWidth = 1;
            OutputHeight = 1;
        }
    }
}