﻿using System;
using System.Runtime.Serialization;

namespace VanisioRofl.extCode.ConvNetSharp
{
    /// <summary>
    ///     Implements Maxout nnonlinearity that computes
    ///     x -> max(x)
    ///     where x is a vector of size group_size. Ideally of course,
    ///     the input size should be exactly divisible by group_size
    /// </summary>
    [DataContract]
    public class MaxoutLayer : LayerBase
    {
        [DataMember]
        private int[] switches;

        public MaxoutLayer()
        {
            GroupSize = 2;
        }

        [DataMember]
        public int GroupSize { get; set; }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            InputActivation = input;
            var depth = OutputDepth;
            var outputActivation = new Volume(OutputWidth, OutputHeight, OutputDepth, 0.0);

            // optimization branch. If we're operating on 1D arrays we dont have
            // to worry about keeping track of x,y,d coordinates inside
            // input volumes. In convnets we do :(
            if (OutputWidth == 1 && OutputHeight == 1)
            {
                for (var i = 0; i < depth; i++)
                {
                    var ix = i * GroupSize; // base index offset
                    var a = input.Weights[ix];
                    var ai = 0;

                    for (var j = 1; j < GroupSize; j++)
                    {
                        var a2 = input.Weights[ix + j];
                        if (a2 > a)
                        {
                            a = a2;
                            ai = j;
                        }
                    }

                    outputActivation.Weights[i] = a;
                    switches[i] = ix + ai;
                }
            }
            else
            {
                var n = 0; // counter for switches
                for (var x = 0; x < input.Width; x++)
                {
                    for (var y = 0; y < input.Height; y++)
                    {
                        for (var i = 0; i < depth; i++)
                        {
                            var ix = i * GroupSize;
                            var a = input.Get(x, y, ix);
                            var ai = 0;

                            for (var j = 1; j < GroupSize; j++)
                            {
                                var a2 = input.Get(x, y, ix + j);
                                if (a2 > a)
                                {
                                    a = a2;
                                    ai = j;
                                }
                            }

                            outputActivation.Set(x, y, i, a);
                            switches[n] = ix + ai;
                            n++;
                        }
                    }
                }
            }

            OutputActivation = outputActivation;
            return OutputActivation;
        }

        public override void Backward()
        {
            var volume = InputActivation; // we need to set dw of this
            var volume2 = OutputActivation;
            var depth = OutputDepth;
            volume.WeightGradients = new double[volume.Weights.Length]; // zero out gradient wrt data

            // pass the gradient through the appropriate switch
            if (OutputWidth == 1 && OutputHeight == 1)
            {
                for (var i = 0; i < depth; i++)
                {
                    var chainGradient = volume2.WeightGradients[i];
                    volume.WeightGradients[switches[i]] = chainGradient;
                }
            }
            else
            {
                // bleh okay, lets do this the hard way
                var n = 0; // counter for switches
                for (var x = 0; x < volume2.Width; x++)
                {
                    for (var y = 0; y < volume2.Height; y++)
                    {
                        for (var i = 0; i < depth; i++)
                        {
                            var chainGradient = volume2.GetGradient(x, y, i);
                            volume.SetGradient(x, y, switches[n], chainGradient);
                            n++;
                        }
                    }
                }
            }
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            OutputDepth = (int)Math.Floor(inputDepth / (double)GroupSize);
            OutputWidth = inputWidth;
            OutputHeight = inputHeight;

            switches = new int[OutputWidth * OutputHeight * OutputDepth]; // useful for backprop
        }
    }
}