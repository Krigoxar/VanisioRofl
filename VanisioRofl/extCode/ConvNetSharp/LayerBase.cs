﻿using System.Collections.Generic;
using System.Runtime.Serialization;

namespace VanisioRofl.extCode.ConvNetSharp
{
    public class ParametersAndGradients
    {
        public double[] Parameters { get; set; }

        public double[] Gradients { get; set; }

        public double? L2DecayMul { get; set; }

        public double? L1DecayMul { get; set; }
    }

    [KnownType(typeof(ConvLayer))]
    [KnownType(typeof(DropOutLayer))]
    [KnownType(typeof(FullyConnLayer))]
    [KnownType(typeof(InputLayer))]
    [KnownType(typeof(MaxoutLayer))]
    [KnownType(typeof(PoolLayer))]
    [KnownType(typeof(RegressionLayer))]
    [KnownType(typeof(ReluLayer))]
    [KnownType(typeof(SigmoidLayer))]
    [KnownType(typeof(SoftmaxLayer))]
    [KnownType(typeof(SvmLayer))]
    [KnownType(typeof(TanhLayer))]
    [DataContract]
    public abstract class LayerBase
    {
        public Volume InputActivation { get; protected set; }

        public Volume OutputActivation { get; protected set; }

        [DataMember]
        public int OutputDepth { get; protected set; }

        [DataMember]
        public int OutputWidth { get; protected set; }

        [DataMember]
        public int OutputHeight { get; protected set; }

        [DataMember]
        protected int InputDepth { get; private set; }

        [DataMember]
        protected int InputWidth { get; private set; }

        [DataMember]
        protected int InputHeight { get; private set; }

        [DataMember]
        protected int Width { get; set; }

        [DataMember]
        protected int Height { get; set; }

        [DataMember]
        public double? DropProb { get; protected set; }

        public abstract Volume Forward(Volume input, bool isTraining = false);

        public abstract void Backward();

        public virtual void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            InputWidth = inputWidth;
            InputHeight = inputHeight;
            InputDepth = inputDepth;
        }

        public virtual List<ParametersAndGradients> GetParametersAndGradients()
        {
            return new List<ParametersAndGradients>();
        }
    }
}