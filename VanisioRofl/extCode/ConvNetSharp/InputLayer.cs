using System.Runtime.Serialization;

namespace VanisioRofl.extCode.ConvNetSharp
{
    [DataContract]
    public sealed class InputLayer : LayerBase
    {
        public InputLayer(int inputWidth, int inputHeight, int inputDepth)
        {
            Init(inputWidth, inputHeight, inputDepth);

            OutputWidth = inputWidth;
            OutputHeight = inputHeight;
            OutputDepth = inputDepth;
        }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            InputActivation = input;
            OutputActivation = input;
            return OutputActivation; // simply identity function for now
        }

        public override void Backward()
        {
        }
    }
}