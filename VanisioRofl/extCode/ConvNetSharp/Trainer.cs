using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace VanisioRofl.extCode.ConvNetSharp
{
    public class Trainer
    {
        public enum Method
        {
            Sgd, // Stochastic gradient descent
            Adam,
            Adagrad,
            Adadelta,
            Windowgrad,
            Netsterov
        }

        private readonly List<double[]> gsum = new List<double[]>(); // last iteration gradients (used for momentum calculations)
        private readonly Net net;
        private readonly List<double[]> xsum = new List<double[]>(); // used in adam or adadelta
        private int k; // iteration counter

        public Trainer(Net net)
        {
            this.net = net;

            LearningRate = 0.01;
            BatchSize = 1;
            TrainingMethod = Method.Sgd;
            Momentum = 0.9;
            Ro = 0.95;
            Eps = 1e-6;
            Beta1 = 0.9;
            Beta2 = 0.999;
        }

        public double L2DecayLoss { get; private set; }

        public double L1DecayLoss { get; private set; }

        public TimeSpan BackwardTime { get; private set; }

        public double CostLoss { get; private set; }

        public TimeSpan ForwardTime { get; private set; }

        public double LearningRate { get; set; }

        public double Ro { get; set; }  // used in adadelta

        public double Eps { get; set; } // used in adam or adadelta

        public double Beta1 { get; set; } // used in adam

        public double Beta2 { get; set; } // used in adam

        public double Momentum { get; set; }

        public double L1Decay { get; set; }

        public double L2Decay { get; set; }

        public int BatchSize { get; set; }

        public Method TrainingMethod { get; set; }

        public double Loss
        {
            get { return CostLoss + L1DecayLoss + L2DecayLoss; }
        }

        public void Train(Volume x, double y)
        {
            Forward(x);

            Backward(y);

            TrainImplem();
        }

        public void Train(Volume x, double[] y)
        {
            Forward(x);

            Backward(y);

            TrainImplem();
        }

        private void TrainImplem()
        {
            k++;
            if (k % BatchSize == 0)
            {
                List<ParametersAndGradients> parametersAndGradients = net.GetParametersAndGradients();

                // initialize lists for accumulators. Will only be done once on first iteration
                if (gsum.Count == 0 && (TrainingMethod != Method.Sgd || Momentum > 0.0))
                {
                    // only vanilla sgd doesnt need either lists
                    // momentum needs gsum
                    // adagrad needs gsum
                    // adam and adadelta needs gsum and xsum
                    for (var i = 0; i < parametersAndGradients.Count; i++)
                    {
                        gsum.Add(new double[parametersAndGradients[i].Parameters.Length]);
                        if (TrainingMethod == Method.Adam || TrainingMethod == Method.Adadelta)
                        {
                            xsum.Add(new double[parametersAndGradients[i].Parameters.Length]);
                        }
                    }
                }

                // perform an update for all sets of weights
                for (var i = 0; i < parametersAndGradients.Count; i++)
                {
                    var parametersAndGradient = parametersAndGradients[i];
                    // param, gradient, other options in future (custom learning rate etc)
                    double[] parameters = parametersAndGradient.Parameters;
                    double[] gradients = parametersAndGradient.Gradients;

                    // learning rate for some parameters.
                    var l2DecayMul = parametersAndGradient.L2DecayMul ?? 1.0;
                    var l1DecayMul = parametersAndGradient.L1DecayMul ?? 1.0;
                    var l2Decay = L2Decay * l2DecayMul;
                    var l1Decay = L1Decay * l1DecayMul;

                    var plen = parameters.Length;
                    for (var j = 0; j < plen; j++)
                    {
                        L2DecayLoss += l2Decay * parameters[j] * parameters[j] / 2; // accumulate weight decay loss
                        L1DecayLoss += l1Decay * Math.Abs(parameters[j]);
                        var l1Grad = l1Decay * (parameters[j] > 0 ? 1 : -1);
                        var l2Grad = l2Decay * parameters[j];

                        var gij = (l2Grad + l1Grad + gradients[j]) / BatchSize; // raw batch gradient

                        double[] gsumi = null;
                        if (gsum.Count > 0)
                        {
                            gsumi = gsum[i];
                        }

                        double[] xsumi = null;
                        if (xsum.Count > 0)
                        {
                            xsumi = xsum[i];
                        }

                        switch (TrainingMethod)
                        {
                            case Method.Sgd:
                                {
                                    if (Momentum > 0.0)
                                    {
                                        // momentum update
                                        var dx = Momentum * gsumi[j] - LearningRate * gij; // step
                                        gsumi[j] = dx; // back this up for next iteration of momentum
                                        parameters[j] += dx; // apply corrected gradient
                                    }
                                    else
                                    {
                                        // vanilla sgd
                                        parameters[j] += -LearningRate * gij;
                                    }
                                }
                                break;
                            case Method.Adam:
                                {
                                    // adam update
                                    gsumi[j] = gsumi[j] * Beta1 + (1 - Beta1) * gij; // update biased first moment estimate
                                    xsumi[j] = xsumi[j] * Beta2 + (1 - Beta2) * gij * gij; // update biased second moment estimate
                                    var biasCorr1 = gsumi[j] * (1 - Math.Pow(Beta1, k)); // correct bias first moment estimate
                                    var biasCorr2 = xsumi[j] * (1 - Math.Pow(Beta2, k)); // correct bias second moment estimate
                                    var dx = -LearningRate * biasCorr1 / (Math.Sqrt(biasCorr2) + Eps);
                                    parameters[j] += dx;
                                }
                                break;
                            case Method.Adagrad:
                                {
                                    // adagrad update
                                    gsumi[j] = gsumi[j] + gij * gij;
                                    var dx = -LearningRate / Math.Sqrt(gsumi[j] + Eps) * gij;
                                    parameters[j] += dx;
                                }
                                break;
                            case Method.Adadelta:
                                {
                                    // assume adadelta if not sgd or adagrad
                                    gsumi[j] = Ro * gsumi[j] + (1 - Ro) * gij * gij;
                                    var dx = -Math.Sqrt((xsumi[j] + Eps) / (gsumi[j] + Eps)) * gij;
                                    xsumi[j] = Ro * xsumi[j] + (1 - Ro) * dx * dx; // yes, xsum lags behind gsum by 1.
                                    parameters[j] += dx;
                                }
                                break;
                            case Method.Windowgrad:
                                {
                                    // this is adagrad but with a moving window weighted average
                                    // so the gradient is not accumulated over the entire history of the run. 
                                    // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                                    gsumi[j] = Ro * gsumi[j] + (1 - Ro) * gij * gij;
                                    var dx = -LearningRate / Math.Sqrt(gsumi[j] + Eps) * gij;
                                    // eps added for better conditioning
                                    parameters[j] += dx;
                                }
                                break;
                            case Method.Netsterov:
                                {
                                    var dx = gsumi[j];
                                    gsumi[j] = gsumi[j] * Momentum + LearningRate * gij;
                                    dx = Momentum * dx - (1.0 + Momentum) * gsumi[j];
                                    parameters[j] += dx;
                                }
                                break;
                            default:
                                throw new ArgumentOutOfRangeException();
                        }

                        gradients[j] = 0.0; // zero out gradient so that we can begin accumulating anew
                    }
                }
            }

            // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
            // in future, TODO: have to completely redo the way loss is done around the network as currently 
            // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
            // and it should all be computed correctly and automatically. 
        }

        private void Backward(double y)
        {
            var chrono = Stopwatch.StartNew();
            CostLoss = net.Backward(y);
            L2DecayLoss = 0.0;
            L1DecayLoss = 0.0;
            BackwardTime = chrono.Elapsed;
        }

        private void Backward(double[] y)
        {
            var chrono = Stopwatch.StartNew();
            CostLoss = net.Backward(y);
            L2DecayLoss = 0.0;
            L1DecayLoss = 0.0;
            BackwardTime = chrono.Elapsed;
        }

        private void Forward(Volume x)
        {
            var chrono = Stopwatch.StartNew();
            net.Forward(x, true); // also set the flag that lets the net know we're just training
            ForwardTime = chrono.Elapsed;
        }
    }
}