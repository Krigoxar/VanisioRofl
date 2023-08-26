using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Collections;
using ConvNetSharp;
using VanisioRofl.extCode.ConvNetSharp;

namespace VanisioRofl
{
    internal class DQN
    {
        public static readonly System.Drawing.Size InputSize = new System.Drawing.Size(100,100);

        private readonly int _netInputs;
        private readonly int _numActions;
        private readonly int _numStates;
        private List<int> _actionWindow;
        private double _epsilon = 1.0;
        private readonly double _epsilonMin = 0.05;
        private readonly double _epsilonTestTime = 0.05;

        private readonly int _experienceSize = 30000; //should work with 30000
        private int _forwardPasses;

        private double _gamma = 0.7;
        private double[] _lastInput;
        private readonly int _learningStepsBurnin = 3000;

        private readonly int _learningStepsTotal = 200000;

        private readonly Net _net;

        private readonly Random _random = new Random(0);

        private double _startLearnThreshold;
        private List<double[]> _stateWindow;
        private List<double[]> _netWindow;
        private List<double> _rewardWindow;

        private readonly int _temporalWindow = 1;

        private readonly Trainer _trainer;
        private readonly int _windowSize = 2;
        private double _latestReward;
        private Averager _averageRewardWindow = new Averager(1000, 10);
        private List<Experience> _experience;
        private Averager _averageLossWindow = new Averager(1000, 10);

        public DQN(int numStates, int numActions)
        {
            _numStates = numStates;
            _numActions = numActions;

            _startLearnThreshold = Math.Floor(Math.Min(_experienceSize * 0.1, 1000));
            _netInputs = _numStates * _temporalWindow + _numActions * _temporalWindow + _numStates;

            _windowSize = Math.Max(_temporalWindow, _windowSize);

            _stateWindow = new List<double[]>(_windowSize);
            _netWindow = new List<double[]>(_windowSize);
            _actionWindow = new List<int>(_windowSize);
            _rewardWindow = new List<double>(_windowSize);

            for (int i = 0; i < _windowSize; i++)
            {
                _stateWindow.Add(new double[] { });
                _netWindow.Add(new double[] { });
                _actionWindow.Add(0);
                _rewardWindow.Add(0);
            }

            _experience = new List<Experience>(_experienceSize);

            _net = new Net();

            _net.AddLayer(new InputLayer(1, 1, _numStates));
            _net.AddLayer(new ConvLayer(8, 8, 32) { Stride = 4 });
            _net.AddLayer(new ConvLayer(4, 4, 64) { Stride = 2 });
            _net.AddLayer(new ConvLayer(3, 3, 64) { Stride = 1 });
            _net.AddLayer(new FullyConnLayer(128, Activation.Relu));
            _net.AddLayer(new FullyConnLayer(128, Activation.Relu));
            _net.AddLayer(new RegressionLayer(_numActions));


            _trainer = new Trainer(_net);
            _trainer.LearningRate = 0.001;
            _trainer.Momentum = 0;
            _trainer.BatchSize = 64;
            _trainer.L2Decay = 0.01;
        }

        public int Age { get; set; } = 0;

        public bool Learning { get; set; }

        private int randomAction()
        {
            return _random.Next(_numActions);
        }

        private int Policy(double[] s)
        {
            double maxval = 0;
            return Policy(s, out maxval);
        }

        private int Policy(double[] s, out double maxval)
        {
            var svol = new Volume(1, 1, _netInputs);
            svol.Weights = s;

            var actionValues = _net.Forward(svol);

            var maxk = 0;
            maxval = actionValues.Weights[0];

            for (var k = 1; k < _numActions; k++)
            {
                if (actionValues.Weights[k] > maxval)
                {
                    maxk = k;
                    maxval = actionValues.Weights[k];
                }
            }

            return maxk;
        }

        public int Forward(double[] inputArray)
        {
            _forwardPasses += 1;

            _lastInput = inputArray;
            int action = 0;
            var netInput = new double[] { };

            if (_forwardPasses > _temporalWindow)
            {
                //We have enough to actually do something reasonable

                netInput = GetNetInput(inputArray);

                if (Learning)
                {
                    _epsilon = Math.Min(1.0,
                        Math.Max(_epsilonMin,
                            1.0 - (Age - _learningStepsBurnin) / (_learningStepsTotal - _learningStepsBurnin)));
                }
                else
                {
                    _epsilon = _epsilonTestTime;
                }

                var rf = _random.NextDouble();


                if (rf < _epsilon)
                {
                    action = randomAction();
                }
                else
                {
                    action = Policy(netInput);
                }
            }
            else
            {

                action = randomAction();
            }

            _netWindow.RemoveAt(0);
            _netWindow.Add(netInput);
            _stateWindow.RemoveAt(0);
            _stateWindow.Add(inputArray);
            _actionWindow.RemoveAt(0);
            _actionWindow.Add(action);

            return action;
        }

        public void Backward(double reward)
        {
            _latestReward = reward;
            _averageRewardWindow.Add(reward);
            _rewardWindow.RemoveAt(0);
            _rewardWindow.Add(reward);

            if (!Learning)
                return;

            Age += 1;

            if (_forwardPasses > _temporalWindow + 1)
            {
                var e = new Experience();

                e.State0 = _netWindow[_windowSize - 2];
                e.Action0 = _actionWindow[_windowSize - 2];
                e.Reward0 = _rewardWindow[_windowSize - 2];
                e.State1 = _netWindow[_windowSize - 1];

                if (_experience.Count < _experienceSize)
                {
                    _experience.Add(e);
                }
                else
                {
                    _experience[_random.Next(_experienceSize)] = e;
                }
            }

            if (_experience.Count > _startLearnThreshold)
            {
                double avcost = 0.0;

                for (int k = 0; k < _trainer.BatchSize; k++)
                {
                    int re = _random.Next(_experience.Count);
                    var e = _experience[re];
                    var x = new Volume(1, 1, _netInputs);
                    x.Weights = e.State0;
                    double maxAct = 0;
                    Policy(e.State1, out maxAct);
                    var r = e.Reward0 + _gamma * maxAct;
                    var y = new double[_numActions];
                    y[e.Action0] = r;
                    _trainer.Train(x, y);
                    avcost += _trainer.Loss;
                }

                avcost = avcost / _trainer.BatchSize;
                _averageLossWindow.Add(avcost);
            }
        }

        private double[] GetNetInput(double[] xt)
        {
            var w = new List<double>(xt);

            var n = _windowSize;

            for (var k = 0; k < _temporalWindow; k++)
            {
                w.AddRange(_stateWindow[n - 1 - k]);

                var action1ofk = new double[_numActions];

                for (var q = 0; q < _numActions; q++)
                    action1ofk[q] = 0.0;

                action1ofk[_actionWindow[_windowSize - 1 - k]] = 1.0 * _numStates;

                w.AddRange(action1ofk);
            }

            return w.ToArray();
        }
    }
    public class Averager
    {
        private readonly int _maxSize;
        private readonly int _minSize;

        List<double> _numbers = new List<double>();
        private double _sum = 0;

        public Averager(int maxSize, int minSize)
        {
            _maxSize = maxSize;
            _minSize = minSize;
        }

        public void Add(double number)
        {
            _numbers.Add(number);
            _sum += number;

            if (_numbers.Count > _maxSize)
            {
                _sum -= _numbers[0];
                _numbers.RemoveAt(0);
            }
        }

        public double GetAverage()
        {
            if (_numbers.Count < _minSize)
                return -1;
            else
                return _sum / _numbers.Count;
        }

        public void Reset()
        {
            _numbers.Clear();
            _sum = 0;
        }
    }
    public class Experience
    {
        public double[] State0 { get; set; }
        public int Action0 { get; set; }
        public double Reward0 { get; set; }
        public double[] State1 { get; set; }
    }
}
