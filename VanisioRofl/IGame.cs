using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static VanisioRofl.Game;

namespace VanisioRofl
{
    internal interface IGame
    {
        public double[] GetInput();
        public void MakeAction(int actions);
        abstract public static int ObserveReward();
        abstract public static bool IsOnline { get; set; }
    }
}
