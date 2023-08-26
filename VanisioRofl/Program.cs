// See https://aka.ms/new-console-template for more information
using System.Diagnostics;
using IronOcr;
using VanisioRofl;
using System.Threading;
DQN DQN = new DQN(DQN.InputSize.Height * DQN.InputSize.Width, 24) { Learning = true};
Random random= new Random();

Stopwatch Watch = new Stopwatch();

int MaxTime = 1000;

Console.ReadLine();

IGame game = new Game();
Game.IsOnline = false;

while (true)
{
    Thread.Sleep(1000);
    Console.WriteLine(Game.IsVanisioOpen());
}

while (DQN.Age < 10000)
{
    Watch.Restart();

    int action = DQN.Forward(game.GetInput());
    game.MakeAction(action);

    Watch.Stop();
    Console.WriteLine("Forwards time: " + Watch.ElapsedMilliseconds);

    if (MaxTime - Watch.ElapsedMilliseconds < 0)
    {
        Console.WriteLine("Late by " + (MaxTime - Watch.ElapsedMilliseconds) + " mill");
    }
    else
    {
        Thread.Sleep((int)(MaxTime - Watch.ElapsedMilliseconds));
    }

    Watch.Restart();

    int Reward = Game.ObserveReward();
    DQN.Backward(Reward);

    Watch.Stop();
    Console.WriteLine("Backward time: " + Watch.ElapsedMilliseconds);

    Console.WriteLine("Age: " + DQN.Age + "\n");
}
