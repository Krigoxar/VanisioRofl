using System.Drawing;
using System.Drawing.Imaging;
using WindowsInput.Native;
using WindowsInput;
using static VanisioRofl.IGame;
using IronOcr;
using IronSoftware.Drawing;
using IronOcr.Events;
using static VanisioRofl.Game;
using System.Collections.Generic;
using SixLabors.ImageSharp;

namespace VanisioRofl
{
    internal class Game : IGame
    {
        public static bool IsOnline { get; set; }
        public enum ActionsEnum
        {
            Up = 0x0000001b,
            Down = 0x0000010b,
            Left = 0x0000100b,
            Right = 0x0001000b,

            UpRight = 0x0001001b,
            UpLeft = 0x0000101b,
            DownRight = 0x0001010b,
            DownLeft = 0x0000110b,

            Split = 0x0010000b,

            UpSplit = 0x0010001b,
            DownSplit = 0x0010010b,
            LeftSplit = 0x0010100b,
            RightSplit = 0x0011000b,

            UpRightSplit = 0x0011001b,
            UpLeftSplit = 0x0010101b,
            DownRightSplit = 0x0011010b,
            DownLeftSplit = 0x0010110b,

            Feed = 0x0100000b,

            UpFeed = 0x0100001b,
            DownFeed = 0x0100010b,
            LeftFeed = 0x0100100b,
            RightFeed = 0x0101000b,

            UpRightFeed = 0x0101001b,
            UpLeftFeed = 0x0100101b,
            DownRightFeed = 0x0101010b,
            DownLeftFeed = 0x0100110b,
        }

        public Dictionary<int, ActionsEnum> IntToAct = new()
        {

            [0] = ActionsEnum.Up,
            [1] = ActionsEnum.Down,
            [2] = ActionsEnum.Left,
            [3] = ActionsEnum.Right,
            [4] = ActionsEnum.UpRight,
            [5] = ActionsEnum.UpLeft,
            [6] = ActionsEnum.DownRight,
            [7] = ActionsEnum.DownLeft,
            [8] = ActionsEnum.UpSplit,
            [9] = ActionsEnum.DownSplit,
            [10] = ActionsEnum.LeftSplit,
            [11] = ActionsEnum.RightSplit,
            [12] = ActionsEnum.UpRightSplit,
            [13] = ActionsEnum.UpLeftSplit,
            [14] = ActionsEnum.DownRightSplit,
            [15] = ActionsEnum.DownLeftSplit,
            [16] = ActionsEnum.UpFeed,
            [17] = ActionsEnum.DownFeed,
            [18] = ActionsEnum.LeftFeed,
            [19] = ActionsEnum.RightFeed,
            [20] = ActionsEnum.UpRightFeed,
            [21] = ActionsEnum.UpLeftFeed,
            [22] = ActionsEnum.DownRightFeed,
            [23] = ActionsEnum.DownLeftFeed,
        };

        static InputSimulator InputSimulator = new();

        static public System.Drawing.Size screenSize = new(1920, 1080);
        static Random Random = new();

        Thread CorectionThread = new(() =>
        {
            while (true)
            {
                if (IsOnline)
                {
                    while (ObserveReward() == 0)
                    {
                        InputSimulator.Mouse.MoveMouseTo(960 * 65535 / screenSize.Width, 850 * 65535 / screenSize.Height);
                        InputSimulator.Mouse.LeftButtonClick();
                        Thread.Sleep(100);
                    }
                    //CorrectZoom();
                }
                Thread.Sleep(1000);
            }
        });

        public Game() 
        {
            IsOnline = true;

            ocr.Language = OcrLanguage.EnglishFast;
            ocr.MultiThreaded = true;
            ocr.Configuration.TesseractVariables["user_defined_dpi"] = 70;
            ocr.Configuration.TesseractVariables["debug_file"] = "NUL";

            CorectionThread.Start();
        }

        private int GetAverageColor(BitmapData imageData, byte[] rgbValues, int startX, int startY, int sizeX, int sizeY)
        {
            int sum = 0;

            // Вычисление суммы значений каналов цвета в заданном квадрате
            for (int y = startY; y < startY + sizeY; y++)
            {
                for (int x = startX; x < startX + sizeX; x++)
                {

                    sum += rgbValues[(y * imageData.Stride) + (x * 4)];
                    sum += rgbValues[(y * imageData.Stride) + (x * 4) + 1];
                    sum += rgbValues[(y * imageData.Stride) + (x * 4) + 2];
                }
            }

            return ((sum / (sizeX * sizeY))/3);
        }
        private static Bitmap TakeScreanShot(System.Drawing.Point point1, System.Drawing.Size blockRegionSize)
        {
            Bitmap bitMap = new Bitmap(blockRegionSize.Width, blockRegionSize.Height);
            using (Graphics g = Graphics.FromImage(bitMap))
            {
                g.CopyFromScreen(point1, System.Drawing.Point.Empty, screenSize);
            }
            return bitMap;
        }
        public double[] GetInput()
        {

            Bitmap bitMap = TakeScreanShot(System.Drawing.Point.Empty, screenSize);
            BitmapData imageData = bitMap.LockBits(new System.Drawing.Rectangle(0, 0, bitMap.Width, bitMap.Height), ImageLockMode.ReadOnly, bitMap.PixelFormat);

            int bytes = Math.Abs(imageData.Stride) * bitMap.Height;
            byte[] rgbValues = new byte[bytes];

            System.Runtime.InteropServices.Marshal.Copy(imageData.Scan0, rgbValues, 0, bytes);

            int pixelSizeX = bitMap.Width / DQN.InputSize.Width;
            int pixelSizeY = bitMap.Height / DQN.InputSize.Height;

            double[] chunkedData = new double[DQN.InputSize.Width * DQN.InputSize.Height];

            Parallel.For(0, DQN.InputSize.Height - 1, (int y) =>
            {
                for (int x = 0; x < DQN.InputSize.Width; x++)
                {
                    double pixelColor = GetAverageColor(imageData, rgbValues, x * pixelSizeX, y * pixelSizeY, pixelSizeX, pixelSizeY)/25f;

                    chunkedData[x + (y * DQN.InputSize.Height)] = pixelColor;
                }
            });

            bitMap.UnlockBits(imageData);

            return chunkedData;
        }

        static IronTesseract ocr = new IronTesseract();

        public static int ObserveReward()
        {
            Bitmap bitmap = TakeScreanShot(System.Drawing.Point.Empty, screenSize);
            String Res = ocr.Read(bitmap, new CropRectangle(x: 50, y: 190, height: 35, width: 150)).Text;
            try
            {
                Console.WriteLine(Res.Split("Mass: ")[1].Split(new char[2] {'k', 'K'})[0]);
                return (int)(10 * Convert.ToDouble(Res.Split("Mass: ")[1].Split("k")[0]));
            }
            catch (Exception)
            {
                return 0;
            }
        }
        public static bool IsVanisioOpen()
        {
            Bitmap bitmap = TakeScreanShot(System.Drawing.Point.Empty, screenSize);
            String Res = ocr.Read(bitmap, new CropRectangle(x: 150, y: 70, height: 35, width: 150)).Text;
            return Res == "vanis.io";
        }
        private static void SneakyMouseMove(System.Drawing.Point point)
        {
            InputSimulator.Mouse.MoveMouseTo((point.X + Random.NextInt64(200) - 100) * 65535 / screenSize.Width, (point.Y + Random.NextInt64(200) - 100) * 65535 / screenSize.Height);
        }
        public void MakeAction(int IntActions)
        {
            ActionsEnum actions = IntToAct[IntActions];

            if (!IsOnline)
            {
                Console.WriteLine(actions);
                return;
            }

            if (actions.HasFlag(ActionsEnum.Feed))
            {
                InputSimulator.Keyboard.KeyPress(VirtualKeyCode.VK_W);
            }
            if (actions.HasFlag(ActionsEnum.Split))
            {
                InputSimulator.Keyboard.KeyPress(VirtualKeyCode.SPACE);
            }

            int X = screenSize.Width / 2;
            int Y = screenSize.Height / 2;
            if (actions.HasFlag(ActionsEnum.Up))
            {
                Y = 0;
            }
            else if (actions.HasFlag(ActionsEnum.Down))
            {
                Y = screenSize.Height;
            }

            if (actions.HasFlag(ActionsEnum.Left))
            {
                X = 0;
            }
            else if (actions.HasFlag(ActionsEnum.Right))
            {
                X = screenSize.Width;
            }
            SneakyMouseMove(new System.Drawing.Point(X,Y));
        }
    }
}
