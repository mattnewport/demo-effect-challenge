using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace data_gen
{
    public partial class MainForm : Form
    {
        private float r;

        public float Alpha { get; set; }
        public float HeatAlpha { get; set; }
        public float Gravity { get; set; }
        public float Buoyancy { get; set; }
        public float Expansion { get; set; }
        public float AmbientTemp { get; set; }
        public float BurnRate
        {
            get { return 1.0f - (float)Math.Exp(-r); }
            set { r = -(float)Math.Log(1.0f - value); } 
        }
        public float Mixture { get; set; }
        public float ThresholdTemp { get; set; }
        public float CombustionTempIncrease { get; set; }

        private float BlackBodyIntensity(float lambda, float t)
        {
            var h = 6.626e-34; // Planck's constant
            var c = 299792458.0; // speed of light
            var k = 1.380649e-23; // Boltzmann constant
            var num = 2.0 * 3.14159 * h * c * c;
            var denom = Math.Pow(lambda, 5.0) * (Math.Exp((h * c) / (lambda * k * t)) - 1.0);
            return (float)(num / denom);
        }

        private int Clamp(int x, int a, int b)
        {
            return Math.Min(Math.Max(x, a), b);
        }

        private Tuple<float, float, float> BlackBodyColor(float t)
        {
            var r = BlackBodyIntensity(700e-9f, t);
            var g = BlackBodyIntensity(546e-9f, t);
            var b = BlackBodyIntensity(436e-9f, t);
            // return Color.FromArgb(255, Clamp((int)(r * 255.0f), 0, 255), Clamp((int)(g * 255.0f), 0, 255), Clamp((int)(b * 255.0f), 0, 255));
            return new Tuple<float, float, float>(r, g, b);
        }

        private float ExponentialMapping(float x, float lAvg)
        {
            return 1.0f - (float)Math.Exp(-x / lAvg);
        }

        private void UpdateBlackBodyImage()
        {
            const int width = 512;
            Func<int, float> tempFunc = (x => 1500.0f + 2000.0f * (float)x / (float)width);
            var rgbs = Enumerable.Range(0, 512).Select(x => BlackBodyColor(tempFunc(x)));
            Func<Tuple<float, float, float>, float> luminance = x => 0.2126f * x.Item1 + 0.7152f * x.Item2 + 0.0722f * x.Item3;
            var min = rgbs.Aggregate(1e30f, (x, rgb) => Math.Min(luminance(rgb), x));
            var max = rgbs.Aggregate(-1e30f, (x, rgb) => Math.Max(luminance(rgb), x));
            var sum = rgbs.Sum(rgb => luminance(rgb));
            var avg = sum / width;
            Func<float, float> normalize = x => ExponentialMapping(x, avg);
            List<Tuple<float, float, float>> normalizedRgbs = rgbs.Select(x => new Tuple<float, float, float>(normalize(x.Item1), normalize(x.Item2), normalize(x.Item3))).ToList();
            Func<float, int> toByte = x => Clamp((int)(x * 255.0f), 0, 255);

            var bmp = new Bitmap(width, 32);
            for (var y = 0; y < bmp.Height; ++y)
            {
                for (var x = 0; x < bmp.Width; ++x)
                {
                    Color col = Color.FromArgb(255, toByte(normalizedRgbs[x].Item1), toByte(normalizedRgbs[x].Item2), toByte(normalizedRgbs[x].Item3));
                    bmp.SetPixel(x, y, col);
                }
            }
            blackBodyPictureBox.Image = bmp;
        }

        public MainForm()
        {
            InitializeComponent();

            Alpha = 0.05f;
            HeatAlpha = 0.05f;
            Gravity = 9.8f * 0.001f;
            Buoyancy = -0.000033f;
            Expansion = 0.00005f;
            AmbientTemp = 0.0f;
            BurnRate = 0.3f;
            Mixture = 1.0f;
            ThresholdTemp = 300.0f;
            CombustionTempIncrease = 50000.0f;

            UpdateBlackBodyImage();

            burnRateTrackBar.Value = (int)(BurnRate * 100.0f);
            burnRateUpDown.Value = (decimal)BurnRate;
        }

        private void burnRateTrackbar_Scroll(object sender, EventArgs e)
        {
            var tb = sender as TrackBar;
            BurnRate = (float)tb.Value / 100.0f;
            burnRateUpDown.Value = (decimal)BurnRate;
        }

        private void burnRateUpDown_ValueChanged(object sender, EventArgs e)
        {
            var upDown = sender as NumericUpDown;
            BurnRate = (float)upDown.Value;
            burnRateTrackBar.Value = (int)(BurnRate * 100.0f);
        }

        private void openImageButton_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
            pictureBox1.Load(openFileDialog1.FileName);
        }

        private void saveButton_Click(object sender, EventArgs e)
        {
            using (var outWriter = new BinaryWriter(File.OpenWrite("data.bin")))
            {
                // Params
                outWriter.Write(Alpha);
                outWriter.Write(HeatAlpha);
                outWriter.Write(Gravity);
                outWriter.Write(Buoyancy);
                outWriter.Write(Expansion);
                outWriter.Write(AmbientTemp);
                outWriter.Write(BurnRate);
                outWriter.Write(Mixture);
                outWriter.Write(ThresholdTemp);
                outWriter.Write(CombustionTempIncrease);

                // Fuel map
                var bitmap = new Bitmap(pictureBox1.Image);
                for (var y = 0; y < bitmap.Height; ++y)
                {
                    for (var x = 0; x < bitmap.Width; ++x)
                    {
                        var pixelColor = bitmap.GetPixel(x, y);
                        var brightness = pixelColor.GetBrightness();
                        outWriter.Write(brightness);
                    }
                }

                // Color ramp
                var blackBodyBitmap = new Bitmap(blackBodyPictureBox.Image);
                for (var x = 0; x < blackBodyBitmap.Width; ++x)
                {
                    outWriter.Write(blackBodyBitmap.GetPixel(x, 0).R);
                    outWriter.Write(blackBodyBitmap.GetPixel(x, 0).G);
                    outWriter.Write(blackBodyBitmap.GetPixel(x, 0).B);
                    outWriter.Write(blackBodyBitmap.GetPixel(x, 0).A);
                }
            }
        }

        private void mixtureTrackBar_Scroll(object sender, EventArgs e)
        {
            var tb = sender as TrackBar;
            Mixture = (float)tb.Value / 10.0f;
            mixtureUpDown.Value = (decimal)Mixture;
        }

        private void mixtureUpDown_ValueChanged(object sender, EventArgs e)
        {
            var upDown = sender as NumericUpDown;
            Mixture = (float)upDown.Value;
            mixtureTrackBar.Value = (int)(Mixture * 10.0f);
        }
    }
}
