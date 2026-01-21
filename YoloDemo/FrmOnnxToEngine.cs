using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using YoloOnnx;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace YoloDemo
{
    public partial class FrmOnnxToEngine : Form
    {
        public FrmOnnxToEngine()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "onnx (*.onnx)|*.onnx";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                textBox1.Text = openFileDialog.FileName;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            DialogResult res = MessageBox.Show("转换过程比较耗时(几分钟)，需耐心等待，待弹窗提示结束！是否继续?","提示!",MessageBoxButtons.YesNo,MessageBoxIcon.Question);
            if (res == DialogResult.Yes)
            {
                button2.Enabled = false;
                textBox1.Enabled = false;
                button1.Enabled = false;
                this.Text = "OnnxTOEngine(转换中...)";
                Task.Run(() =>
                {
                    Stopwatch time = new Stopwatch();
                    time.Restart();
                    Yolo_TensorRT.Onnx2Engine(textBox1.Text,50);
                    time.Stop();
                    this.Invoke(new Action(() =>
                    {
                        button2.Enabled = true;
                        textBox1.Enabled = true;
                        button1.Enabled = true;
                        this.Text = "OnnxTOEngine(转换完成)";
                        MessageBox.Show($"转换完成,耗时[{time.ElapsedMilliseconds / 1000.0}s],文件保存在Onnx模型同文件夹下！");
                    }));

                   
                });
            }
        }
    }
}
