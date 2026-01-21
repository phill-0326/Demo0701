using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using YoloOnnx;
using YoloOnnxByteTracker;
using static System.Net.Mime.MediaTypeNames;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace YoloDemo
{
    public partial class Form1 : Form
    {
        Bitmap img = null;
        Yolo_OnnxRuntime yolo_Onnx = new Yolo_OnnxRuntime();
        Yolo_OpenVINO yolo_OpenVINO = new Yolo_OpenVINO();
        Yolo_TensorRT yolo_TensorRT = new Yolo_TensorRT();

        Yolo_OnnxRuntime yolo_Onnx_Dyamics = new Yolo_OnnxRuntime();
        Yolo_OpenVINO yolo_OpenVINO_Dyamics = new Yolo_OpenVINO();

        OnnxRuntimeTracker onnxRuntimeTracker = new OnnxRuntimeTracker();
        OpenVinoTracker openVinoTracker = new OpenVinoTracker();
        TensorRTTracker tensorRTTracker = new TensorRTTracker();



        public Form1()
        {
            InitializeComponent();
            onnxRuntimeTracker.GetFrameEvent += OnnxRuntimeTracker_GetFrameEvent;
            openVinoTracker.GetFrameEvent += OpenVinoTracker_GetFrameEvent;
            tensorRTTracker.GetFrameEvent += TensorRTTracker_GetFrameEvent;
            onnxRuntimeTracker.StartEvent += OnnxRuntimeTracker_StartEvent;
            onnxRuntimeTracker.StopEvent += OnnxRuntimeTracker_StopEvent;
            openVinoTracker.StartEvent += OpenVinoTracker_StartEvent;
            openVinoTracker.StopEvent += OpenVinoTracker_StopEvent;
            tensorRTTracker.StartEvent += TensorRTTracker_StartEvent;
            tensorRTTracker.StopEvent += TensorRTTracker_StopEvent;
        }

       

        private void Form1_Load(object sender, EventArgs e)
        {
            comboBox1.SelectedIndex = 0;
            comboBox3.SelectedIndex = 0;
            comboBox4.SelectedIndex = 0;
            comboBox2.SelectedIndex = 0;
            comboBox5.SelectedIndex = 0;
            comboBox6.SelectedIndex = 0;
            comboBox7.SelectedIndex = 0;
            comboBox8.SelectedIndex = 0;

        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            yolo_Onnx?.Dispose();
            yolo_OpenVINO?.Dispose();
            yolo_TensorRT?.Dispose();
            yolo_Onnx_Dyamics?.Dispose();
            yolo_OpenVINO_Dyamics?.Dispose();
        }

        #region 固定尺寸输入模型

        #region 加载图片
        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "图片|*.jpg;*.jpeg;*.bmp;*.png;";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                textBox1.Text = openFileDialog.FileName;
                img = new Bitmap(textBox1.Text);
                pictureBox1.Image = img;
            }
        }
        #endregion

        #region 标签颜色选择
        private void button4_Click(object sender, EventArgs e)
        {
            ColorDialog colorDialog = new ColorDialog();

            DialogResult res = colorDialog.ShowDialog();
            if (res == DialogResult.OK)
            {
                button4.ForeColor = colorDialog.Color;
                button4.Text = colorDialog.Color.Name.ToString();
            }
        }
        #endregion

        #region Onnx模型加载
        private void button7_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "onnx (*.onnx)|*.onnx";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // 设置要加载的模型的路径，跟据需要改为你的模型名称
                string modelPath = openFileDialog.FileName;
                ModelInfo modelInfo = new ModelInfo();

                if (comboBox3.SelectedIndex == 0)
                {
                    modelInfo = yolo_Onnx.LoadOnnxModel(modelPath, false);
                }
                else
                {
                    modelInfo = yolo_Onnx.LoadOnnxModel(modelPath, true);
                }


                label15.Text = $"task：{modelInfo.TaskName}";
                label14.Text = $"batch：{modelInfo.Batch}";
                label13.Text = $"imgsz：{modelInfo.Imgsz}";
                label12.Text = $"names：{modelInfo.LabelNames}";
                MessageBox.Show("Onnx模型加载成功!");
            }
        }
        #endregion

        #region Onnx模型推理
        bool onnxStartFlag = false;
        private void button2_Click(object sender, EventArgs e)
        {
            if (onnxStartFlag)
            {
                return;
            }
            onnxStartFlag = true;
            Stopwatch time = new Stopwatch();
            if (img != null)
            {
                try
                {
                    img = new Bitmap(textBox1.Text);
                    float confidence = Convert.ToSingle(numericUpDown1.Value);
                    float iou = Convert.ToSingle(numericUpDown2.Value);
                    InferenceResult res = null;

                    res = yolo_Onnx.Inference(img, confidence, iou);

                    img?.Dispose();

                    label9.Text = $"预处理耗时:{res.PreprocessTime}ms";
                    label6.Text = $"推理耗时:{res.InferenceTime}ms";
                    label10.Text = $"后处理耗时:{res.PostprocessTime}ms";

                    time.Restart();
                    Bitmap outputImg = res.DrawReg(button4.ForeColor, checkBox1.Checked);
                    pictureBox1.Image = outputImg;
                    time.Stop();
                    label7.Text = $"绘图耗时:{time.ElapsedMilliseconds}ms";
                    GC.Collect();
                }
                catch (Exception ex)
                {
                    onnxStartFlag = false;
                    MessageBox.Show(ex.Message);
                }

            }
            onnxStartFlag = false;
        }
        #endregion

        #region OpenVINO模型加载
        private void button8_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "onnx (*.onnx)|*.onnx";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // 设置要加载的模型的路径，跟据需要改为你的模型名称
                string modelPath = openFileDialog.FileName;
                ModelInfo modelInfo = new ModelInfo();

                modelInfo = yolo_OpenVINO.LoadOnnxModel(modelPath, comboBox4.Text);

                label19.Text = $"task：{modelInfo.TaskName}";
                label18.Text = $"batch：{modelInfo.Batch}";
                label17.Text = $"imgsz：{modelInfo.Imgsz}";
                label16.Text = $"names：{modelInfo.LabelNames}";
                MessageBox.Show("OpenVINO模型加载成功!");
            }
        }
        #endregion

        #region OpenVINO模型推理
        bool openvinoStartFlag = false;
        private void button9_Click(object sender, EventArgs e)
        {
            if (openvinoStartFlag)
            {
                return;
            }
            openvinoStartFlag = true;
            Stopwatch time = new Stopwatch();
            if (img != null)
            {
                try
                {
                    img = new Bitmap(textBox1.Text);
                    float confidence = Convert.ToSingle(numericUpDown1.Value);
                    float iou = Convert.ToSingle(numericUpDown2.Value);
                    InferenceResult res = null;

                    res = yolo_OpenVINO.Inference(img, confidence, iou);

                    //var imgs = res.GetMaskImages();
                    //var img1 = res.GetCombineMaskImage();
                    //img1.Save("combineMaskImg.jpg");
                    //int index = 0;
                    //foreach (var item in imgs)
                    //{
                    //    item.Save($"mask{index}.jpg");
                    //    index++;
                    //}

                    img?.Dispose();

                    label22.Text = $"预处理耗时:{res.PreprocessTime}ms";
                    label23.Text = $"推理耗时:{res.InferenceTime}ms";
                    label20.Text = $"后处理耗时:{res.PostprocessTime}ms";

                    time.Restart();
                    Bitmap outputImg = res.DrawReg(button4.ForeColor, checkBox1.Checked);
                    pictureBox1.Image = outputImg;
                    time.Stop();
                    label24.Text = $"绘图耗时:{time.ElapsedMilliseconds}ms";
                    GC.Collect();
                }
                catch (Exception ex)
                {
                    openvinoStartFlag = false;
                    MessageBox.Show(ex.Message);
                }

            }
            openvinoStartFlag = false;
        }
        #endregion

        #region TensorRT模型加载
        private void button10_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "engine (*.engine)|*.engine";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // 设置要加载的模型的路径，跟据需要改为你的模型名称
                string modelPath = openFileDialog.FileName;

                bool res = yolo_TensorRT.LoadTensorRTModel(modelPath);

                if (res)
                {
                    MessageBox.Show("TensorRT模型加载成功!");
                }
                else
                {
                    MessageBox.Show("TensorRT模型加载失败!");
                }
            }
        }
        #endregion

        #region TensorRT加载Label标签
        private void button12_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "txt (*.txt)|*.txt";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                textBox2.Text = openFileDialog.FileName;
            }
        }
        #endregion

        #region TensorRT模型推理
        bool tensorRTStartFlag = false;
        private void button13_Click(object sender, EventArgs e)
        {
            if (tensorRTStartFlag)
            {
                return;
            }
            tensorRTStartFlag = true;
            Stopwatch time = new Stopwatch();
            if (img != null)
            {
                try
                {
                    img = new Bitmap(textBox1.Text);
                    float confidence = Convert.ToSingle(numericUpDown1.Value);
                    float iou = Convert.ToSingle(numericUpDown2.Value);
                    InferenceResult res = null;
                    string content = "";
                    if (!string.IsNullOrEmpty(textBox2.Text))
                    {
                        content = File.ReadAllText(textBox2.Text); // 加载并读取文本文件内容
                    }
                    res = yolo_TensorRT.Inference(img, (ModelType)comboBox1.SelectedIndex, content, confidence, iou);

                    //var imgs = res.GetMaskImages();
                    //var img1 = res.GetCombineMaskImage();
                    //img1.Save("combineMaskImg.jpg");
                    //int index = 0;
                    //foreach (var item in imgs)
                    //{
                    //    item.Save($"mask{index}.jpg");
                    //    index++;
                    //}

                    img?.Dispose();

                    label27.Text = $"预处理耗时:{res.PreprocessTime}ms";
                    label28.Text = $"推理耗时:{res.InferenceTime}ms";
                    label26.Text = $"后处理耗时:{res.PostprocessTime}ms";

                    time.Restart();
                    Bitmap outputImg = res.DrawReg(button4.ForeColor, checkBox1.Checked);
                    pictureBox1.Image = outputImg;
                    time.Stop();
                    label29.Text = $"绘图耗时:{time.ElapsedMilliseconds}ms";
                    GC.Collect();
                }
                catch (Exception ex)
                {
                    tensorRTStartFlag = false;
                    MessageBox.Show(ex.Message);
                }

            }
            tensorRTStartFlag = false;
        }
        #endregion

        #region onnx模型转engine
        private void button11_Click(object sender, EventArgs e)
        {
            FrmOnnxToEngine frmOnnxToEngine = new FrmOnnxToEngine();
            frmOnnxToEngine.Show();
        }
        #endregion

        #endregion

        #region 动态尺寸输入模型

        #region 加载图片
        private void button3_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Multiselect = true;
            openFileDialog.Filter = "图片|*.jpg;*.jpeg;*.bmp;*.png;";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                images.Clear();
                foreach (var item in openFileDialog.FileNames)
                {
                    if (openFileDialog.FileNames.Length > 0)
                    {
                        LoadImages(openFileDialog.FileNames);
                    }
                }
            }
        }


        List<Bitmap> images = new List<Bitmap>();
        private List<PictureBox> pictureBoxes = new List<PictureBox>();

        private void LoadImages(string[] imagePaths)
        {
            // 清空FlowLayoutPanel
            flowLayoutPanel1.Controls.Clear();
            pictureBoxes.Clear();
            images.Clear();

            foreach (string imagePath in imagePaths)
            {
                var img = new Bitmap(imagePath);
                PictureBox pictureBox = new PictureBox
                {
                    Width = flowLayoutPanel1.Height - 20,
                    Height = flowLayoutPanel1.Height - 20,
                    Margin = new Padding(3),
                    BorderStyle = BorderStyle.None,
                    SizeMode = PictureBoxSizeMode.Zoom,
                    Cursor = Cursors.Hand,
                    Tag = imagePath
                };

                pictureBox.Image = img;
                images.Add(new Bitmap(img));
                pictureBox.Click += PictureBox_Click;
                pictureBoxes.Add(pictureBox);
            }
            pictureBox2.Image = new Bitmap(images[0]);
            pictureBoxes[0].BorderStyle = BorderStyle.FixedSingle; // 设置第一个图片的边框样式为固定单线
            flowLayoutPanel1.Controls.AddRange(pictureBoxes.ToArray());
        }

        private void PictureBox_Click(object sender, EventArgs e)
        {
            PictureBox pictureBox = sender as PictureBox;
            string imagePath = pictureBox.Tag.ToString();

            foreach (var item in flowLayoutPanel1.Controls)
            {
                PictureBox pic = item as PictureBox;
                pic.BorderStyle = BorderStyle.None; // 重置其他图片的边框样式
            }

            pictureBox.BorderStyle = BorderStyle.FixedSingle;
            pictureBox2.Image = pictureBox.Image; // 显示选中图片在大图框中

        }
        #endregion

        #region Onnx模型加载
        private void button5_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "onnx (*.onnx)|*.onnx";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // 设置要加载的模型的路径，跟据需要改为你的模型名称
                string modelPath = openFileDialog.FileName;
                ModelInfo modelInfo = new ModelInfo();

                if (comboBox2.SelectedIndex == 0)
                {
                    modelInfo = yolo_Onnx_Dyamics.LoadOnnxModel(modelPath, false);
                }
                else
                {
                    modelInfo = yolo_Onnx_Dyamics.LoadOnnxModel(modelPath, true);
                }


                label32.Text = $"task：{modelInfo.TaskName}";
                label31.Text = $"batch：{modelInfo.Batch}";
                label30.Text = $"imgsz：{modelInfo.Imgsz}";
                label8.Text = $"names：{modelInfo.LabelNames}";
                MessageBox.Show("Onnx模型加载成功!");
            }
        }
        #endregion

        #region Onnx模型推理
        bool onnxStartFlag_Dynamic = false;
        private void button6_Click(object sender, EventArgs e)
        {
            if (onnxStartFlag_Dynamic)
            {
                return;
            }
            onnxStartFlag_Dynamic = true;
            Stopwatch time = new Stopwatch();
            if (images.Count > 0)
            {
                try
                {
                    int w = Convert.ToInt32(numericUpDown5.Value);
                    int h = Convert.ToInt32(numericUpDown6.Value);
                    float confidence = Convert.ToSingle(numericUpDown4.Value);
                    float iou = Convert.ToSingle(numericUpDown3.Value);


                    var res = yolo_Onnx_Dyamics.Inference_Dyamics(images, out InferTimeInfo inferTimeInfo, w, h, confidence, iou);

                    time.Restart();
                    if (res != null)
                    {
                        for (int i = 0; i < res.Length; i++)
                        {
                            Bitmap outputImg = res[i].DrawReg(button20.ForeColor, checkBox2.Checked);
                            pictureBoxes[i].Image = outputImg;
                        }
                        pictureBox2.Image = pictureBoxes[0].Image;
                    }
                    else
                    {
                        MessageBox.Show("推理异常！", "提示!");
                    }
                    time.Stop();
                    label35.Text = $"预处理耗时:{inferTimeInfo.PreprocessTime}ms";
                    label36.Text = $"推理耗时:{inferTimeInfo.InferenceTime}ms";
                    label33.Text = $"后处理耗时:{inferTimeInfo.PostprocessTime}ms";
                    label37.Text = $"绘图耗时:{time.ElapsedMilliseconds}ms";
                    GC.Collect();
                }
                catch (Exception ex)
                {
                    onnxStartFlag_Dynamic = false;
                    MessageBox.Show(ex.Message);
                }

            }
            onnxStartFlag_Dynamic = false;
        }
        #endregion

        #region OpenVINO模型加载
        private void button14_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "onnx (*.onnx)|*.onnx";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // 设置要加载的模型的路径，跟据需要改为你的模型名称
                string modelPath = openFileDialog.FileName;
                ModelInfo modelInfo = new ModelInfo();

                modelInfo = yolo_OpenVINO_Dyamics.LoadOnnxModel(modelPath, comboBox5.Text);

                label43.Text = $"task：{modelInfo.TaskName}";
                label42.Text = $"batch：{modelInfo.Batch}";
                label41.Text = $"imgsz：{modelInfo.Imgsz}";
                label40.Text = $"names：{modelInfo.LabelNames}";
                MessageBox.Show("OpenVINO模型加载成功!");
            }
        }
        #endregion

        #region OpenVINO模型推理
        bool openvinoStartDyamicsFlag = false;
        private void button15_Click(object sender, EventArgs e)
        {
            if (openvinoStartDyamicsFlag)
            {
                return;
            }
            openvinoStartDyamicsFlag = true;
            Stopwatch time = new Stopwatch();
            if (images.Count > 0)
            {
                try
                {
                    int w = Convert.ToInt32(numericUpDown5.Value);
                    int h = Convert.ToInt32(numericUpDown6.Value);
                    float confidence = Convert.ToSingle(numericUpDown4.Value);
                    float iou = Convert.ToSingle(numericUpDown3.Value);


                    var res = yolo_OpenVINO_Dyamics.Inference_Dyamics(images, out InferTimeInfo inferTimeInfo, w, h, confidence, iou);

                    time.Restart();
                    if (res != null)
                    {
                        for (int i = 0; i < res.Length; i++)
                        {
                            Bitmap outputImg = res[i].DrawReg(button20.ForeColor, checkBox2.Checked);
                            pictureBoxes[i].Image = outputImg;
                        }
                        pictureBox2.Image = pictureBoxes[0].Image;
                    }
                    else
                    {
                        MessageBox.Show("推理异常！", "提示!");
                    }
                    time.Stop();
                    label46.Text = $"预处理耗时:{inferTimeInfo.PreprocessTime}ms";
                    label47.Text = $"推理耗时:{inferTimeInfo.InferenceTime}ms";
                    label44.Text = $"后处理耗时:{inferTimeInfo.PostprocessTime}ms";
                    label48.Text = $"绘图耗时:{time.ElapsedMilliseconds}ms";
                    GC.Collect();
                }
                catch (Exception ex)
                {
                    openvinoStartDyamicsFlag = false;
                    MessageBox.Show(ex.Message);
                }
            }
            openvinoStartDyamicsFlag = false;
        }
        #endregion

        #region 标签颜色选择
        private void button20_Click(object sender, EventArgs e)
        {
            ColorDialog colorDialog = new ColorDialog();

            DialogResult res = colorDialog.ShowDialog();
            if (res == DialogResult.OK)
            {
                button20.ForeColor = colorDialog.Color;
                button20.Text = colorDialog.Color.Name.ToString();
            }
        }

        #endregion

        #endregion

        #region 目标跟踪
        private void button16_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "视频|*.mp4;";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                textBox3.Text = openFileDialog.FileName;
            }
        }

        private void button17_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "onnx (*.onnx)|*.onnx";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // 设置要加载的模型的路径，跟据需要改为你的模型名称
                string modelPath = openFileDialog.FileName;
                ModelInfo modelInfo = new ModelInfo();

                if (comboBox6.SelectedIndex == 0)
                {
                    modelInfo = onnxRuntimeTracker.LoadModel(modelPath, false);
                }
                else
                {
                    modelInfo = onnxRuntimeTracker.LoadModel(modelPath, true);
                }


                label52.Text = $"task：{modelInfo.TaskName}";
                label51.Text = $"batch：{modelInfo.Batch}";
                label50.Text = $"imgsz：{modelInfo.Imgsz}";
                label49.Text = $"names：{modelInfo.LabelNames}";
                MessageBox.Show("Onnx模型加载成功!");
            }
        }

        private void button18_Click(object sender, EventArgs e)
        {
            if (onnxRuntimeTracker.startFlag)
            {
                onnxRuntimeTracker.startFlag = false;
            }
            else
            {
                onnxRuntimeTracker.Start(textBox3.Text, (int)numericUpDown9.Value, (float)numericUpDown10.Value, (float)numericUpDown11.Value, (float)numericUpDown12.Value, (float)numericUpDown8.Value, (float)numericUpDown7.Value);

            }
        }

        private void OnnxRuntimeTracker_GetFrameEvent(FrameResult obj)
        {
            this.Invoke(new Action(() =>
            {
                pictureBox3.Image = obj.Image;
                label59.Text = $"帧率:{obj.FPS}";
            }));
        }

        private void OnnxRuntimeTracker_StopEvent()
        {
            this.Invoke(new Action(() =>
            {
                button18.Text = "检测";

            }));
        }

        private void OnnxRuntimeTracker_StartEvent()
        {
            this.Invoke(new Action(() =>
            {
                button18.Text = "停止";

            }));
        }


        private void button19_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "onnx (*.onnx)|*.onnx";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // 设置要加载的模型的路径，跟据需要改为你的模型名称
                string modelPath = openFileDialog.FileName;
                ModelInfo modelInfo = new ModelInfo();

                modelInfo = openVinoTracker.LoadModel(modelPath, comboBox7.Text);

                label65.Text = $"task：{modelInfo.TaskName}";
                label64.Text = $"batch：{modelInfo.Batch}";
                label63.Text = $"imgsz：{modelInfo.Imgsz}";
                label62.Text = $"names：{modelInfo.LabelNames}";
                MessageBox.Show("OpenVINO模型加载成功!");
            }
        }

        private void button21_Click(object sender, EventArgs e)
        {
            if (openVinoTracker.startFlag)
            {
                openVinoTracker.startFlag = false;
            }
            else
            {
                openVinoTracker.Start(textBox3.Text, (int)numericUpDown9.Value, (float)numericUpDown10.Value, (float)numericUpDown11.Value, (float)numericUpDown12.Value, (float)numericUpDown8.Value, (float)numericUpDown7.Value);

            }
        }

        private void OpenVinoTracker_GetFrameEvent(FrameResult obj)
        {
            this.Invoke(new Action(() =>
            {
                pictureBox3.Image = obj.Image;
                label68.Text = $"帧率:{obj.FPS}";
            }));
        }

        private void OpenVinoTracker_StopEvent()
        {
            this.Invoke(new Action(() =>
            {
                button21.Text = "检测";

            }));
        }

        private void OpenVinoTracker_StartEvent()
        {
            this.Invoke(new Action(() =>
            {
                button21.Text = "停止";

            }));

        }

        private void button24_Click(object sender, EventArgs e)
        {
            FrmOnnxToEngine frmOnnxToEngine = new FrmOnnxToEngine();
            frmOnnxToEngine.Show();
        }

        private void button25_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "engine (*.engine)|*.engine";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // 设置要加载的模型的路径，跟据需要改为你的模型名称
                string modelPath = openFileDialog.FileName;

                bool res = yolo_TensorRT.LoadTensorRTModel(modelPath);

                if (res)
                {
                    MessageBox.Show("TensorRT模型加载成功!");
                }
                else
                {
                    MessageBox.Show("TensorRT模型加载失败!");
                }
            }
        }

        private void button22_Click(object sender, EventArgs e)
        {
            if (tensorRTTracker.startFlag)
            {
                tensorRTTracker.startFlag = false;
            }
            else
            {
                string content = "";
                if (!string.IsNullOrEmpty(textBox4.Text))
                {
                    content = File.ReadAllText(textBox4.Text); // 加载并读取文本文件内容
                }
                tensorRTTracker.Start(textBox3.Text, (ModelType)comboBox8.SelectedIndex, content,(int)numericUpDown9.Value, (float)numericUpDown10.Value, (float)numericUpDown11.Value, (float)numericUpDown12.Value, (float)numericUpDown8.Value, (float)numericUpDown7.Value);

            }
        }

        private void TensorRTTracker_StopEvent()
        {
            this.Invoke(new Action(() =>
            {
                button22.Text = "检测";

            }));
        }

        private void TensorRTTracker_StartEvent()
        {
            this.Invoke(new Action(() =>
            {
                button22.Text = "停止";

            }));
        }

        private void TensorRTTracker_GetFrameEvent(FrameResult obj)
        {
            this.Invoke(new Action(() =>
            {
                pictureBox3.Image = obj.Image;
                label73.Text = $"帧率:{obj.FPS}";
            }));
        }

        #endregion


    }
}
