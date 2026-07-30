# Emotion-detector

# Jetson Emotion Detector

## Project Overview

I built this project to detect facial expressions in real time using an NVIDIA Jetson Orin Nano. The system detects a face, crops it, and classifies the visible expression.

This project is the first stage of a larger debate body-language analysis system. My goal is to use computer vision to analyze how speakers present arguments during debates.

## Model Development

I first trained an eight-class model using FERPlus:

* Angry
* Contempt
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

Contempt was difficult to classify consistently and was often confused with other classes. I removed the contempt class and retrained the model with seven classes:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## System Pipeline

```text
USB camera
    ↓
FaceNet detects faces
    ↓
The largest face is selected and cropped
    ↓
ResNet-18 classifies the expression
    ↓
The label and confidence are displayed
```

FaceNet handles face detection. A custom ResNet-18 model handles emotion classification. Both models run on the Jetson Orin Nano.

## Training Results

```text
Best epoch:          34
Training loss:       0.69492
Training accuracy:   73.9564%
Validation loss:     0.72810
Validation accuracy: 75.2758%
```

The final model was exported to ONNX for deployment.

## Hardware

* NVIDIA Jetson Orin Nano Developer Kit
* Approximately 8 GB RAM
* USB camera
* NVIDIA JetPack
* CUDA 12.2
* 15W power mode

## Software

* Python 3
* PyTorch
* Torchvision
* NVIDIA Jetson Inference
* CUDA
* TensorRT
* ONNX
* FaceNet
* ResNet-18
* Git
* Git LFS

## Dataset

The model was trained using FERPlus:

https://www.kaggle.com/datasets/arnabkumarroy02/ferplus?select=validation

FERPlus contains cropped facial-expression images without face bounding boxes. FaceNet detects and crops faces before classification.

The dataset is not included in this repository.

```text
data/emotions/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── test/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── validation/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## Repository Structure

```text
Emotion-detector/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
├── src/
│   ├── train.py
│   ├── live_emotion.py
│   └── onnx_export.py
├── models/
│   └── emotions7_v2/
│       ├── resnet18.onnx
│       ├── model_best.pth.tar
│       └── labels.txt
├── docs/
│   └── project-report.md
└── media/
    ├── live-demo.jpg
    ├── training-results.png
    └── system-diagram.png
```

## Installation

Clone the repository:

```bash
git clone git@github.com:peter-xu261/Emotion-detector.git
cd Emotion-detector
```

Install Git LFS and download the model files:

```bash
sudo apt update
sudo apt install -y git-lfs
git lfs install
git lfs pull
```

Install NVIDIA Jetson Inference if needed:

```bash
cd ~
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build
cd build
cmake ../
make -j$(nproc)
sudo make install
sudo ldconfig
```

Installation steps may vary by JetPack version or Docker setup.

## Model Files

Deployment files:

```text
models/emotions7_v2/resnet18.onnx
models/emotions7_v2/labels.txt
```

Training checkpoint:

```text
models/emotions7_v2/model_best.pth.tar
```

Labels:

```text
angry
disgust
fear
happy
neutral
sad
surprise
```

## Export and Test

Export the trained model from the Jetson Inference classification directory:

```bash
python3 onnx_export.py \
  --model-dir=models/emotions7_v2
```

Verify the files:

```bash
ls -lh models/emotions7_v2/
```

### Test One Image

```bash
NET="models/emotions7_v2"
IMAGE=$(find data/emotions/val -type f | head -n 1)

imagenet.py \
  --model="$NET/resnet18.onnx" \
  --labels="$NET/labels.txt" \
  --input_blob=input_0 \
  --output_blob=output_0 \
  "$IMAGE" \
  test_result.jpg
```

### Test the Camera Classifier

This command classifies the entire camera frame:

```bash
NET="$HOME/jetson-inference/python/training/classification/models/emotions7_v2"

DISPLAY=:0 imagenet.py \
  --model="$NET/resnet18.onnx" \
  --labels="$NET/labels.txt" \
  --input_blob=input_0 \
  --output_blob=output_0 \
  --input-width=640 \
  --input-height=480 \
  --input-rate=20 \
  /dev/video0 \
  display://0
```

### Run the Full Emotion Detector

```bash
NET="$HOME/jetson-inference/python/training/classification/models/emotions7_v2"

DISPLAY=:0 python3 "$HOME/jetson-inference/live_emotion.py" \
  /dev/video0 \
  display://0 \
  --model="$NET/resnet18.onnx" \
  --labels="$NET/labels.txt" \
  --input-blob=input_0 \
  --output-blob=output_0 \
  --face-network=facenet \
  --face-threshold=0.15 \
  --emotion-threshold=0.05 \
  --padding=0.05 \
  --smoothing=5 \
  --classify-every=2 \
  --input-width=640 \
  --input-height=480 \
  --input-rate=20
```

Press `Ctrl+C` to stop.

## Performance

Enable Jetson performance mode:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Monitor system usage:

```bash
sudo tegrastats --interval 1000
```

## Limitations

The system classifies only the largest detected face. It may produce less accurate results with poor lighting, blur, head rotation, small faces, or partial face coverage.

The model may confuse similar expressions, including fear and surprise, anger and disgust, or neutral and sadness.

The model classifies visible facial expressions only. It cannot determine a person’s true feelings, intentions, or mental state.

FaceNet and ResNet-18 run sequentially on the GPU, which can create processing delay.

## Removing the Contempt Class

I first trained an eight-class model. Contempt was difficult to classify because it often appears as a small facial movement and closely resembles other expressions.

I removed contempt and retrained the model with seven classes. The final model reached 75.2758% validation accuracy.

## Future Improvements

Planned improvements include:

* Multi-face classification
* Custom webcam training data
* Better performance under varied lighting
* Confusion-matrix analysis
* Faster inference
* Head-pose, gesture, and posture analysis
* Integration with my debate-training application

## Ethical Use

This project is an educational facial-expression classifier. Predictions may be incorrect and should not be used for employment, education, policing, healthcare, credibility assessment, or judgments about a person’s character.

## Demo

Demo video:

```text
Add demo video link here
```

## Acknowledgments

This project uses NVIDIA Jetson Inference, Jetson Orin Nano, PyTorch, Torchvision, FERPlus, FaceNet, and ResNet-18.

Jetson Inference:

https://github.com/dusty-nv/jetson-inference

## License

This project uses the license included in the `LICENSE` file.
