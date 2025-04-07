# Dog vs. Cat Real-Time Classifier

This project implements a real-time dog and cat classifier using a deep learning model on an NVIDIA RTX 4090 GPU, with live webcam feed processing and bounding box visualization. Built with TensorFlow and OpenCV, it achieves 93.65% accuracy on a custom-trained CNN, with ongoing optimization to hit 30+ FPS (currently at 9.5 FPS).

The goal is to detect dogs or cats in a video stream and display their classification with a bounding box, originally designed to prevent dogs from accessing a litter box when paired with a Raspberry Pi actuator setup.

## Features
- Real-time classification from a 720p webcam feed.
- Bounding box visualization (centered, ~360x360) in green (dog) or red (cat).
- FPS and confidence score overlay.
- GPU-accelerated preprocessing and inference via TensorFlow.
- TensorRT optimization for enhanced performance (in progress).

## Prerequisites
- Hardware: NVIDIA GPU (tested on RTX 4090, 24 GB VRAM).
- OS: Ubuntu 24.04 (tested), with NVIDIA drivers and CUDA installed.
- Python: 3.12 (or 3.10 for TensorRT stability).
- Dependencies: TensorFlow, OpenCV, NVIDIA TensorRT.

## Installation
1. Clone the Repository:
   ```
   git clone https://github.com/yourusername/dog-cat-classifier.git
   cd dog-cat-classifier
   ```

2. Set Up Virtual Environment:
   ```
   python3 -m venv tf_env
   source tf_env/bin/activate
   ```

3. Install Dependencies:
   ```
   pip install tensorflow opencv-python nvidia-tensorrt
   ```

4. Install NVIDIA TensorRT (System Level):
   ```
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
   sudo apt update
   sudo apt install -y libnvinfer8 libnvinfer-dev libnvinfer-plugin8
   ```

5. Model File:
   - Place `dog_cat_model_4090.keras` in the repo root (not included due to size—download from [link TBD] or train your own).

## Usage
1. Convert Model to TensorRT (optional, for speed):
   ```
   python3 convert_to_trt.py
   ```
   - Outputs `dog_cat_model_trt/`—enhances inference to ~5 ms.

2. Run Webcam Classifier:
   ```
   python3 webcam_dog_cat_realtime.py
   ```
   - Press `q` to quit.
   - Displays 720p feed with a centered bounding box and classification.

## Performance
- Current: 9.5 FPS, ~80W GPU usage (RTX 4090).
- Target: 30+ FPS, 150-250W GPU usage.
- Monitor with:
  ```
  watch -n 1 nvidia-smi
  ```

## Files
- `webcam_dog_cat_realtime.py`: Main script for real-time webcam classification with bounding box.
- `convert_to_trt.py`: Converts the Keras model to TensorRT for performance boost.
- `dog_cat_model_4090.keras`: Trained model (93.65% accuracy, not included—see Installation).

## Optimization Notes
- Current bottleneck: ~105 ms/frame (9.5 FPS). Likely preprocessing or inference—TensorRT should drop inference to ~5 ms.
- Next steps: If FPS lags, check timing output and consider Python 3.10/TensorFlow 2.15 downgrade.

## Contributing
Feel free to fork, submit issues, or PRs—especially for FPS boosts or true bounding box detection (e.g., YOLO/SAM integration).

## License
MIT License—free to use, modify, and distribute.

## Acknowledgments
- Built with help from xAI’s Grok—thanks for the assist!
- Inspired by the need to keep dogs out of litter boxes.
