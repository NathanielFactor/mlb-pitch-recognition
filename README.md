# MLB Pitch Recognition from Video Clips Using CNN + LSTM

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Problem Statement](#problem-statement)  
- [Motivation](#motivation)  
- [Methodology](#methodology)  
  - [CNN + LSTM Architecture](#cnn--lstm-architecture)  
  - [On-the-Fly Frame Extraction](#on-the-fly-frame-extraction)  
- [Dataset](#dataset)  
- [Engineering Process](#engineering-process)  
- [Installation & Setup](#installation--setup)  
- [Usage](#usage)  
  - [Training](#training)  
  - [Inference](#inference)  
- [Project Structure](#project-structure)  
- [Future Work & Roadmap](#future-work--roadmap)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  
- [Contact](#contact)

---

## Project Overview

This project develops a deep learning pipeline to classify MLB pitch types from short video clips of pitchers using a hybrid **Convolutional Neural Network (CNN) + Long Short-Term Memory (LSTM)** model. The CNN extracts spatial features from individual video frames, while the LSTM models the temporal dynamics of pitching mechanics to identify pitch types such as fastballs, curveballs, sliders, and more.

---

## Problem Statement

Pitch recognition is a challenging problem due to subtle motion differences and high-speed video data. Accurate automatic classification from video can benefit broadcasters, coaches, and fans by providing real-time pitch analysis without manual annotation.

Our goal is to create a model that, given a 4-second clip of a pitcher’s throw, predicts the pitch type with high accuracy, while efficiently processing video data through frame sampling and sequential modeling.

---

## Motivation

- **Enhance sports analytics** by automating pitch type detection.
- Provide a **real-time tool** for broadcasters and coaches.
- Explore applications of **deep learning in computer vision and sequence modeling**.
- Build a foundation for more advanced biomechanics or action recognition research.

---

## Methodology

### CNN + LSTM Architecture

- **CNN (Convolutional Neural Network):** Extracts meaningful spatial features from each video frame. We use pretrained CNN backbones (e.g., ResNet18) fine-tuned for this task.
- **LSTM (Long Short-Term Memory):** Processes the sequence of frame features to learn temporal patterns and motion dynamics across frames.
- **Classifier Head:** Fully connected layers convert LSTM outputs into pitch type probabilities.

### Architecture Diagram

```plaintext
Video Clip: [Frame_1, Frame_2, ..., Frame_T]

           ↓           ↓           ↓
         CNN         CNN         CNN
           ↓           ↓           ↓
        f_1         f_2         f_T  → Feature Sequence [T, D]
           \           |           /
                 LSTM (Sequence Model)
                        ↓
                  Fully Connected Layer
                        ↓
                Pitch Type Classification
```
---

### On-the-Fly Frame Extraction

- Use **OpenCV** to read videos directly.
- Sample a fixed number of frames (e.g., 16) evenly spaced across the clip.
- Apply transforms and normalization on-the-fly to avoid storage overhead.
- Enables dynamic data loading suitable for training and inference.

---

## Dataset

- Dataset format: video files organized by pitch type folders or CSV with video paths and labels.
- Data augmentation can include random cropping, color jitter, or temporal jittering.
- Currently seeking publicly available MLB pitch video datasets or constructing a custom dataset.

---

## Engineering Process (V-Cycle Model)

1. **Requirement Analysis:** Define accuracy goals, latency requirements, and dataset needs.
2. **System Design:** Architect data pipelines, model components, and training workflows.
3. **Implementation:** Develop modular code for dataset loading, model architecture, training, and inference.
4. **Verification:** Validate model performance on validation sets, perform error analysis.
5. **Validation:** Deploy inference pipeline for single video prediction, test with new inputs.
6. **Maintenance:** Update dataset, retrain models, optimize for deployment efficiency.

---

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mlb-pitch-recognition.git
   cd mlb-pitch-recognition
   ```

2. (Optional) Create and activate a Python virtual environment:

- On Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

- On Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
Usage
Training
Update train.py with your dataset paths and training parameters.

Run training with:

```bash
python train.py
Monitor training logs and metrics.
```

Save best-performing model checkpoints.

Inference
Use inference.py to predict pitch types on new videos:

```bash
python inference.py --video_path path/to/video.mp4
Outputs predicted pitch type with confidence scores.
```

## Project Structure

├── data_loader.py         # Dataset class and video processing logic
├── model.py               # CNN + LSTM model definition
├── train.py               # Training loop and utilities
├── inference.py           # Single video prediction pipeline
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # MIT License file
└── videos/                # Example pitch video clips

---

## Future Work & Roadmap

- User Interface: Integrate Gradio or Streamlit for web-based video upload and real-time pitch prediction.
- 3D CNN Exploration: Test 3D convolutional architectures (e.g., ResNet3D) for potentially better temporal feature extraction.
- Dataset Expansion: Collect more labeled videos for diverse pitchers and pitch types.
- Model Optimization: Quantization and pruning for mobile and edge deployment.
- Multi-Modal Inputs: Incorporate sensor data or broadcast audio for enhanced predictions.

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## Acknowledgments

Inspired by deep learning research in action recognition and sports analytics.

Thanks to open-source contributors of PyTorch, OpenCV, and torchvision.

Special thanks to mentors and collaborators providing guidance and feedback.

---

## Contact

If you have any questions, suggestions, or collaboration ideas, please reach out:

Email: nathaniel.factor@mail.mcgill.ca

GitHub: NathanielFactor
