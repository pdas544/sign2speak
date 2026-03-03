# sign2speak

## 🚀 Overview
Sign2Speak is a Python-based project that converts Sign Language to Speech in real-time English and Hindi Audio using PyTorch and TTS. 
This project aims to bridge the communication gap between people who use sign language and those who do not, by providing a real-time translation service.

### Key Features
- Real-time conversion of Sign Language to Speech
- Utilizes PyTorch for deep learning models
- TTS (Text-to-Speech) integration for natural speech output
- Easy-to-use API for developers

### Who This Project Is For
- Sign language interpreters
- Developers interested in computer vision and machine learning
- Researchers in the field of sign language recognition
- Anyone looking to improve accessibility

## ✨ Features
- 📊 **Data Analysis**: Tools to analyze and balance datasets
- 📹 **Video Processing**: Extract keypoints from videos
- 🧠 **Model Training**: Train deep learning models for sign language recognition
- 🎤 **TTS Integration**: Convert recognized signs to natural speech
- 📈 **Evaluation**: Evaluate model performance with detailed metrics

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Frameworks**: PyTorch, TTS
- **Libraries**: MediaPipe, OpenCV, Pandas, NumPy
- **Tools**: Jupyter Notebook, GitHub Actions

## 📦 Installation

### Prerequisites
- Python 3.8 or later
- PyTorch
- MediaPipe
- OpenCV
- Pandas
- NumPy
- TTS

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/sign2speak.git

# Make Sure Python is installled, check with the following command:
python --version

# Create a Virtual Environment
python -m venv myenv

# Activate the Virtual Environment
source myenv/bin/activate

# Navigate to the project directory
cd sign2speak

# Install dependencies
pip install -r requirements.txt

# Run the prediction script
python realtime_prediction.py
```

## 🎯 Usage

### Basic Usage
```python
# Example of using the model to predict a sign
# Run the file named "realtime_prediction.py"
python realtime_prediction.py

```

**It opens the attached/integrated webcam which waits for the sign language**
**Once the sign language is captured, it is recognized and gets translated into english and hindi audio in the audio directory**


### Advanced Usage
- **Custom Model Training**: Execute the file extract_keypoints_lstm.py -> Opens the Integrated/Attached Camera and captures 
- **Data Augmentation**: Use the `KeypointAugmentation` class to augment keypoints for better training.
- **Evaluation**: Run the `evaluate.py` script to evaluate the model's performance.

## 📁 Project Structure
```
sign2speak/
│
├── .gitignore
├── requirements.txt
├── analyze_dataset.py
├── analyze-msasl.py
├── analyze.py
├── download-asl-old_2.py
├── download-asl-old.py
├── download-videos.py
├── extract-keypoints-full.py
├── filtered_annotations_selected_glosses.json
├── filtered_annotations_top_10_old.json
├── model_baseline.py
├── model_transformer.py
├── model.py
├── move_videos.py
├── MSASL_test.json
├── MSASL_train.json
├── MSASL_val.json
├── sign2speech_pipeline_tcn.py
├── updated-filtered-annotations.py
├── processed/
│   ├── unmatched_videos.txt
│   └── ...
├── src/
│   ├── dataload.py
│   ├── deploy.py
│   ├── evaluate.py
│   ├── model.py
│   └── training.py
└── README.md
```

## 🔧 Configuration
- **Configuration Files**: Modify `config.json` for model parameters and other settings.

## 🤝 Contributing
- Fork the repository
- Create a new branch for your feature or bug fix
- Write clean, well-commented code
- Submit a pull request


## 👥 Authors & Contributors
- **Maintainers**: Priyabrata Das

## 🐛 Issues & Support
- Report issues on the [GitHub Issues page](https://github.com/pdas544/sign2speak/issues)
- Get help on the [GitHub Discussions page](https://github.com/pdas544/sign2speak/discussions)

## 🗺️ Roadmap
- **Future Improvements**:
  - Real-time video processing
  - Mobile app integration
  - Addition of other languages

---

**Badges:**
[![Build Status](https://github.com/pdas544/sign2speak/workflows/CI/badge.svg)](https://github.com/pdas544/sign2speak/actions)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/pdas544/sign2speak)](https://github.com/pdas544/sign2speak/stargazers)
[![Forks](https://img.shields.io/github/forks/pdas544/sign2speak)](https://github.com/pdas544/sign2speak/network/members)
