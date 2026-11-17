# SkinAugment AI

**Consumer Skincare Photo Augmenter**

SkinAugment AI is an image augmentation research project that generates realistic variations of skincare photographs. It aims to improve the robustness of skincare image classification models under different image conditions.

This project is an image augmentation tool, not a medical diagnostic system.

## Features

- Upload skincare photographs.
- Generate multiple augmented image variations.
- Preview and download generated images.
- Classify uploaded images using EfficientNet-B0.
- Display predicted categories and model confidence.

## Dataset

The cleaned dataset contains **9,455 original images** across five categories:

- Acne
- Blackheads
- Dark spots
- Pores
- Wrinkles

The training dataset was augmented while the validation and test datasets were kept unchanged.

## Technologies Used

- Python
- PyTorch
- EfficientNet-B0
- Streamlit
- OpenCV
- Pillow
- NumPy
- Matplotlib

## Methodology

1. Dataset collection and cleaning.
2. Duplicate removal and group-aware dataset splitting.
3. Image preprocessing.
4. Image augmentation.
5. EfficientNet-B0 model training.
6. Model evaluation.
7. Streamlit application development and deployment.

## Model Performance

| Metric | Result |
|---|---:|
| Model | EfficientNet-B0 |
| Test Accuracy | 91.95% |
| Number of Classes | 5 |

The augmented model improved test accuracy by approximately 2.01 percentage points compared with the original model.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/AP24110010260/SkinAugment-AI.git
cd SkinAugment-AI
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Model Checkpoint

The trained model checkpoint is available at:

`models/efficientnet_b0_phase4_best.pth`

## Live Application

[Open SkinAugment AI](https://ap24110010260-skinaugment-ai-app-ietyq4.streamlit.app/)

## Disclaimer

SkinAugment AI is developed for image augmentation and research purposes only. It does not provide medical diagnosis or treatment recommendations.
