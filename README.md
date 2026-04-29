# Posture Detection System

(Own dataset is needed)
130 images each of
- Slouching
- Leaning_back
- Proper

## Overview
Real-time posture detection using MobileNetV2 and computer vision.  
Classifies:
- Leaning Back
- Proper
- Slouch

## Features
- Deep learning (MobileNetV2)
- Real-time webcam detection
- Cropped input for better accuracy
- Custom dataset with blurred faces (privacy-safe)

## Setup

```bash
git clone https://github.com/Egg-Lord/posture-detection.git
cd posture-detection
conda create -n posture python=3.10
conda activate posture
pip install -r requirements.txt
