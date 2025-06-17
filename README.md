# Lightsaber Detector 🔦🧠  
*A Convolutional Neural Network for Lightsaber Segmentation in Video Duels*

![example pixel labeling](photos/image1.png)

> ⚠️ This is a work-in-progress hobby project to detect lightsabers at the pixel level using deep learning. The goal is to eventually integrate with pose estimation models (e.g., OpenPose) to enable automatic hit detection in live-action lightsaber duels for use in auto-refereeing systems.

---

## Table of Contents
- [Overview](#overview)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Training Data](#training-data)
- [Training Process](#training-process)
- [Output Visualization](#output-visualization)
- [Requirements](#requirements)
- [License](#license)
- [TODO](#todo)

---

## Overview

This project uses a U-Net convolutional neural network to perform pixel-wise segmentation of lightsabers in video frames. The model is trained on manually annotated footage of mock lightsaber duels and is designed to label only pixels that correspond to lightsabers.

The long-term vision is to combine this model with body segmentation and pose detection frameworks (e.g., OpenPose) to automatically referee duels by identifying valid hits.

---

## How to Run

### Step 1: Extract Lightsaber Duel Footage

```bash
python MAIN_test_label_video_data.py

data/raw_video_training_test_data/test_footage.mp4


