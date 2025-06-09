# ECE284 Final Project

Using [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) (Wearable Stress and Affect Detection) dataset.

# Stress Detection from Wearable Signals using Classical Machine Learning

This repository contains code for a stress and emotion detection system using classical machine learning methods on physiological data from wearable devices.

## Project Summary

I use the **WESAD dataset**, collected from chest-worn RespiBAN sensors, to classify four emotional states:
- Baseline
- Stress
- Amusement
- Meditation

Two machine learning models are implemented:
- **Random Forest (RF)**
- **Support Vector Machine (SVM)**

## Why Classical ML?

Deep learning methods often require large datasets and are difficult to interpret. This project shows that **well-preprocessed features and balanced sampling** enable classical ML models to achieve **high accuracy** (RF ~99.9%) in stress detection.

## 📁 File Structure

```bash
.
├── rf.py                       # Random Forest training and evaluation
├── svm.py                      # SVM training and evaluation
├── feature_visualization.ipynb # Optional notebook for feature plotting
├── training_visualization.ipynb # Accuracy and metrics visualization
├── training_records/           # Auto-saved confusion matrix and params (generated after training)
└── WESAD/                      # Folder to place WESAD dataset (not included here)


