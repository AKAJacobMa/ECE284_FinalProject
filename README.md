# ECE284 Final Project

Using [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) (Wearable Stress and Affect Detection) dataset.

# Emotion and Stress Detection Using Classical Machine Learning

This repository presents a lightweight and interpretable machine learning pipeline for detecting emotional states using physiological signals collected from wearable devices. The project evaluates two classical ML models—Random Forest (RF) and Support Vector Machine (SVM)—on a 4-class classification task using the WESAD dataset.

##  Motivation

Stress affects both physical and mental health. Early detection is crucial for prevention, but many people don’t recognize stress until it's too late. Wearable devices offer an opportunity to monitor stress in real time through physiological signals like heart rate, skin conductance, and respiration.

The goal of this project is to develop an efficient, accurate, and interpretable stress detection system that is lightweight enough to be deployed on mobile or wearable devices.

##  Objectives

- Classify four emotional states: Baseline, Stress, Amusement, Meditation.
- Compare two classical ML models (Random Forest and SVM).
- Ensure the system is easy to interpret, accurate, and ready for real-world use.

##  Dataset

**WESAD (Wearable Stress and Affect Detection)**  
Collected from 15 subjects using a RespiBAN chest-worn device.

### Signals Collected:
- ECG (Electrocardiogram)
- EDA (Electrodermal Activity)
- EMG (Electromyography)
- TEMP (Temperature)
- RESP (Respiration)
- ACC (Accelerometer)

### Experimental States:
- Baseline (calm)
- Stress (arithmetic + public speaking)
- Amusement (funny videos)
- Meditation (rest/transition)

> Note: This implementation uses only subject `S2` for demonstration.

##  Methodology

1. **Data Loading**: Parse raw `.pkl` files into NumPy arrays.
2. **Preprocessing**:
   - Remove incomplete/missing samples.
   - Normalize features using `StandardScaler`.
   - Balance data (equal samples per class).
3. **Model Training**:
   - Train SVM and RF with hyperparameter tuning using Optuna.
4. **Evaluation**:
   - Accuracy, Precision, Recall, F1-score (per class)
   - Confusion matrices
   - Per-class performance plots

##  File Structure

```bash
.
├── rf.py                       # Random Forest training and evaluation
├── svm.py                      # SVM training and evaluation
├── feature_visualization.ipynb # Optional notebook for feature plotting
├── training_visualization.ipynb # Accuracy and metrics visualization
├── training_records/           # Auto-saved confusion matrix and params (generated after training)
└── Visualization/              # Generated Visualization Images



