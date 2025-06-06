# ECE284 Final Project

Using [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) (Wearable Stress and Affect Detection) dataset.

This project implements and compares Random Forest and Support Vector Machine (SVM) classifiers for detecting emotional states from physiological signals collected from wearable sensors, based on the WESAD dataset.


Dataset
The project uses the WESAD dataset. The signals extracted include:
Accelerometer (x, y, z)
ECG (Electrocardiogram)
EMG (Electromyogram)
EDA (Electrodermal Activity)
Temperature
Respiration

Classify four emotional states:
1: Baseline
2: Stress
3: Amusement
4: Meditation

How to Run

Run rf.py or svm.py directly:

python rf.py
python svm.py

Dependencies
Python 3.8+
scikit-learn
pandas, numpy
Optuna

