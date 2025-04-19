# 🛣️ Road Condition Classification Using LSTM

This project focuses on detecting various types of road conditions using motion sensor data collected from a smartphone attached to a two-wheeler. The primary objective is to build a classifier using LSTM (Long Short-Term Memory) that can identify:

- 🛣️ Bitumen Road  
- 🧱 Kankar Road  
- 🧩 Concrete Road  
- 🔘 Single Speed Breaker  
- 🔘 Multiple Speed Breakers  

---

## 📱 Android App: Data Collection

An Android application was developed to capture motion data from the phone’s *accelerometer* and *gyroscope* sensors at a fixed interval (50 Hz). Each sample corresponds to a 3-second window of motion during the ride.

### Collected Features:
- Timestamp  
- X, Y, Z Accelerometer values  
- X, Y, Z Gyroscope values  

---
## Rash Driving Detection:

In addition to road condition classification, we have introduced a *rash driving detection system*. This feature monitors abnormal or aggressive riding behavior based on spikes in sensor data such as:

- Sudden acceleration or deceleration  
- High-frequency changes in gyroscope readings (sharp turns or instability)  
- Frequent and harsh jerks indicating unsafe maneuvers  


## 🧠 Model: LSTM Classifier

A deep learning model using an LSTM architecture was trained on the collected data. Each input sequence (3 seconds long) is classified into one of the five road types.

### Model Architecture:
- LSTM layer with 64 units  
- Dropout for regularization  
- Dense layer with ReLU  
- Output layer with softmax activation  

---

## 🧪 Results

- ✅ *Test Accuracy:* ~86%
- 📉 Evaluated using accuracy score and confusion matrix

## 🗂️ Project Structure
' Root Folder: NNProject
' ├── Data\
' │   ├── X.npy                  ' Numpy array of input features
' │   ├── y.npy                  ' Numpy array of labels
' │   └── processed_dataset.csv  ' Final scaled dataset for training/testing
' ├── models\
' │   ├── lstm_classifier.h5     ' Saved LSTM model
' │   └── evaluate_model.py      ' Script to load model and evaluate test data
' ├── requirements.txt           ' Python dependencies required to run the project
' └── README.md                  ' Project documentation and overview