# Web_interface_har
Explored Explainable AI and Attention Mechanisms in  Human Activity Recognition &amp; Created a Web Interface which uses the trained model to predict Activity

In this project, A human activity recognition (HAR) system is successfully developed and evaluated using Long Short-Term Memory (LSTM) models. We employed the MHEALTH 
dataset, containing 12 activity classes, to train and test these models.The high-performing LSTM model was integrated with a web application. This user-friendly interface allows users to upload live sensor data from their mobile devices (accelerometer, magnetometer, and gyroscope) in CSV format. The application then utilizes the integrated model to predict the user's activity in real-time.
This project highlights the potential of deep learning models for HAR tasks. By combining LSTM models with XAI techniques like LIME ( Local Interpretable Model-Agnostic 
Explanations, Not only high accuracy can be achieved but insights can also be gained into the decision-making process of the model. 
The web application integration further enhances the usability and accessibility of this technology.
This project establishes a solid foundation for further development and exploration in the field of HAR using deep learning and explainable AI.



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras import layers
import keras
! pip install -q -U keras-tuner
import keras_tuner as kt
import warnings
warnings.filterwarnings('ignore')

base_dir = '/kaggle/input/mhealth/data/mHealth_subject'

df = pd.DataFrame()
for i in range(10):
    df_tmp = pd.read_csv(base_dir + str(i+1) + '.csv', header=0)
    df = pd.concat([df, df_tmp])

# View top 5 rows of dataframe
df.head()

df.Label.value_counts()

# DownSamping 
from sklearn.utils import resample
 
df_majority = df[df.Label==0]
df_minorities = df[df.Label!=0]
 
df_majority_downsampled = resample(df_majority,n_samples=40000)
df = pd.concat([df_majority_downsampled, df_minorities])

df.Label.value_counts()

split_point = int(len(df) * 0.8)
train_data = df.iloc[:split_point, :]
test_data = df.iloc[split_point:, :]

print("Number of train spamples: ", len(train_data))
print("Number of test spamples: ", len(test_data))
print("Number of total spamples: ", len(df))

def concat(data):
    
    # Select right arm data
    rgt_arm = data.iloc[:,15:24]
    rgt_arm.columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz']
    
    # Extract labels 
    labels = data.iloc[:, -1] 
    labels = labels.to_frame()
    labels.columns=['Activity_Label']
    
    df = rgt_arm
   
    return df, labels

# Generate input data and labels
X, y = concat(df)
train_X, train_y = concat(train_data)
test_X, test_y = concat(test_data)

from scipy import stats
from sklearn import metrics

%matplotlib inline

train_X.head()

print(type(train_X))
print(type(train_y))
print(type(X))

TIME_STEPS = 23 #sliding window length
STEP = 10 #Sliding window step size
N_FEATURES = 9 

#function to create time series datset for seuence modeling
def create_dataset(X, y, time_steps, step):
    Xs, ys = [], []
    
    for i in range(0, len(X) - time_steps, step):
        x = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]

        mode_result = stats.mode(labels)
        
        if np.isscalar(mode_result.mode):
            mode_value = mode_result.mode
        elif len(mode_result.mode) > 0:
            mode_value = mode_result.mode[0]
        else:
            mode_value = labels.values[0]

        ys.append(mode_value)
        Xs.append(x)

    return np.array(Xs), np.array(ys).reshape(-1, 1)

train_X, train_y = create_dataset(train_X, train_y, 23, step=10)
test_X, test_y = create_dataset(test_X, test_y, 23, step=10)

train_X.shape, train_y.shape

train_y

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Define activity labels as a list
labelss = ['Unknown','Standing still', 'Sitting and relaxing', 'Lying down', 'Walking',
          'Climbing stairs', 'Waist bends forward', 'Frontal elevation of arms',
          'Knees bending (crouching)', 'Cycling', 'Jogging', 'Running', 'Jump front & back']  
         # Added a background class at class 0

y_train_cat = to_categorical(train_y, num_classes=len(labelss))  # One-hot encode for multi-class
y_test_cat = to_categorical(test_y, num_classes=len(labelss))

# Define the model
model = keras.Sequential()
model.add(keras.Input(shape=(23, 9)))

model.add(LSTM(512, return_sequences=True, activation='relu'))
model.add(LSTM(256, return_sequences=False, activation='relu'))

# Dense layer for feature extraction
model.add(Dense(128, activation='relu'))

# Output layer with softmax activation for probability distribution
model.add(Dense(len(labelss), activation='softmax'))

print(model.summary())

from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.activations import softmax
from keras.optimizers import Adam

# Assuming labels is a list containing class labels
num_classes = len(labelss)

# Define the model
model = Sequential()

# Input layer with 23 timesteps, each with 9 features
model.add(keras.Input(shape=(23, 9)))

# Stacked LSTMs for learning temporal dependencies
model.add(LSTM(512, return_sequences=True, activation='relu'))  # First LSTM layer
model.add(LSTM(256, return_sequences=False, activation='relu'))  # Second LSTM layer

# Dense layer for feature extraction
model.add(Dense(128, activation='relu'))

# Output layer with softmax activation (already included)
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())  # View layer details

# Compile the model (adjust hyperparameters as needed)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_X, y_train_cat, epochs=30, batch_size=128, validation_data=(test_X, y_test_cat))

model.save('lstm_timeseries_model.h5')

# Prediction and getting class with highest probability
#predictions = model.predict(X_new)  # Get softmax probabilities
#predicted_classes = np.argmax(predictions, axis=1)  # Apply argmax to get class index

model.save('op_lstm.h5')

import os 
os.chdir(r'/kaggle/working')

#%cd /kaggle/working

#Now save your required files in this Directory .
model.save('op_lstm3.h5')

#Then run the following cell
from IPython.display import FileLink 
FileLink(r'/kaggle/working/op_lstm3.h5')

Data Collection:
A Application named ‘Sensor Logger’[15] is used to collect the data from mobile 
sensors
Sensor Logger Overview:
•Sensor Logger is a free, cross-platform data logger designed to record readings from 
various motion-related sensors found in smartphones.
•It supports sensors like Accelerometer, Gyroscope, Magnetometer, Barometer, GPS,

Microphone, Camera, Pedometer, Heart Rate (with the iOS watch app), and more.
•The app allows exporting recorded data in various formats including .zip, .csv, .json, 
.kml, and .sqlite for further analysis.
•Sensor Logger supports live data streaming via HTTP Push to a specified URL.
•The streamed data is in JSON format and contains sensor readings with timestamps.
•Setting up a server to receive and process live data is necessary for real-time 
visualization or analysis

![sensorlogger_app](https://github.com/user-attachments/assets/cc385676-5d63-45fc-ad5f-7d4f860f6471)

Web Interface :

![webpic](https://github.com/user-attachments/assets/06b059de-4ee8-4843-bfff-6e8529c725d8)

Result :

![act_res](https://github.com/user-attachments/assets/7998e2cc-740f-4ebb-ae1e-515331ee0083)

