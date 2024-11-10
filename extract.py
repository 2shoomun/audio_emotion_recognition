import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=60).T, axis=0)
            delta_mfccs = np.mean(librosa.feature.delta(mfccs).T, axis=0)
            delta2_mfccs = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)
            result = np.hstack((result, mfccs, delta_mfccs, delta2_mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result

# Emotions in the RAVDESS dataset
emotions = {
  '01': 'neutral',
  '02': 'calm',
  '03': 'happy',
  '04': 'sad',
  '05': 'angry',
  '06': 'fearful',
  '07': 'disgust',
  '08': 'surprised'
}
# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("C:\\Users\\anshu\\Downloads\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"): # Modify this path to your dataset location
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# Normalize Features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize the Multi Layer Perceptron Classifier with modified architecture
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300, 150), learning_rate='adaptive', max_iter=1000)

# Train the model
model.fit(x_train, y_train)

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Optionally, implement cross-validation to evaluate model performance
cross_val_scores = cross_val_score(model, np.vstack((x_train, x_test)), y_train + y_test, cv=5)
print("Cross-Validation Accuracy Scores:", cross_val_scores)

import pickle
# After training the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Assuming 'model' is your trained MLPClassifier
with open('emotion_model.pkl', 'wb') as file:
    pickle.dump(model, file)
