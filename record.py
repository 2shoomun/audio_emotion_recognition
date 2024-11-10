import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
from extract import extract_feature
import pickle

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the trained model
with open('emotion_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Record Audio
def record_audio(duration=5, sample_rate=22050):
    print("Recording for {} seconds...".format(duration))
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    return recording, sample_rate

# Save Recording
def save_recording(recording, sample_rate, file_name='my_audio.wav'):
    write(file_name, sample_rate, recording)  # Save as WAV file 

# Predict Emotion
def predict_emotion(file_name):
    feature = extract_feature(file_name, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1)  # Reshape for the model input
    feature = scaler.transform(feature)  # Normalize features
    emotion_prediction = model.predict(feature)
    return emotion_prediction

# Record your own audio
duration = 5  # Duration of recording in seconds
my_recording, sr = record_audio(duration)
save_recording(my_recording, sr)

# Predict the emotion of the recorded audio
emotion = predict_emotion('my_audio.wav')
print("Predicted Emotion:", emotion)
