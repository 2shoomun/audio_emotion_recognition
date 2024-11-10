# Speech Emotion Recognition

This project is a Speech Emotion Recognition (SER) system that identifies emotions such as "calm," "happy," "fearful," and "disgust" from audio recordings. It uses an MLP (Multi-Layer Perceptron) classifier with features extracted from audio files using `librosa`. The project includes two main scripts:

- `extract.py`: For feature extraction, model training, and saving the trained model.
- `record.py`: For recording new audio, feature extraction, and emotion prediction using the trained model.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/2shoomun/audio_emotion_recognition.git
    cd audio_emotion_recognition
    ```

2. **Install dependencies**:
    This project requires the following Python libraries:
    - `librosa`
    - `numpy`
    - `soundfile`
    - `scikit-learn`
    - `sounddevice`
    - `scipy`

    Install them using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Dataset**:
    - Download the RAVDESS dataset from [here](https://zenodo.org/record/1188976).
    - Place the dataset in the `speech-emotion-recognition-ravdess-data` folder and ensure the structure follows:
      ```
      speech-emotion-recognition-ravdess-data/
      ├── Actor_01
      ├── Actor_02
      ├── ...
      ```

## Project Structure

- `extract.py`: Extracts audio features, trains the model, and saves it to disk.
- `record.py`: Records audio from the microphone, extracts features, and predicts the emotion using the trained model.
- `scaler.pkl`: Saves the scaler for normalizing features.
- `emotion_model.pkl`: Saves the trained MLP model.

## Usage

### Training the Model

1. **Run `extract.py`** to train the model and save the `emotion_model.pkl` and `scaler.pkl` files:
    ```bash
    python extract.py
    ```
    - This script loads audio data, extracts features (MFCC, chroma, and mel), trains an MLP classifier, and saves the model and scaler for future predictions.
    - After training, it outputs the model’s accuracy and cross-validation scores.

### Predicting Emotions

1. **Run `record.py`** to record an audio sample, save it, and predict the emotion:
    ```bash
    python record.py
    ```
    - The script will record 5 seconds of audio, save it as `my_audio.wav`, and then predict the emotion based on the trained model.
    - The predicted emotion will be displayed in the console.

**Note**: You can modify the recording duration and sample rate by changing the parameters in the `record_audio()` function.

## Acknowledgements

- The RAVDESS dataset used in this project can be found [here](https://zenodo.org/record/1188976).
- This project uses `librosa` for feature extraction, `scikit-learn` for machine learning, and `sounddevice` for audio recording.
