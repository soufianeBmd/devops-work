from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import librosa
import numpy as np
import io
import base64

app = Flask(__name__)

model_vgg = tf.keras.models.load_model('best_model.h5')

#Function to extract features from an audio file
def extract_mel_spectrogram(audio_data):
    try:
        audio, sr = librosa.load(audio_data, res_type='kaiser_fast')
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram
    except Exception as e:
        raise RuntimeError(f"Error extracting mel spectrogram from audio data: {e}")

#Function to preprocess mel spectrogram for VGG16 model
def preprocess_mel_spectrogram(mel_spectrogram):
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    mel_spectrogram = mel_spectrogram / 255.0  # Normalize to [0, 1]
    return mel_spectrogram

def get_genre_from_label(label):
    print(label)
    genres = ["metal","hiphop","disco","blues","rock","classical","country","pop","reggae","jazz"]
    return genres[label]

#decode and predict function
@app.route('/predict_vgg19', methods=['POST'])
def decodeAndPredict():
    try:
        #print("function in the upload point work fine ")
        data = request.get_json()
        encoded_file = data.get("encoded_file")

        #decoding the file and saving it
        with open("./vgg19.wav","wb") as decoded_svm_file:
          decoded_svm_file.write(base64.b64decode(encoded_file))

        # Extract mel spectrogram from the audio data
        mel_spectrogram = extract_mel_spectrogram("./vgg19.wav")

        # Preprocess mel spectrogram for VGG16 model
        input_data = preprocess_mel_spectrogram(mel_spectrogram)
        print("Input shape:", input_data.shape)
        print("Input data type:", input_data.dtype)

        # Make a prediction using the VGG16 model
        prediction = model_vgg.predict(input_data)

        # Get the predicted genre
        predicted_genre = get_genre_from_label(np.argmax(prediction))

        # Return the result as JSON
        result = {'prediction': predicted_genre}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__" :
  app.run(debug=True,host="0.0.0.0",port=5002)
