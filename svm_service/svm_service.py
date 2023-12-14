from flask import  Flask, request,jsonify
import librosa
import base64
import joblib
app = Flask(__name__)

#importing the model
model = joblib.load("svm_model.joblib")

#extract feature function the same used in the training
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        raise RuntimeError(f"Error extracting features from {file_path}: {e}")

def get_genre_from_label(label):
    print(label)
    genres = ["metal","hiphop","disco","blues","rock","classical","country","pop","reggae","jazz"]
    return genres[label]


@app.route('/predict_svm',methods=['POST'])
def decodeAndPredict():
    #print("function in the upload point work fine ")
    data = request.get_json()
    encoded_file = data.get("encoded_file")
    #decoding the file and saving it
    with open("./temp.wav","wb") as decoded_svm_file:
      decoded_svm_file.write(base64.b64decode(encoded_file))
    try:
        # Extract features from the uploaded file
        features = extract_features("./temp.wav")

        # Make a prediction
        prediction = model.predict([features])

        # Get the predicted genre
        predicted_genre = get_genre_from_label(prediction[0])

        # Return the result as JSON
        result = {'prediction': predicted_genre}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__" :
  app.run(debug=True,host="0.0.0.0",port=5003)

