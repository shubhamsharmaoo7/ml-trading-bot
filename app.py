from flask import Flask, request, jsonify
import joblib, numpy as np

app = Flask(__name__)
model = joblib.load('xgboost_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    proba = model.predict_proba(features)[0,1]
    pred = model.predict(features)[0]
    return jsonify({'prediction': int(pred), 'confidence': float(proba)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
