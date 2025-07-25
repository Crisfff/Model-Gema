from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar el modelo Keras
modelo = load_model("gema_ai.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get("features", [])
    if not features:
        return jsonify({"error": "No features provided"}), 400

    # Preprocesamiento: convierte la lista a numpy array 2D, como espera Keras
    features_np = np.array([features])
    prediction = modelo.predict(features_np)
    # Si el modelo es de clasificación, puedes hacer argmax
    signal = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))

    response = {
        "signal": str(signal),
        "confianza": f"{confidence*100:.2f}%",
        "input_recibido": features,
        "explicacion": "Predicción usando tu modelo gema_ai.h5"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
