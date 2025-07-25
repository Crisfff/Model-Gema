from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar el modelo
model = load_model("gema_ai.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get("features")
    if not features or not isinstance(features, list):
        return jsonify({"error": "Debes enviar 'features' como lista"}), 400

    # Ajusta para tu input_shape (debe ser [n_features])
    arr = np.array([features], dtype=np.float32)
    pred = model.predict(arr)[0][0]
    signal = "CALL" if pred > 0.5 else "PUT"
    confianza = f"{pred*100:.2f}%" if signal == "CALL" else f"{(1-pred)*100:.2f}%"

    return jsonify({
        "signal": signal,
        "confianza": confianza,
        "input_recibido": features,
        "explicacion": "Predicción generada por tu modelo IA entrenado para señales binarias."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
