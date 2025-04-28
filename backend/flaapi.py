from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os

# 🔥 Import your models
from models import FakeNewsModel, TextCNNModel, LSTMModel  # (Create a 'models.py' file if needed)

# ⚡ Setup Flask app
app = Flask(__name__)
CORS(app)

# 🧠 Load models once
device = torch.device('cpu')

model_paths = {
    "bilstm": "best_fake_news_model.pt",
    "cnn": "textcnn_model.pt",
    "lstm": "lstm_model.pt"
}

models = {}

# 🛠️ Initialize and load all 3 models
def load_models():
    bilstm = FakeNewsModel()
    bilstm.load_state_dict(torch.load(model_paths["bilstm"], map_location=device))
    bilstm.eval()

    cnn = TextCNNModel()
    cnn.load_state_dict(torch.load(model_paths["cnn"], map_location=device))
    cnn.eval()

    lstm = LSTMModel()
    lstm.load_state_dict(torch.load(model_paths["lstm"], map_location=device))
    lstm.eval()

    models["bilstm"] = bilstm
    models["cnn"] = cnn
    models["lstm"] = lstm

load_models()

@app.route('/setup')
def setup():
    return jsonify({"message": "Setup complete, all models loaded."})

@app.route('/predict', methods=['POST'])
def predict_news():
    try:
        data = request.get_json()
        text = data.get('text', '')
        selected_model = data.get('model', 'bilstm')  # default to bilstm

        if not text.strip():
            return jsonify({"error": "No input text provided"}), 400

        if selected_model not in models:
            return jsonify({"error": "Invalid model choice"}), 400

        model = models[selected_model]

        # ➡️ Here you should vectorize text exactly as done during training
        # For now, let's assume simple dummy vectorization (fix as needed)

        input_vector = your_vectorizer.transform([text])  # You must load your TF-IDF or custom vectorizer
        input_tensor = torch.tensor(input_vector.toarray(), dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        result = {
            "prediction": "FAKE" if prediction.item() == 1 else "REAL",
            "confidence": float(confidence.item()),
            "fake_probability": float(probs[0][1].item()),
            "real_probability": float(probs[0][0].item())
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
