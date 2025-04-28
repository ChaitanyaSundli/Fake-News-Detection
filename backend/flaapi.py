from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os

from fakenewscode import (
    TextVectorizer, BiLSTMAttention, TextCNN, LSTMClassifier,
    preprocess_text, contains_negation
)

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global objects
vectorizer = None
model = None
current_model_name = None

@app.route("/setup", methods=["GET"])
def setup():
    global vectorizer, model, current_model_name
    model_name = request.args.get("model", "bilstm.pt")

    try:
        # Step 1: Load vectorizer
        vectorizer = TextVectorizer()

        word_index_path = os.path.join("models", "word_index.pt")
        if os.path.exists(word_index_path):
            word_index = torch.load(word_index_path)
            vectorizer.word_index = word_index
            vectorizer.vocab_size = len(word_index)
            print(f"✅ Loaded vocabulary with {vectorizer.vocab_size} words.")
        else:
            return jsonify({"error": "Word index file not found!"}), 500

        # Step 2: Create correct model based on selection
        model_path = os.path.join("models", model_name)
        if os.path.exists(model_path):
            vocab_size = vectorizer.vocab_size
            pad_idx = 0

            if model_name == "bilstm.pt":
                # BiLSTM hyperparameters
                model = BiLSTMAttention(
                    vocab_size=vocab_size,
                    embedding_dim=100,
                    hidden_dim=128,
                    output_dim=2,
                    n_layers=2,
                    dropout=0.5,
                    pad_idx=pad_idx
                ).to(device)

            elif model_name == "cnn.pt":
                # CNN hyperparameters
                model = TextCNN(
                    vocab_size=vocab_size,
                    embedding_dim=100,
                    n_filters=100,
                    filter_sizes=[2, 3, 4, 5],
                    output_dim=2,
                    dropout=0.5,
                    pad_idx=pad_idx
                ).to(device)

            elif model_name == "lstm.pt":
                # LSTM hyperparameters
                model = LSTMClassifier(
                    vocab_size=vocab_size,
                    embedding_dim=100,
                    hidden_dim=128,
                    output_dim=2,
                    n_layers=2,
                    bidirectional=True,
                    dropout=0.5,
                    pad_idx=pad_idx
                ).to(device)

            else:
                return jsonify({"error": "Unsupported model selected."}), 400

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            current_model_name = model_name
            print(f"✅ Model '{model_name}' loaded successfully.")

        else:
            return jsonify({"error": f"Model file '{model_name}' not found."}), 404

        return jsonify({"status": "Model setup complete", "model": model_name})

    except Exception as e:
        print(f"❌ Setup error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    global vectorizer, model
    try:
        if model is None or vectorizer is None:
            return jsonify({"error": "Model not initialized. Please run /setup first."}), 500

        data = request.get_json()
        text = data.get("text", "")
        if len(text.strip()) < 10:
            return jsonify({"error": "Text is too short."}), 400

        processed_text = preprocess_text(text)
        negation_info = contains_negation(text)
        sequence = vectorizer.transform([processed_text])
        input_tensor = torch.tensor(sequence).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        fake_prob = probs[0, 1].item()
        real_prob = probs[0, 0].item()

        # Adjust for negations
        if negation_info['has_negation']:
            fake_prob, real_prob = real_prob, fake_prob

        prediction = 'FAKE' if fake_prob >= 0.5 else 'REAL'

        return jsonify({
            "prediction": prediction,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": max(fake_prob, real_prob),
            "negation_detected": negation_info['has_negation'],
            "negation_count": negation_info['negation_count'],
            "processed_text": processed_text
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
