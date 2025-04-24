from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
import os
import subprocess

from fakenewscode import (
    setup_and_prepare, train_model_by_choice, predict_text,
    TextVectorizer, BiLSTMAttention, preprocess_text, contains_negation
)

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables to hold vectorizer and model
vectorizer = None
model = None

@app.route("/setup", methods=["GET"])
def setup():
    global vectorizer, model
    try:
        vectorizer, model = setup_and_prepare("news.csv", device)

        # Load the trained model weights
        model_path = "best_fake_news_model.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("✅ Model weights loaded successfully.")
        else:
            print("⚠️ Trained model not found. Run training first.")

        return jsonify({"status": "setup complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    return jsonify({"error": "Use /stream-training for live updates."}), 400

@app.route("/stream-training")
def stream_training():
    model = request.args.get("model", "bilstm")
    model_arg = {
        "bilstm": "1",
        "cnn": "2",
        "lstm": "3"
    }.get(model, "1")

    def generate_logs():
        process = subprocess.Popen(
            ["python", "-u", "fakenewscode.py", "--train", "news.csv", model_arg],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        for line in process.stdout:
            yield f"data: {line}\n\n"
        process.stdout.close()
        process.wait()

    return Response(generate_logs(), mimetype="text/event-stream")

@app.route("/predict", methods=["POST"])
def predict():
    global model, vectorizer
    try:
        if model is None or vectorizer is None:
            return jsonify({"error": "Model or vectorizer not initialized. Please run /setup first."}), 500

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
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        metrics_files = [
            "training_metrics.png",
            "confusion_matrix.png",
            "roc_curve.png",
            "precision_recall_curve.png",
            "score_distribution.png",
            "cost_benefit_analysis.png"
        ]
        return jsonify({"images": metrics_files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    setup()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
