from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import subprocess
import os
import time
import json

# 🛠️ Import your local code
from fakenewscode import load_vectorizer, load_model, predict

app = Flask(__name__)
CORS(app)

# 🔥 Load vectorizer + model ONCE globally
vectorizer = load_vectorizer('vectorizer.json')
model = load_model('bilstm.h5')  # Default model for prediction

@app.route('/setup')
def setup():
    return jsonify({"message": "Setup complete, model loaded."})

@app.route('/stream-training')
def stream_training():
    model_choice = request.args.get('model', 'bilstm')

    def generate():
        process = subprocess.Popen(
            ["python", "-u", "fakenewscode.py", "--train", "news.csv", model_choice],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in iter(process.stdout.readline, ''):
            yield f"data: {line}\n\n"
            time.sleep(0.1)

        process.stdout.close()

    return Response(generate(), mimetype='text/event-stream')

@app.route('/metrics')
def metrics():
    files = os.listdir('metrics') if os.path.exists('metrics') else []
    return jsonify({"images": files})

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text.strip():
            return jsonify({"error": "No input text provided"}), 400

        result = predict(text, model, vectorizer)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ❌ REMOVED: /train route (no instant training)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
