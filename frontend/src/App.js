import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import './App.css';
import { Loader2 } from 'lucide-react';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

export default function App() {
  const [step, setStep] = useState(0);
  const [selectedModel, setSelectedModel] = useState('bilstm');
  const [training, setTraining] = useState(false);
  const [trainingImages, setTrainingImages] = useState([]);
  const [selectedChart, setSelectedChart] = useState('roc_curve');
  const [logLines, setLogLines] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    fetch('https://fake-news-detection-fcr3.onrender.com/setup')
      .then((res) => res.json())
      .then((data) => {
        console.log('Setup complete:', data);
        setStep(1);
      })
      .catch((err) => console.error('Setup error:', err));
  }, []);

  const handleTrain = async () => {
  setTraining(true);
  setStep(2);

  try {
    const response = await fetch('https://fake-news-detection-fcr3.onrender.com/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selectedModel })
    });

    if (!response.ok) {
      throw new Error('Training failed');
    }

    const data = await response.json();
    console.log('Training complete:', data);

    const metricsRes = await fetch('https://fake-news-detection-fcr3.onrender.com/metrics');
    const metrics = await metricsRes.json();
    setTrainingImages(metrics.images);

    // 🎯 After training, GO TO Step 3 (charts)
    setStep(3);

  } catch (error) {
    console.error('Training error:', error);
    setTraining(false);
  }
};


  const handleChartChange = (e) => {
    setSelectedChart(e.target.value);
  };

  return (
    <div className="container">
      <motion.div className="branding">
        <img src="/logo.png" alt="Logo" className="logo" />
        <h1 className="brand-name">NewsVeritas</h1>
      </motion.div>

      <motion.div className="card">
        <motion.h1 className="title">Fake News Detection</motion.h1>

        {step === 0 && (
          <motion.div className="status">
            <Loader2 className="spinner" />
            Loading dataset and initializing backend...
          </motion.div>
        )}

        {step === 1 && (
          <motion.div className="step">
            <h2>Step 2: Choose Model</h2>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="select"
            >
              <option value="bilstm">BiLSTM + Attention</option>
              <option value="cnn">TextCNN</option>
              <option value="lstm">Regular LSTM</option>
            </select>

            <button onClick={handleTrain} className="btn-green" style={{ marginTop: '1rem' }}>
              Start Training
            </button>
          </motion.div>
        )}

        {step === 2 && (
          <motion.div className="step">
            <p>Training your <strong>{selectedModel}</strong> model...</p>
            <Loader2 className="spinner big" />
          </motion.div>
        )}

        {step === 3 && (
          <motion.div className="step">
            <h2>📊 Training Visualizations</h2>
            <div className="chart-selector">
              <label htmlFor="chart-select">Select Chart:</label>
              <select id="chart-select" value={selectedChart} onChange={handleChartChange} className="select">
                <option value="roc_curve">ROC Curve</option>
                {trainingImages.map((img) => {
                  const chartName = img
                    .split('.')[0]
                    .replace(/_/g, ' ')
                    .replace(/\b\w/g, (char) => char.toUpperCase());
                  return (
                    <option key={img} value={img}>{chartName}</option>
                  );
                })}
              </select>
            </div>

            <div className="image-grid">
              {trainingImages.map((img, idx) => (
                img.includes(selectedChart) && (
                  <img
                    key={idx}
                    src={`https://fake-news-detection-fcr3.onrender.com/static/${img}`}
            {step === 4 && (
  <motion.div className="step">
    <h2>🧠 Try a Prediction</h2>
    <textarea
      rows="5"
      className="select"
      value={userInput}
      onChange={(e) => setUserInput(e.target.value)}
      placeholder="Enter a news article..."
    />

    <div className="example-buttons">
      <button onClick={() => setUserInput("Breaking: The president has signed a new executive order...")}>Example 1</button>
      <button onClick={() => setUserInput("NASA confirms Earth has two moons!")}>Example 2</button>
    </div>

    <button
      className="btn-green"
      onClick={async () => {
        try {
          const res = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: userInput })
          });
          const data = await res.json();

          if (res.ok) {
            setPredictionResult(data);
          } else {
            alert(data.error || "Prediction failed.");
            setPredictionResult(null);
          }
        } catch (err) {
          alert("Failed to connect to the backend.");
          console.error(err);
        }
      }}
    >
      🔍 Predict
    </button>

    {predictionResult && (
      <motion.div
        className="result-card"
        style={{ backgroundColor: predictionResult.confidence > 0.85 ? '#dcfce7' : predictionResult.confidence > 0.6 ? '#fef9c3' : '#fee2e2' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <h3 className={predictionResult.prediction === 'FAKE' ? 'text-red' : 'text-green'}>
          {predictionResult.prediction === 'FAKE' ? '🚨 FAKE NEWS DETECTED' : '✅ REAL NEWS'}
        </h3>
        <p><strong>Confidence:</strong> {(predictionResult.confidence * 100).toFixed(2)}%</p>
        <p className="subtext">
          (Fake: {(predictionResult.fake_probability * 100).toFixed(2)}% | Real: {(predictionResult.real_probability * 100).toFixed(2)}%)
        </p>
        {predictionResult.negation_detected && (
          <p className="subtext">⚠️ Detected <strong>{predictionResult.negation_count}</strong> negation word(s).</p>
        )}
        {predictionResult.processed_text && (
          <p className="subtext">🔍 Processed Text: {predictionResult.processed_text}</p>
        )}
      </motion.div>
    )}
    
    <button className="btn-purple" onClick={handleDownload}>
      🎓 Download Verification Certificate
    </button>
    
    <button
      className="btn-green"
      onClick={() => {
        setUserInput('');
        setPredictionResult(null);
      }}
    >
      🧹 Clear Result
    </button>
    
    <button
      className="btn-purple"
      onClick={() => {
        setStep(1);
        setUserInput('');
        setPredictionResult(null);
      }}
    >
      🔁 Back to Start
    </button>

  </motion.div>
)}

                    alt={img}
                    className="chart"
                  />
                )
              ))}
            </div>

            <button onClick={() => { setTrainingImages([]); setStep(1); setLogLines([]); }} className="btn-purple">
              🔄 Reset Wizard
            </button>

            <button onClick={() => setStep(4)} className="btn-green">
              🧠 Try a Prediction
            </button>
          </motion.div>
        )}

        {/* Step 4 Prediction UI (no change needed) */}
        {/* Your prediction and certificate code remains the same */}
      </motion.div>
    </div>
  );
}
