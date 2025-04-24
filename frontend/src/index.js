import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';  // Make sure this import is correct
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />  {/* Make sure you're rendering FakeNewsApp here */}
  </React.StrictMode>
);

reportWebVitals();
