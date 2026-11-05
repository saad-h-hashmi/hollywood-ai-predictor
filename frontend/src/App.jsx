// src/App.jsx
import { useState } from 'react';
import './App.css';

function App() {
  const [budget, setBudget] = useState('');
  const [marketing, setMarketing] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    
    try {
      // Talk to your Python Flask server
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          budget: Number(budget), 
          marketing: Number(marketing) 
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setPrediction(data.predicted_revenue);
      } else {
        alert("Error from AI server: " + data.message);
      }
    } catch {
      alert("Failed to connect to the backend. Is your Python Flask server running on port 5000?");
    }
    
    setLoading(false);
  };

  // Helper function to format money nicely (e.g. $150,000,000)
  const formatMoney = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(amount);
  };

  return (
    <div className="App">
      <header className="header">
        <h1>Hollywood AI Predictor</h1>
        <p>Enter the film's budget to forecast its global box office revenue.</p>
      </header>

      <main className="main-content">
        <form onSubmit={handlePredict} className="prediction-form">
          <div className="input-group">
            <label>Production Budget ($)</label>
            <input 
              type="number" 
              value={budget} 
              onChange={(e) => setBudget(e.target.value)} 
              placeholder="e.g., 200000000"
              required 
            />
          </div>

          <div className="input-group">
            <label>Marketing Spend ($)</label>
            <input 
              type="number" 
              value={marketing} 
              onChange={(e) => setMarketing(e.target.value)} 
              placeholder="e.g., 85000000"
              required 
            />
          </div>

          <button type="submit" disabled={loading}>
            {loading ? 'Analyzing Data...' : 'Predict Box Office'}
          </button>
        </form>

        {prediction !== null && (
          <div className="result-card">
            <h2>Predicted Box Office</h2>
            <div className="revenue-number">{formatMoney(prediction)}</div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;