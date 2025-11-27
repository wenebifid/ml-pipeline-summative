import React, {useState, useEffect} from 'react';
import axios from 'axios';
import './index.css';

function App(){
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [retrainMsg, setRetrainMsg] = useState("");

  useEffect(()=>{
    fetch('/api/health').catch(()=>{}); // placeholder if proxy not set
  },[]);

  const handleFile = (e) => {
    setFile(e.target.files[0]);
  }

  const submitPredict = async () => {
    if(!file) return alert('Choose a file first');
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await axios.post('/predict', form, { headers: {'Content-Type': 'multipart/form-data'}});
      setPredictions(res.data.predictions);
    } catch (err) {
      alert('Prediction error: ' + err.message);
    }
  }

  const triggerRetrain = async () => {
    try {
      setRetrainMsg('Retraining...');
      const res = await axios.post('/trigger-retrain', null, { params: { epochs: 3 }});
      setRetrainMsg('Retrain done: ' + JSON.stringify(res.data));
    } catch (err) {
      setRetrainMsg('Retrain failed: ' + err.message);
    }
  }

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">EuroSAT Image Classifier</h1>
      <div className="mb-6">
        <input type="file" onChange={handleFile} />
        <button className="ml-2 p-2 bg-blue-600 text-white rounded" onClick={submitPredict}>Predict</button>
      </div>
      <div className="mb-6">
        <button className="p-2 bg-green-600 text-white rounded" onClick={triggerRetrain}>Trigger Retrain (3 epochs)</button>
        <div className="mt-2">{retrainMsg}</div>
      </div>
      <div>
        <h2 className="text-xl font-semibold">Predictions</h2>
        {predictions ? (
          <ul>
            {predictions.map((p, i) => <li key={i}>{p.class}: {(p.probability*100).toFixed(2)}%</li>)}
          </ul>
        ) : <div>No predictions yet</div>}
      </div>
    </div>
  );
}

export default App;
