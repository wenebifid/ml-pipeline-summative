# ML Pipeline – Summative Project

## 📌 Project Overview

This project demonstrates an end‑to‑end **Machine Learning Pipeline** that covers data preparation, model training, API deployment, and system testing under load. The system exposes prediction endpoints via an API and includes a user‑facing interface for interaction. A flood request (stress) simulation was conducted to evaluate performance, stability, and scalability.

The project is designed to showcase:

* Clean ML pipeline structure
* Model retraining capability
* API‑based predictions
* Frontend interaction
* System behaviour under high request volume

---

## 🎥 Video Demo

A full walkthrough of the project (architecture, setup, training, API usage, and flood simulation results) is available here:

**YouTube Demo:** 

---

## 🌐 Application URLs

> Replace these with your deployed URLs if applicable.

* **API Base URL:** https://ml-pipeline-summative-1d62.onrender.com/docs

* **Frontend** https://ml-pipeline-summative-1d62.onrender.com

If running locally, URLs will be provided in the terminal during startup.

---

## 🗂️ Repository Structure (High Level)

```
ml-pipeline-summative/
│
├── data/                  # Training and testing datasets
│   ├── train/
│   └── test/
│
├── src/                   # Core ML logic
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   ├── utils.py
│   └── charts.py
│
├── api/                   # FastAPI application
│
├── streamlit_app.py       # Frontend interface
├── requirements.txt       # Dependencies
├── README.md
└── tests/                 # Load / flood simulation scripts
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ml-pipeline-summative.git
cd ml-pipeline-summative
```

---

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Prepare Dataset

Ensure the dataset is structured as follows:

```
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

> Each class is split **80% for training** and **20% for testing**.

---

### 5️⃣ Train or Retrain the Model

```bash
python src/model.py
```

This will:

* Load the training data
* Train the model
* Save the trained model for inference

---

### 6️⃣ Start the API Server

```bash
uvicorn api.main:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

Swagger Docs:

```
http://127.0.0.1:8000/docs
```

---

### 7️⃣ Run the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

---

## 🔁 Flood Request Simulation

A flood (stress) test was conducted to simulate a large number of concurrent prediction requests to the API.

### 🔧 Tool Used

* Custom Python script / load testing tool (e.g. asyncio / requests / locust)

### 📊 Simulation Results

<img width="2532" height="1175" alt="Screenshot (115)" src="https://github.com/user-attachments/assets/4db4096a-299a-455a-aa4e-581af27c4f8f" />


### ✅ Observations

* The API remained stable under heavy load
* No crashes or memory leaks observed
* Response times increased slightly but stayed within acceptable limits
* System successfully handled concurrent prediction requests

---

## 🧪 Key Features Demonstrated

* End‑to‑end ML workflow
* Model retraining support
* REST API integration
* Frontend interaction
* Load and stress testing
* Clear modular code structure

---

## 📌 Notes

* This project is intended for academic demonstration purposes
* All components can be deployed locally or on cloud platforms

---

## 👤 Author

**Name:** Oyinwenebi Fiderikumo


---



