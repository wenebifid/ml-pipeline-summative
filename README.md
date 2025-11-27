# EuroSAT Image Classification Pipeline (Submission-ready)

This repository is a complete scaffold for the end-to-end ML pipeline using the **EuroSAT (RGB)** dataset.
It includes preprocessing, model training (transfer learning), evaluation notebook, FastAPI server for prediction
and retraining, Docker files, Locust load-test script, and monitoring hooks.

See `notebook/project_eurosat.ipynb` for a runnable walkthrough (exploration, training, evaluation).

## Quick structure

```
project_name/
├── README.md
├── notebook/
│   └── project_eurosat.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   ├── api.py
│   └── utils.py
├── data/
│   ├── train/   # place class folders here
│   └── test/
├── models/
│   └── model_latest.h5  # created after training
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── locust/
│   └── locustfile.py
└── requirements.txt
```

## How to use

1. Download the EuroSAT RGB dataset (links provided in the assignment). Place images into `data/train/<class>` and `data/test/<class>` or use the notebook's TFDS loader.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the notebook to train and evaluate the model or run:
   ```
   python -m src.api
   ```
   to start the FastAPI server (or use Docker).
4. Use `/predict` endpoint to upload a single image file.
5. Use `/trigger-retrain` to retrain using data in `data/train` and `data/test`.

## Notes

- This scaffold uses MobileNetV2 transfer learning and saves the model to `models/model_latest.h5`.
- The dataset link: see assignment message or `DATASET_LINKS.txt`.



## React UI
A minimal React app is in `ui/react-app`. Start the backend and run the React dev server (set proxy to backend or configure CORS).

## Streamlit Dashboard Added
Run `streamlit run app_streamlit.py` to start the dashboard locally.
Docker: `docker build -t eurosat-streamlit . && docker run -p 8501:8501 eurosat-streamlit`
