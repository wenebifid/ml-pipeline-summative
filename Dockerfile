# Dockerfile for Streamlit app
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential ffmpeg git wget libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
ENV PYTHONPATH=/app
EXPOSE 8501

CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
