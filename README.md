# Naive Bayes Classifier Microservice Project

## Overview
This project implements a microservice-based system for training and serving a Naive Bayes classifier using FastAPI. The system is split into two main components:

- **trainer**: Responsible for data processing, model training, and evaluation.
- **predictor**: Provides an API for classifying new records using the trained model.

Both services are containerized using Docker for easy deployment.

---

## Project Structure

```
.
├── trainer/
│   ├── server.py                # FastAPI server for training and evaluation
│   ├── naive_bayes_classifier.py# Naive Bayes model implementation
│   ├── evaluator.py             # Model evaluation logic
│   ├── data_procesor.py         # Data cleaning and splitting
│   ├── data_loader.py           # Data loading utilities
│   ├── data.csv                 # Example dataset
│   ├── Dockerfile               # Dockerfile for trainer service
│   └── requirements.txt         # Python dependencies
├── predictor/
│   ├── predictor_server.py      # FastAPI server for predictions
│   ├── predictor_model.py       # Prediction logic
│   ├── Dockerfile               # Dockerfile for predictor service
│   └── requirements.txt         # Python dependencies
└── README.md                    # Project documentation
```

---

## How It Works

1. **trainer** loads and cleans the data, splits it into train/test sets, trains a Naive Bayes model, and exposes endpoints to retrieve the model and evaluation results.
2. **predictor** fetches the trained model from the trainer service on startup and exposes an endpoint to classify new records.

---

## Setup & Usage

### Prerequisites
- Docker
- (Optional) Python 3.10+ if running locally without Docker

### Running with Docker Compose (Recommended)
Create a `docker-compose.yml` file (not included here) to orchestrate both services. Example service names:
- `trainer` (exposes port 8000)
- `predictor` (exposes port 8001)

### Manual Docker Run
From the project root:

```sh
# Build and run the trainer service
cd trainer
docker build -t trainer-service .
docker run -p 8000:8000 trainer-service

# In a new terminal, build and run the predictor service
cd ../predictor
docker build -t predictor-service .
docker run -p 8001:8001 predictor-service
```

### Local Development (without Docker)
Install dependencies for each service:

```sh
cd trainer
pip install -r requirements.txt
uvicorn server:app --reload --port 8000

cd ../predictor
pip install -r requirements.txt
uvicorn predictor_server:app --reload --port 8001
```

---

## API Endpoints

### Trainer Service (`localhost:8000`)
- `GET /get_model/` — Retrieve the trained model as JSON
- (Startup) — Trains and evaluates the model, prints results to console

### Predictor Service (`localhost:8001`)
- `POST /classify_record/` — Classify a record. Example body:
  ```json
  {
    "feature1": "value1",
    "feature2": "value2",
    ...
  }
  ```
  Response:
  ```json
  { "prediction": "class_name" }
  ```

---

## Customization
- Replace `trainer/data.csv` with your own dataset (last column should be the target/class).
- Adjust data processing, model, or API logic as needed in the respective Python files.

---

## License
This project is provided for educational purposes. Modify and use as needed. 