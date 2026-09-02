Ghana Health AI: Hybrid Diagnostic System

An advanced, multi-modal Artificial Intelligence diagnostic platform designed to assist healthcare professionals with rapid triage and clinical decision support. This system evaluates medical imagery and patient tabular data to predict disease presence, outputting clinical-grade explainability metrics (Grad-CAM and SHAP) alongside its predictions.

## Architecture

The project uses a microservices architecture:

ML Service (FastAPI): A high-performance Python backend dedicated entirely to running PyTorch and XGBoost inference, generating heatmaps, and computing triage urgency.

Web Portal (Django - Upcoming): A secure clinical frontend for doctors to upload patient files, manage records, and view AI-generated diagnostic reports.

Diagnostic Modules

Currently, the ML Service supports three distinct diagnostic pipelines:

## Project structure

```
ml_service\
├── main.py                              FastAPI app entry point
├── requirements.txt
├── Dockerfile
├── .env.example
│
├── core\
│   ├── config.py                        All env variables (import `settings`)
│   ├── logger.py                        Loguru logger
│   └── triage.py                        Triage flag computation
│
├── api\
│   ├── routes\
│   │   ├── predict.py                   POST /api/ml/predict/image  (pneumonia)
│   │   ├── malaria.py                   POST /api/ml/predict/malaria
│   │   ├── diabetes.py                  POST /api/ml/predict/diabetes
│   │   └── health.py                    GET  /api/ml/health
│   └── schemas\
│       └── prediction.py                PredictionResponse Pydantic model
│
├── models\
│   ├── pneumonia\
│   │   ├── model.py                     DenseNet-121 architecture
│   │   └── inference.py                 PneumoniaPredictor
│   ├── malaria\
│   │   ├── model.py                     EfficientNet-B0 architecture
│   │   └── inference.py                 MalariaPredictor
│   └── diabetes\
│       ├── model.py                     XGBoost + RandomForest ensemble
│       └── inference.py                 DiabetesPredictor
│
├── preprocessing\
│   ├── image_transforms.py              Chest X-ray Albumentations pipeline
│   ├── malaria_transforms.py            Blood smear Albumentations pipeline
│   └── tabular_transforms.py            Tabular feature engineering + validation
│
├── training\
│   ├── pneumonia\
│   │   ├── train.py                     DenseNet-121 training loop
│   │   └── dataset.py                   Chest X-ray PyTorch Dataset
│   ├── malaria\
│   │   ├── train.py                     EfficientNet-B0 training loop
│   │   └── dataset.py                   Cell image PyTorch Dataset
│   └── diabetes\
│       └── train.py                     XGBoost ensemble training + Optuna tuning
│
├── explainability\
│   ├── gradcam.py                       Grad-CAM for pneumonia (DenseNet-121)
│   ├── malaria_gradcam.py               Grad-CAM for malaria (EfficientNet-B0)
│   └── shap_explainer.py                SHAP for all tabular models
│
├── tests\
│   └── unit\
│       ├── test_pneumonia_inference.py
│       ├── test_malaria_inference.py
│       └── test_diabetes_inference.py
│
├── data\
│   ├── raw\
│   │   ├── pneumonia\                   Kaggle Chest X-Ray dataset
│   │   ├── malaria\                     NIH Malaria Cell Images
│   │   └── diabetes\                   PIMA Diabetes CSV
│   └── processed\
│
├── model_registry\
│   ├── pneumonia\best_model.pth
│   ├── malaria\best_model.pth
│   └── diabetes\best_model.joblib
│
├── media\
│   ├── gradcam\                         Generated heatmap PNGs
│   └── uploads\
│
├── scripts\
│   ├── download_pneumonia_dataset.py
│   ├── download_malaria_dataset.py
│   └── download_diabetes_dataset.py
│
└── mlruns\                              MLflow experiment logs
```

## Django integration

Django calls the ML service over internal HTTP. The ML service is never exposed to the internet.

### Endpoints Django uses

| Method | Endpoint | Input | Use case |
|---|---|---|---|
| POST | `/api/ml/predict/image` | Image file | Pneumonia / chest X-ray |
| POST | `/api/ml/predict/malaria` | Image file | Malaria blood smear |
| POST | `/api/ml/predict/diabetes` | JSON body | Type 2 Diabetes |
| GET | `/api/ml/health` | — | Service health check |

### Sample response (all endpoints return the same shape)
```json
{
  "prediction_id": "a3f2c1d4...",
  "domain": "diabetes_metabolic",
  "predicted_class": "DIABETIC",
  "confidence": 0.883,
  "confidence_pct": "88.3%",
  "differentials": ["NON-DIABETIC (11.7%)"],
  "triage": "high",
  "recommended_action": "Seek immediate medical attention...",
  "explainability": {
    "type": "shap",
    "heatmap_url": null,
    "top_features": [
      {"feature": "Glucose", "value": 168.0, "shap": 0.312, "direction": "positive", "rank": 1},
      {"feature": "BMI",     "value": 39.5,  "shap": 0.187, "direction": "positive", "rank": 2},
      {"feature": "Age",     "value": 52.0,  "shap": 0.143, "direction": "positive", "rank": 3}
    ]
  },
  "model_version": "1.0.0",
  "disclaimer": "This prediction is AI-assisted..."
}
```

1. Pneumonia Detection (Computer Vision)

Input: Chest X-Ray (CXR) image.

Model: Deep Convolutional Neural Network (PyTorch).

Explainability: Grad-CAM Heatmaps highlight the specific pulmonary regions (opacities/infiltrates) driving the AI's prediction.

2. Malaria Detection (Computer Vision)

Input: Thin blood smear microscopic image.

Model: Deep Convolutional Neural Network (PyTorch).

Explainability: Grad-CAM Heatmaps isolate the exact cellular location of the parasite.

3. Type 2 Diabetes Prediction (Tabular / Machine Learning)

Input: 8 Clinical Vitals (Glucose, BMI, Age, Blood Pressure, Insulin, etc.).

Model: XGBoost Classifier optimized with Optuna and SMOTE for class-imbalance.

Explainability: SHAP Values provide a weighted breakdown showing exactly which patient vitals pushed the diagnosis toward or away from a Diabetic prediction.

Quickstart: Running the ML Service Locally

Prerequisites

Python 3.11

Pip

1. Setup Environment

Navigate to the ML service directory, create a virtual environment, and install dependencies:

cd ml_service
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt


2. Start the FastAPI Server

Ensure you are in the root project directory (one level above ml_service if that's where your main.py is located) and run:

uvicorn main:app --host 0.0.0.0 --port 8001 --reload


The ML service will now be active at http://localhost:8001.
You can view the interactive API documentation (Swagger UI) at http://localhost:8001/docs.

Testing the API via Command Line

You can send test payloads to the API using curl.

Test the Diabetes Tabular Model:

curl -X POST http://localhost:8001/api/ml/predict/diabetes ^
  -H "Content-Type: application/json" ^
  -d "{\"pregnancies\":5,\"glucose\":168,\"blood_pressure\":90,\"skin_thickness\":35,\"insulin\":0,\"bmi\":39.5,\"diabetes_pedigree_function\":1.1,\"age\":52}"


Test the Malaria Image Model (Absolute Path Example):

curl -X POST http://localhost:8001/api/ml/predict/malaria -F "file=@C:\Path\To\Your\BloodSmear.png"


Docker Deployment

To containerize the ML service for production:

Build the image:

docker build -t ghana-health-ai-ml .


Run the container:

docker run -p 8001:8001 ghana-health-ai-ml


!!! Clinical Disclaimer

This software is a research prototype intended for clinical decision support. It does not replace the judgement of a qualified healthcare provider. Always consult a doctor for definitive medical diagnoses.