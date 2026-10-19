# End-to-End MLOps for Heart Disease Classification
**Course:** AIMLCZG523 — Machine Learning Operations  
**Assignment:** 01  
**Student:** [Your name and ID]  
**Date:** [Submission date]

> Replace bracketed fields and include evidence from your actual runs. Do not claim cloud deployment, passing tests, metrics, or screenshots that you have not generated.

## 1. Abstract
This project implements an educational binary classification workflow using the processed Cleveland subset of the UCI Heart Disease dataset. The workflow covers data acquisition, exploratory analysis, preprocessing, model comparison, experiment tracking, automated tests, containerization, API serving, and deployment configuration. The model is not clinically validated and is not intended for diagnosis.

## 2. Problem statement and objectives
The goal is to demonstrate reproducible ML engineering rather than build a medical device. The target is transformed to binary: original target 0 becomes absence; values 1–4 become presence. Objectives include repeatable data preparation, fair model evaluation, experiment traceability, tested inference, and a deployable service.

## 3. Dataset and acquisition
Source: UCI Machine Learning Repository, Heart Disease dataset: https://archive.ics.uci.edu/dataset/45/heart+disease. The downloader retrieves the archive and extracts `processed.cleveland.data`. There are 13 input features and one target column. `?` values are represented as missing and imputed within the model pipeline. **Actual row count and missing-value summary:** [insert from run].

## 4. EDA
Run `python scripts/run_eda.py`. Insert `class_balance.png`, `feature_histograms.png`, and `correlation_heatmap.png`. Discuss class distribution, feature ranges, missing values, and correlations based on your generated figures. Avoid interpreting correlation as causation.

## 5. Feature engineering and models
The feature pipeline standardizes numeric measurements and imputes missing values. Coded clinical categories are imputed with the most frequent value and one-hot encoded. The preprocessing is fitted within each training fold to reduce data leakage. Models compared: class-weighted Logistic Regression and Random Forest. Data split: 80% train / 20% hold-out test, stratified with random seed 42. Five-fold stratified CV is run on the training partition. Final model selection rule: highest mean CV ROC-AUC.

## 6. Evaluation
Insert actual `reports/model/metrics.json` values. Report hold-out accuracy, precision, recall, ROC-AUC and CV means/variability. Discuss trade-offs and limitations of a small dataset. Do not choose a model based only on accuracy; in a health-risk context, false negatives and recall matter, but the model must not be used clinically.

**Results table (fill after training):**

| Model | Test accuracy | Test precision | Test recall | Test ROC-AUC | CV ROC-AUC mean |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | [run] | [run] | [run] | [run] | [run] |
| Random Forest | [run] | [run] | [run] | [run] | [run] |

## 7. Experiment tracking
MLflow records model name, hyperparameters, random seed, test split, CV settings, metrics, confusion matrix, ROC curve, classification report, and sklearn model artifact. Insert screenshot of your own MLflow run page and list actual run IDs/selected model.

## 8. Packaging and reproducibility
The complete sklearn pipeline is persisted with joblib, including preprocessing and classifier. `requirements.txt` describes dependencies. The API loads the model artifact through `MODEL_PATH`. Explain environment setup and include the exact commands used.

## 9. API, Docker and monitoring
FastAPI exposes `/health` and `/predict`. The prediction response contains the binary class, label, maximum class probability, per-class probabilities, and a safety disclaimer. Request middleware logs method, path, status, and duration and emits `X-Process-Time-ms`. Insert actual API screenshots and a sample request/response. Explain that confidence is a model probability, not calibrated clinical certainty.

## 10. CI/CD, deployment, and conclusion
GitHub Actions runs Ruff, pytest, dataset acquisition, training smoke test, and artifact upload. Kubernetes manifests specify one replica, service, and health probes. Document actual deployment method and commands, pod/service output, and access URL. If deployment was only tested locally, state that accurately.

Conclusion: summarize what ran successfully, lessons learned, known limitations, and future work (data validation, drift monitoring, model registry/staging, security, privacy, clinical validation, and retraining policy).

## Architecture
```text
UCI dataset
   ↓ download + normalization
EDA / data checks
   ↓
train-test split + sklearn preprocessing Pipeline
   ↓
Logistic Regression ──┐
Random Forest ────────┼── 5-fold CV + hold-out metrics
                      ↓
             MLflow runs + plots
                      ↓
             versioned joblib pipeline
                      ↓
        FastAPI /predict + request logs
                      ↓
             Docker container
                      ↓
       Kubernetes Deployment + Service
```

## Evidence appendix
- [ ] EDA plots
- [ ] MLflow run screenshot
- [ ] Pytest output
- [ ] GitHub Actions run screenshot
- [ ] Docker build/run screenshots
- [ ] API `/health` and `/predict` screenshots
- [ ] Kubernetes pods/service screenshot, if used
- [ ] GitHub repository URL
- [ ] Video recording link or filename
