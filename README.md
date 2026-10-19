# Heart Disease Risk Classification — MLOps Assignment 01

Course: AIMLCZG523 — Machine Learning Operations (MLOps)

This repository is a reproducible learning project implementing data acquisition, EDA, model comparison, MLflow tracking, a FastAPI inference service, tests, GitHub Actions CI, Docker packaging, Kubernetes manifests, and request logging.

> **Important medical limitation:** This is an educational classifier, not a clinically validated diagnostic tool. Never use its predictions to make medical decisions.

## Project structure

```text
.
├── data/                         # downloaded dataset is stored here (not committed)
├── reports/                      # EDA plots, model metrics, report assets
├── models/                       # trained model pipeline and metadata
├── src/heart_ml/
│   ├── __init__.py
│   ├── data.py                   # UCI download and cleaning
│   ├── features.py               # feature schema and preprocessing
│   ├── train.py                  # training, CV, evaluation, MLflow
│   └── api.py                    # FastAPI /health and /predict
├── tests/
├── scripts/
├── k8s/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## 1. Requirements

- Python 3.10 or 3.11 recommended
- Git (for repository workflow)
- Docker Desktop is optional for the first stages; install it only when reaching containerization.
- Internet access is needed once to download the UCI dataset and install Python packages.

## 2. Create an environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3. Download and inspect data

```bash
python -m heart_ml.data --output data/raw/heart.csv
```

The downloader retrieves the UCI Heart Disease archive, reads the processed Cleveland file, converts `?` to missing values, coerces columns to numeric, and converts target values 0 to absence and 1–4 to presence. Dataset provenance and the original source are recorded in `data/README.md`.

Run exploratory analysis:

```bash
python scripts/run_eda.py --input data/raw/heart.csv --output reports/eda
```

Outputs include class balance, feature histograms, and a correlation heatmap. The dataset is small; interpret all patterns cautiously.

## 4. Train and compare models

```bash
python -m heart_ml.train --data data/raw/heart.csv --output models --reports reports/model --experiment heart-disease-assignment
```

The script uses a stratified hold-out set, 5-fold stratified cross-validation on the training portion, and compares Logistic Regression with Random Forest. Preprocessing is inside each sklearn Pipeline to reduce leakage. It logs parameters, CV scores, hold-out metrics, model artifacts, and plots to MLflow when available. It also writes `models/heart_disease_pipeline.joblib` and `reports/model/metrics.json`.

Start the MLflow UI in another terminal:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open the local address printed by MLflow. Do not claim experiment runs until the training command has completed successfully.

## 5. Run tests

```bash
pytest -q
```

The CI workflow installs dependencies, runs Ruff lint checks, runs tests, and executes a training smoke test when data can be downloaded. For a fully offline CI run, add a small committed test fixture rather than committing the full dataset.

## 6. Start the API locally

After training:

```bash
uvicorn heart_ml.api:app --host 127.0.0.1 --port 8000
```

Check health at `http://127.0.0.1:8000/health` and interactive API docs at `http://127.0.0.1:8000/docs`.

Example prediction body (all 13 features are required):

```json
{
  "age": 57, "sex": 1, "cp": 2, "trestbps": 140, "chol": 240,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
  "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 3
}
```

The API returns a binary model prediction and class probabilities. This example is for demonstrating the schema, not a real patient.

## 7. Docker

Install and start Docker Desktop, then from the repository root:

```bash
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```

The Docker build runs training during image build so the image is self-contained. This makes builds slower and is suitable for a student demonstration; a production system should build and validate a versioned model artifact separately and promote it through a registry.

Test `http://localhost:8000/health` and POST the example JSON to `/predict`. Save screenshots of actual build output, health response, and prediction response.

## 8. Kubernetes (optional after Docker works)

For local Kubernetes, install Docker Desktop and enable Kubernetes, or install Minikube. Build the image and ensure your cluster can access it. With Minikube:

```bash
minikube start
eval $(minikube docker-env)       # macOS/Linux; Windows PowerShell differs
docker build -t heart-disease-api:local .
kubectl apply -f k8s/
kubectl get pods,services
minikube service heart-disease-api --url
```

If using Docker Desktop Kubernetes, build the image locally and use the manifest's `imagePullPolicy: IfNotPresent`. Do not expose a public service unless you understand the security and cost implications. Kubernetes deployment is optional to validate only after local API/Docker success.

## 9. GitHub repository and CI

Create an empty GitHub repository, then:

```bash
git init
git add .
git commit -m "Build reproducible heart disease MLOps pipeline"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository. The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests. GitHub Actions logs and screenshots must be captured from your own run.

## 10. Evidence checklist

- [ ] Dataset source and download command
- [ ] EDA plots
- [ ] Cross-validation and hold-out metrics from actual run
- [ ] MLflow experiment page and logged artifacts
- [ ] `pytest -q` passing output
- [ ] GitHub Actions successful run
- [ ] Docker build and running container
- [ ] `/health` and `/predict` responses
- [ ] Kubernetes resources and service URL, if used
- [ ] Request logs / monitoring evidence
- [ ] GitHub repository URL
- [ ] Short screen recording of the pipeline

Never insert fabricated metrics, screenshots, deployment URLs, or experiment IDs into the final report. Replace all report placeholders only after running the corresponding commands.


### Troubleshooting: MLflow `UntrustedTypesFoundException` while saving a model

If training reaches model logging and then fails with `skops.io.exceptions.UntrustedTypesFoundException` mentioning `numpy.dtype`, use the current project version, which explicitly configures MLflow's scikit-learn model logging to use `cloudpickle` serialization. This error is from model serialization, not from EDA. Re-run `python -m heart_ml.train` from the project root after updating the code. Only load model artifacts from sources you trust.
