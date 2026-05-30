# Fraud Detection MLOps - Phase 5 Project

Complete end-to-end fraud detection project with production-grade MLOps tools: Docker, Evidently (drift detection), DVC (data versioning), and Azure ML (cloud training).

This is the practical hands-on project for **Phase 5** of my MLOps learning journey. It builds on Phase 4 by adding enterprise-ready tools.

---

## What This Project Demonstrates

| MLOps Concept | Tool/Implementation | Where to See It |
|---------------|---------------------|-----------------|
| **Containerization** | Docker | `Dockerfile` + container deployment |
| **Model drift detection** | Evidently AI | `src/monitor.py` + HTML reports |
| **Data versioning** | DVC | `.dvc/` folder + data tracking |
| **Cloud training** | Azure ML | Training job on Azure compute |
| **Experiment tracking** | MLflow | `src/train.py` + MLflow UI |
| **Feature engineering** | Scikit-learn | `src/features.py` |
| **Model serving** | Flask API | `src/predict.py` |

This project shows how to take an ML model from laptop to production-ready deployment.

---

## Project Structure

```
fraud-detection-project/
├── data/
│   ├── raw/
│   │   ├── creditcard.csv           # Generated dataset (gitignored, DVC tracked)
│   │   └── creditcard.csv.dvc       # DVC pointer file (in Git)
│   └── processed/
│       └── creditcard_processed.csv # Feature-engineered data
├── models/
│   ├── fraud_model.pkl              # Trained RandomForest model
│   └── scaler.pkl                   # Feature scaler (for inference)
├── reports/
│   └── drift_report.html            # Evidently drift detection report
├── src/
│   ├── features.py                  # Feature engineering pipeline
│   ├── train.py                     # Training with MLflow (local)
│   ├── train_azure.py               # Training script for Azure ML
│   ├── predict.py                   # Flask API for inference
│   └── monitor.py                   # Drift detection with Evidently
├── .dvc/                            # DVC configuration
├── .amlignore                       # Azure ML upload exclusions
├── .gitignore                       # Git exclusions
├── Dockerfile                       # Container definition
├── generate_data.py                 # Synthetic dataset generator
├── requirements.txt                 # Python dependencies
├── create_compute.py                # Azure ML compute cluster setup
├── submit_job.py                    # Submit training job to Azure
└── README.md                        # This file
```

---

## Prerequisites

Before starting, you need:

1. **Python 3.11** installed
2. **Docker Desktop** installed and running
3. **Git** installed
4. **Azure account** (optional, for cloud training - free tier available)
5. **Basic command line** knowledge

---

## Complete Setup Guide - Step by Step

Follow these steps exactly as written. Each step explains what it does and why.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-phase5.git
cd fraud-detection-phase5
```

**What this does:** Downloads the project code to your local machine.

**Why:** You need the code locally to run experiments and make changes.

---

### Step 2: Create Virtual Environment

```bash
python -m venv fraud_env
```

**What this does:** Creates an isolated Python environment called `fraud_env`.

**Why:** Virtual environments prevent package conflicts. Your system Python stays clean, and this project gets its own packages. If something breaks, just delete `fraud_env/` and recreate - your system is unaffected.

**Troubleshooting:** If `python` command not found, try `python3` or `py`.

---

### Step 3: Activate Virtual Environment

**Windows (Git Bash):**
```bash
source fraud_env/Scripts/activate
```

**Linux/Mac:**
```bash
source fraud_env/bin/activate
```

**What this does:** Switches your terminal to use the virtual environment's Python.

**Why:** Now when you run `pip install`, packages go into `fraud_env/`, not system-wide.

**How to confirm:** Your terminal prompt should show `(fraud_env)` at the beginning.

---

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**What this does:** Installs all Python packages needed for the project.

**Packages installed:**
- `numpy`, `pandas`: Data manipulation
- `scikit-learn`: Machine learning algorithms
- `mlflow`: Experiment tracking
- `flask`: Web API framework
- `joblib`: Model serialization
- `dvc`: Data version control
- `azure-ai-ml`: Azure ML SDK (for cloud training)

**Why specific versions:** The `requirements.txt` specifies compatible versions to avoid conflicts (e.g., `numpy<2`, `pandas<3` work with `mlflow==2.16.2`).

**Estimated time:** 2-3 minutes

**Troubleshooting:** 
- If `pip install` fails, upgrade pip first: `python -m pip install --upgrade pip`
- If you get `pkg_resources` errors, install setuptools: `pip install setuptools`

---

### Step 5: Create Required Folders

```bash
mkdir -p data/raw data/processed models reports
```

**What this does:** Creates folder structure for data, models, and outputs.

**Why:** Python scripts will write to these folders. Creating them upfront prevents "directory not found" errors.

**Folder purposes:**
- `data/raw/`: Original datasets
- `data/processed/`: Cleaned, feature-engineered data
- `models/`: Trained models and scalers
- `reports/`: Drift detection reports

---

### Step 6: Generate Synthetic Data

```bash
python generate_data.py
```

**What this does:** Creates a synthetic credit card fraud dataset with 10,000 transactions.

**How it works:**
- Legitimate transactions: ~9,800 (98%), amounts drawn from gamma distribution (average ~$100)
- Fraudulent transactions: ~200 (2%), higher amounts and different patterns
- 10 anonymized features (V1-V10) simulating PCA-transformed transaction data
- Random seed = 42 (for reproducibility)

**Output:** `data/raw/creditcard.csv` (about 2.3 MB)

**Why synthetic data:** Real fraud data is sensitive/proprietary. Synthetic data lets you learn without compliance issues. Patterns are realistic enough to train a working model.

**Expected output:**
```
Generating synthetic fraud dataset...
Done! Total: 10000, Fraud: 200
```

**Verification:** Check file exists: `ls -lh data/raw/creditcard.csv`

---

### Step 7: Feature Engineering

```bash
python src/features.py
```

**What this does:** Transforms raw data into ML-ready features.

**Features created:**
1. **Hour** - Hour of day (0-23) from timestamp
2. **Is_Night** - Binary flag (1 if transaction between 10 PM - 6 AM)
3. **Log_Amount** - Log-transformed amount (handles skewed distribution)
4. **Amount_Bin** - Categorical amount bins (low/medium/high/very_high)
5. **High_Risk** - Combination feature (large amount + night time = suspicious)

**Why feature engineering matters:**
- Raw data rarely works well - models need meaningful features
- Time-based features: Frauds happen more at night (3 AM)
- Amount patterns: Fraudsters test with small amounts, then large
- Feature combinations: High amount + odd time = strong fraud signal

**Scaling:**
All features are standardized (mean=0, std=1) using `StandardScaler`. This is critical because:
- RandomForest doesn't strictly need scaling, but...
- It helps model convergence
- Required if you later use models like Logistic Regression, Neural Networks
- **Most important:** The scaler is saved to `models/scaler.pkl`

**Why save the scaler?**
At inference time (when API gets a new transaction), we must apply THE EXACT SAME scaling. If training data had mean=100 and we scaled it, but production data has mean=200 and we don't scale it the same way, predictions will be garbage. Saved scaler ensures training and inference transformations match.

**Expected output:**
```
Loading data from data/raw/creditcard.csv
Original shape: (10000, 13), Fraud ratio: 2.00%
Scaler saved to models/scaler.pkl
Processed data saved to data/processed/creditcard_processed.csv
```

**Verification:** Check files:
```bash
ls models/scaler.pkl
ls data/processed/creditcard_processed.csv
```

---

### Step 8: Train Model (Local with MLflow)

```bash
python src/train.py
```

**What this does:** Trains TWO RandomForest models with different hyperparameters and tracks both experiments in MLflow.

**Training details:**

**Experiment 1 - Baseline:**
- 100 trees (n_estimators)
- Max depth: 10
- Class weight: balanced (handles imbalanced data)

**Experiment 2 - More Capacity:**
- 200 trees
- Max depth: 15
- Class weight: balanced

**Why two experiments?**
Real ML work involves trying different configurations. MLflow tracks both so you can compare and pick the best.

**What gets logged to MLflow:**
- **Parameters:** n_estimators, max_depth, etc.
- **Metrics:** accuracy, precision, recall, F1 score, ROC-AUC
- **Artifacts:** The trained model
- **Tags:** model_type, project name

**Metrics explained:**
- **Accuracy:** Overall correctness (but misleading for imbalanced data)
- **Recall:** Of all real frauds, how many did we catch? (MOST IMPORTANT for fraud)
- **Precision:** Of what we flagged, how many were really fraud?
- **F1 Score:** Harmonic mean of precision and recall (balance)
- **ROC-AUC:** Area under ROC curve (overall model quality)

**Why recall matters most in fraud:**
Missing a fraud (false negative) = regulatory risk, money laundering
Flagging a legit transaction (false positive) = customer inconvenience
**Missing fraud is more costly**, so recall is prioritized.

**Expected output:**
```
Experiment 1: Baseline
Training set: (8000, 17), Test set: (2000, 17)
==================================================
RESULTS
accuracy    : 0.9845
precision   : 0.8923
recall      : 0.7000
f1_score    : 0.7778
...
==================================================

Experiment 2: More capacity
...
recall      : 0.7000
...

Done! Run 'mlflow ui' to see experiments.
```

**Save trained model:** The better-performing model is saved to `models/fraud_model.pkl`

---

### Step 9: View Experiments in MLflow UI

```bash
mlflow ui
```

**What this does:** Starts MLflow web interface on `http://localhost:5000`

**How to use it:**
1. Open browser: `http://localhost:5000`
2. Left sidebar: Click on **"fraud-detection"** experiment
3. You'll see two runs (rf-YYYYMMDD-HHMMSS timestamps)
4. **Compare runs:**
   - Select both checkboxes
   - Click "Compare" button
   - Side-by-side comparison of all metrics
5. **Pick the winner:** Look at `recall` and `f1_score` - higher is better

**Why this matters:**
- Without MLflow: You'd manually note down results in a spreadsheet (error-prone, hard to search)
- With MLflow: Automatic tracking, searchable, filterable, visual comparison
- Enterprise context: Data scientists run 100+ experiments - MLflow makes them manageable

**What you learn:**
- Experiment 2 (200 trees) typically has slightly better recall
- Both models have similar performance (diminishing returns)
- This informs decisions: is extra compute worth marginal gain?

**To stop MLflow UI:** Press `Ctrl+C` in the terminal

---

### Step 10: Run Inference API (Local)

```bash
python src/predict.py
```

**What this does:** Starts a Flask web API on `http://localhost:5001` that serves fraud predictions.

**API loads:**
1. Trained model (`models/fraud_model.pkl`)
2. Feature scaler (`models/scaler.pkl`)

Both are loaded **once at startup** (not per request) for efficiency.

**API endpoints:**

**1. Health check:** `GET /health`
```bash
curl http://localhost:5001/health
```
Response: `{"status": "healthy", "timestamp": "..."}`

**Purpose:** Load balancers use this to check if API is alive.

**2. Prediction:** `POST /predict`
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 50000, "Amount": 999.99, "V1": -3.5, "V2": 3.1, "V3": -2.8, "V4": 2.5, "V5": -3.2, "V6": 2.8, "V7": -2.3, "V8": 2.7, "V9": -3.1, "V10": 2.9}'
```

Response:
```json
{
  "prediction": 1,
  "fraud_probability": 0.665,
  "risk_level": "MEDIUM",
  "timestamp": "..."
}
```

**How prediction works:**
1. API receives transaction JSON
2. Applies feature engineering (same as training: Hour, Is_Night, Log_Amount, etc.)
3. Scales features using saved scaler
4. Model predicts: 0 (legit) or 1 (fraud)
5. Returns prediction + probability + risk level

**Risk levels:**
- HIGH: fraud_probability > 0.7
- MEDIUM: 0.3 to 0.7
- LOW: < 0.3

**Why this matters:**
Real-time fraud detection needs immediate response. This API can be called from:
- ATM transaction systems
- Online payment gateways
- Banking apps
Response time: ~50ms (fast enough for real-time)

**To stop API:** Press `Ctrl+C`

---

### Step 11: Containerize API with Docker

**What is containerization:** Package the API + all dependencies into a portable container that runs identically everywhere.

#### Build Docker Image

```bash
docker build -t fraud-api:v1 .
```

**What this does:** Reads `Dockerfile` and creates a Docker image.

**Dockerfile explanation (line by line):**

```dockerfile
FROM python:3.11-slim
```
Start with lightweight Python 3.11 base image (40 MB instead of 900 MB full version).

```dockerfile
WORKDIR /app
```
Set `/app` as working directory inside container.

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
Copy requirements first, install packages. This leverages Docker **layer caching** - if requirements.txt doesn't change, this layer is reused (faster rebuilds).

```dockerfile
COPY src/ ./src/
COPY models/ ./models/
```
Copy application code and trained models into container.

```dockerfile
EXPOSE 5001
```
Document that container uses port 5001 (metadata, doesn't actually open port).

```dockerfile
CMD ["python", "src/predict.py"]
```
When container starts, run this command (start the API).

**Build time:** First build takes 2-3 minutes (downloads Python image, installs packages). Subsequent builds are faster (uses cache).

**Verify image created:**
```bash
docker images
```
You should see `fraud-api` with tag `v1`.

#### Run Container

```bash
docker run -p 5001:5001 fraud-api:v1
```

**What this does:** Creates and starts a container from the image.

**Flag explanation:**
- `-p 5001:5001`: Maps container's port 5001 to your machine's port 5001
  - First 5001 = your machine (localhost)
  - Second 5001 = inside container
  - Without this, you can't access the API

**Expected output:**
```
Loading model and scaler...
Model loaded successfully!
Starting Fraud Detection API on http://localhost:5001
* Running on http://127.0.0.1:5001
```

**Test the containerized API:**
```bash
# In a new terminal
curl http://localhost:5001/health
```

**What you achieved:** API now runs in a container. You can:
- Share this container with anyone (via Docker Hub)
- Deploy to Kubernetes
- Run on any machine (Windows/Linux/Mac)
- No environment setup needed on target machine

**To stop container:** Press `Ctrl+C`

---

### Step 12: Share Container via Docker Hub (Optional)

If you want to run this on another machine:

#### Login to Docker Hub

```bash
docker login
```

Enter your Docker Hub username and password.

#### Tag Image

```bash
docker tag fraud-api:v1 YOUR_DOCKERHUB_USERNAME/fraud-api:v1
```

Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username.

#### Push to Docker Hub

```bash
docker push YOUR_DOCKERHUB_USERNAME/fraud-api:v1
```

**What this does:** Uploads your image to Docker Hub (like GitHub for containers).

**On another machine:**
```bash
docker pull YOUR_DOCKERHUB_USERNAME/fraud-api:v1
docker run -p 5001:5001 YOUR_DOCKERHUB_USERNAME/fraud-api:v1
```

API runs without installing Python, packages, or anything. Just Docker.

---

### Step 13: Track Data with DVC

DVC (Data Version Control) versions large data files without bloating Git.

#### Initialize DVC

```bash
dvc init
```

**What this does:** Creates `.dvc/` folder (like `.git/` for Git).

**Output:** "Initialized DVC repository"

#### Track Dataset

```bash
dvc add data/raw/creditcard.csv
```

**What this does:**
1. Moves actual data file to `.dvc/cache/` (hidden storage)
2. Creates `data/raw/creditcard.csv.dvc` (small pointer file, ~200 bytes)
3. Adds `creditcard.csv` to `.gitignore` (so Git ignores the big file)

**Why this matters:**
- Big file (2.3 MB) doesn't go to Git
- Only pointer file (~200 bytes) goes to Git
- You can version data like code: commit, go back to old versions
- Audit trail: "Which data was used to train model v2.3?" → DVC tells you

#### Commit DVC Files

```bash
git add data/raw/creditcard.csv.dvc data/raw/.gitignore .dvc/
git commit -m "Track fraud dataset with DVC"
```

**What this does:** Commits the small DVC pointer file to Git. The actual data stays in `.dvc/cache/` locally.

#### Push Data to Remote (Optional, for team collaboration)

```bash
# Configure remote storage (e.g., Azure Blob)
dvc remote add -d myremote azure://mycontainer/dvc-storage

# Push data to cloud
dvc push
```

**What this does:** Uploads data from local cache to cloud storage. Team members can then `dvc pull` to download.

**Team workflow:**
```bash
# Colleague clones repo
git clone <repo-url>
cd fraud-detection-phase5
dvc pull  # Downloads data from cloud to local cache
```

**Banking use case:**
Training data from January is stored in DVC remote. In July, regulator asks for that exact data. Any team member can:
```bash
git checkout <jan-commit>
dvc checkout
# Exact January data restored
```

---

### Step 14: Detect Data Drift with Evidently

Evidently monitors if production data has changed compared to training data (drift).

#### Setup Evidently Environment

Because Evidently has dependency conflicts with MLflow, use a separate environment:

```bash
python -m venv evidently_env
source evidently_env/Scripts/activate  # Windows: evidently_env/Scripts/activate
pip install evidently pandas "numpy<2"
```

**Why separate environment:** Evidently needs pydantic v1, but newer MLflow needs pydantic v2. Conflict. Separate environments = no conflict.

#### Run Drift Detection

```bash
python src/monitor.py
```

**What this script does:**
1. Loads reference data (original training data)
2. Simulates production data with drift (shifts some features to mimic changed fraud patterns)
3. Compares the two using Evidently's `DataDriftPreset`
4. Generates HTML report with visual charts
5. Prints drift summary and alert if threshold exceeded

**Expected output:**
```
Loading reference data...
Simulating new production data with drift...
Checking for drift...
Done! Open reports/drift_report.html in your browser.

Drifted features: 30%
ALERT: Too much drift! Model retraining recommended.
```

#### View Drift Report

Open `reports/drift_report.html` in your browser.

**What you'll see:**
- **Dashboard:** Overall drift summary
- **Data Drift:** Feature-by-feature analysis
  - Green = no drift (production similar to training)
  - Red = drift detected (distribution changed)
- **Charts:** Histograms comparing training vs production distributions
- **Statistics:** P-values, drift scores for each feature

**How to interpret:**
- If V1, V2, V3 show red: These features shifted significantly
- In fraud detection: Could mean new fraud techniques, merchant category changes, or seasonal patterns
- **Action:** If >30% features drifted → retrain model with recent data

**Real scenario:**
Spectre AI runs drift detection weekly. If drift detected:
1. Alert sent to ML team
2. Investigate cause (data quality issue? Real pattern change?)
3. If real change: Schedule retraining
4. If data issue: Fix pipeline, reprocess

**Return to main environment:**
```bash
deactivate  # Exit evidently_env
source fraud_env/Scripts/activate  # Back to fraud_env
```

---

### Step 15: Train on Azure ML (Cloud)

Now we move from laptop to cloud. Train the same model on Azure's powerful machines.

#### Prerequisites

1. **Azure account:** Free tier ($200 credit) - https://azure.microsoft.com/free
2. **Azure ML Workspace created:** (covered in Phase 5 learning)
3. **Azure CLI installed and logged in:** `az login`

#### Install Azure ML SDK

```bash
pip install azure-ai-ml azure-identity
```

#### Create Compute Cluster

```bash
python create_compute.py
```

**What this script does:**
- Connects to your Azure ML workspace
- Creates a compute cluster named `cpu-cluster`
- VM size: `STANDARD_DS3_V2` (4 cores, 14 GB RAM)
- Auto-scaling: 0 to 1 nodes (scales to zero when idle = save money)

**Cost:** Only pay when jobs run (~$0.096/hour). When idle, it's free.

**Expected output:**
```
Creating compute 'cpu-cluster'...
Compute created!
```

**Verify in portal:** 
- Go to `portal.azure.com` → Azure ML workspace → Compute tab
- `cpu-cluster` should appear

#### Submit Training Job to Azure

Before submitting, create `.amlignore` to exclude unnecessary files:

```bash
cat > .amlignore << 'EOF'
fraud_env/
evidently_env/
mlruns/
.dvc/
.git/
__pycache__/
*.pyc
reports/
Dockerfile
*.py~
EOF
```

**Why .amlignore:** Azure ML uploads your entire folder to cloud. Without ignore file, it would upload:
- `fraud_env/` (100+ MB of packages)
- `mlruns/` (MLflow experiment data)
- Git history

This wastes upload time and storage. `.amlignore` excludes them - only src/, data/, and essentials upload.

Now submit:

```bash
python submit_job.py
```

**What this script does:**
1. Connects to Azure ML workspace
2. Defines a training job:
   - Code: Current folder (excluding .amlignore patterns)
   - Command: `python src/train_azure.py`
   - Environment: Preconfigured scikit-learn image
   - Compute: `cpu-cluster`
3. Uploads code/data to Azure
4. Submits job to queue

**Expected output:**
```
Job submitted! Name: friendly_eagle_abc123xyz
View in portal: https://ml.azure.com/runs/friendly_eagle_abc123xyz...
```

**Click the URL** - opens Azure ML Studio where you can:
- See job status: Queued → Running → Completed
- View live logs: `Outputs + logs` → `user_logs` → `std_log.txt`
- Download outputs: `outputs/fraud_model.pkl`

**Timeline:**
- Queued: 1-2 minutes (compute starting)
- Running: 5-7 minutes (training)
- Completed: Model saved

**Results on Azure:**
```
Training set: (8000, 17), Test set: (2000, 17)
==================================================
RESULTS
accuracy    : 0.9920
precision   : 0.8750
recall      : 0.7000
f1_score    : 0.7778
==================================================
Model saved to outputs/
```

**Same model, cloud-trained!** Results match local training (reproducibility).

**What you achieved:**
- Trained model on cloud compute (scalable)
- Job tracked in Azure ML (team visibility)
- Can scale to larger data/more complex models
- Production-ready workflow

---

## File-by-File Explanation

### `generate_data.py`

**Purpose:** Create synthetic fraud dataset for learning/testing.

**Key functions:**
- `generate_fraud_dataset()`: Creates 10K transactions, 2% fraud
- Legitimate: Gamma distribution (average ~$100)
- Fraudulent: Different distribution (higher amounts, different V-features)
- Random seed 42: Makes dataset reproducible

**Why synthetic:** Real fraud data is confidential. Synthetic data mimics real patterns for learning.

**Output:** `data/raw/creditcard.csv`

---

### `src/features.py`

**Purpose:** Transform raw data into ML-ready features.

**Key functions:**
- `engineer_features()`: Creates time-based, amount-based, and composite features
- `prepare_data()`: Loads data, engineers features, scales, saves

**Features created:**
- `Hour`: Time of day (fraud patterns vary by hour)
- `Is_Night`: Night flag (frauds spike 10 PM - 6 AM)
- `Log_Amount`: Log-transform (handles skewed amount distribution)
- `Amount_Bin`: Categorical bins (models can learn thresholds)
- `High_Risk`: Composite (large + night = high fraud signal)

**Scaling:** StandardScaler fits on training data, transforms to mean=0, std=1.

**Critical:** Scaler saved to `models/scaler.pkl` - must use same scaler at inference.

**Output:** `data/processed/creditcard_processed.csv`, `models/scaler.pkl`

---

### `src/train.py`

**Purpose:** Train fraud detection model with MLflow experiment tracking (local).

**Key features:**
- Trains TWO models with different hyperparameters
- Logs everything to MLflow: params, metrics, model artifacts
- Uses `class_weight='balanced'` to handle imbalanced data (2% fraud)
- Evaluates on multiple metrics: accuracy, precision, recall, F1, ROC-AUC
- Saves best model to `models/fraud_model.pkl`

**Why two experiments:** Compare configurations to find optimal model.

**MLflow integration:**
- `mlflow.set_experiment()`: Group related runs
- `mlflow.start_run()`: Begin tracking one experiment
- `mlflow.log_params()`: Record hyperparameters
- `mlflow.log_metrics()`: Record performance metrics
- `mlflow.sklearn.log_model()`: Save model as artifact

**Output:** Trained model + MLflow tracking data in `mlruns/`

---

### `src/train_azure.py`

**Purpose:** Simplified training script for Azure ML (without MLflow).

**Why different from train.py:**
- Azure ML has built-in experiment tracking (don't need MLflow)
- Simpler dependencies (faster cloud setup)
- Saves outputs to `outputs/` folder (Azure standard)
- Metrics saved as JSON for Azure to parse

**When to use:** Submitting jobs to Azure ML via `submit_job.py`.

**Output:** `outputs/fraud_model.pkl`, `outputs/metrics.json`

---

### `src/predict.py`

**Purpose:** Flask REST API for real-time fraud predictions.

**Architecture:**
- Loads model and scaler once at startup (efficient)
- `/health` endpoint: Liveness check
- `/predict` endpoint: Takes transaction JSON, returns prediction

**Key functions:**
- `engineer_features()`: Applies same feature engineering as training
- Ensures training-inference consistency (critical!)

**Production considerations:**
- Logging: Each prediction logged (for monitoring)
- Error handling: Returns 400/500 errors with messages
- Column order: Must match training exactly

**Why Flask:** Lightweight, easy to deploy. In production, you'd use:
- Gunicorn (production WSGI server)
- NGINX (reverse proxy)
- Kubernetes (orchestration)

But Flask is perfect for learning and prototyping.

---

### `src/monitor.py`

**Purpose:** Detect data drift using Evidently AI.

**How it works:**
1. Loads reference data (training baseline)
2. Simulates production data (with intentional drift for demo)
3. Compares distributions using statistical tests
4. Generates HTML report with visualizations
5. Prints alert if drift > 30%

**Real production usage:**
- Run weekly/daily as scheduled job
- Compare last week's production data vs training data
- If drift detected: Alert ML team, trigger retraining pipeline
- Save reports for audit trail

**Metrics checked:**
- Feature distribution shifts (KS-test, Chi-squared)
- Share of drifted columns
- Individual feature drift scores

**Output:** `reports/drift_report.html` (visual report), console alert

---

### `Dockerfile`

**Purpose:** Recipe for building Docker container.

**Best practices used:**
- Slim base image (smaller size)
- Copy requirements before code (layer caching)
- No-cache-dir in pip (reduces image size)
- Specific working directory
- Document exposed port

**Image size:** ~250 MB (vs 900+ MB with full Python image)

---

### `create_compute.py`

**Purpose:** Create Azure ML compute cluster.

**Configuration:**
- `STANDARD_DS3_V2`: Good balance of CPU/RAM for training
- Min 0, Max 1: Auto-scales to zero (cost savings)
- Idle timeout: 120 seconds

**Why separate script:** Compute creation is one-time setup. Training jobs reuse the same compute.

---

### `submit_job.py`

**Purpose:** Submit training job to Azure ML.

**Job configuration:**
- `code="./"`Upload current folder (minus .amlignore patterns)
- `command`: What to run (`python src/train_azure.py`)
- `environment`: Pre-built scikit-learn environment
- `compute`: Which cluster to use

**Output:** Job ID and studio URL (to monitor in browser)

---

### `requirements.txt`

**Purpose:** List all Python dependencies with versions.

**Why version constraints:**
```txt
numpy<2          # Numpy 2.x breaks compatibility with older packages
pandas<3         # Pandas 3.x not compatible with MLflow 2.16
mlflow==2.16.2   # Specific version tested to work
flask<3          # MLflow requires Flask<3
```

**Without version control:** Package updates break your code months later.

**With version control:** Project runs the same way years later.

---

### `.gitignore`

**Purpose:** Tell Git which files to ignore (not track).

**Key exclusions:**
- `fraud_env/`, `evidently_env/`: Virtual environments (heavy, user-specific)
- `data/raw/*.csv`: Data files (tracked by DVC, not Git)
- `models/*.pkl`: Model files (generated, large)
- `mlruns/`: MLflow experiment data (local, regenerated)
- `__pycache__/`: Python bytecode (temporary)

**Why ignore these:** Keep Git repository small, fast, focused on code.

---

### `.amlignore`

**Purpose:** Tell Azure ML which files NOT to upload when submitting jobs.

**Key exclusions:**
- `fraud_env/`, `evidently_env/`: Environments (Azure provides its own)
- `mlruns/`: Local experiment data (not needed on Azure)
- `.dvc/`, `.git/`: Version control metadata (not needed for training)
- `Dockerfile`, test scripts: Not part of training pipeline

**Why:** Faster uploads, lower storage costs, cleaner jobs.

---

## Troubleshooting Guide

### Issue: `mlflow ui` gives ImportError

**Cause:** MLflow version incompatible with Python or setuptools.

**Fix:**
```bash
pip uninstall mlflow
pip install mlflow==2.16.2
pip install setuptools
```

### Issue: API returns wrong predictions

**Cause:** Feature engineering at inference doesn't match training.

**Fix:** Ensure `predict.py` uses same:
- Feature columns
- Column order
- Scaling (same scaler.pkl)
- Feature formulas

### Issue: Docker build fails

**Cause:** Usually missing files or wrong paths.

**Fix:**
- Check Dockerfile paths
- Ensure `src/` and `models/` exist
- Run `docker build` from project root

### Issue: Azure ML job fails with "No module named X"

**Cause:** Missing package in environment.

**Fix:** Use curated environment or create custom:
```yaml
# conda.yml
dependencies:
  - python=3.8
  - pip:
    - pandas
    - scikit-learn
    - your-package
```

### Issue: DVC "file not found" error

**Cause:** Tried to run `dvc pull` but no remote configured.

**Fix:**
```bash
# Local only? Skip dvc pull, run generate_data.py instead
python generate_data.py

# Team setup? Configure remote first
dvc remote add -d myremote <storage-url>
```

---

## Key Learnings & Takeaways

### What makes this production-ready?

| Aspect | Development (Phase 4) | Production (Phase 5) |
|--------|----------------------|----------------------|
| **Portability** | Runs only on my laptop | Docker → runs anywhere |
| **Monitoring** | Manual checks | Evidently → automated drift detection |
| **Data versioning** | Manual file copies | DVC → audit-ready versioning |
| **Compute** | Limited to laptop | Azure ML → scalable cloud compute |
| **Collaboration** | Solo | Team can access workspace, share models |
| **Deployment** | Manual Flask app | Azure endpoints with auto-scaling |
| **Cost** | Free (laptop) | Pay-per-use (efficient for production) |

### MLOps Concepts Demonstrated

1. **Reproducibility:** DVC + fixed seeds + saved scalers = exact replication possible
2. **Versioning:** Code (Git) + Data (DVC) + Models (MLflow/Azure) = complete lineage
3. **Automation:** Scripts replace manual steps (one command to train, deploy, monitor)
4. **Monitoring:** Evidently detects drift before model fails
5. **Scalability:** Azure compute handles any data size
6. **Collaboration:** Central platform (Azure ML) for team access

### Banking/Compliance Benefits

** Audit Scenario:**

**Question:** "Your model missed fraud transaction XYZ on March 15. Investigate."

**Answer with this setup:**
1. **Git log:** Find code version deployed on March 15
2. **DVC checkout:** Retrieve exact training data from that time
3. **MLflow:** See model version, hyperparameters, accuracy at deployment
4. **Evidently reports:** Check if drift was detected before March 15
5. **Azure ML:** Job history shows who trained, when, what compute

**Complete audit trail.** Every decision is data-driven and logged.

---

## What to Do Next

### Immediate

- Push all changes to GitHub (new repo for Phase 5)
- Update learning repo with Phase 5 notes
- Review Azure ML Studio - explore job history

### Short-term

- Deploy model to Azure ML endpoint (real-time API in cloud)
- Set up scheduled drift monitoring
- Integrate with CI/CD pipeline

### Long-term

- Learn Phase 6: DevOps to MLOps mapping
- Learn Phase 7: Advanced deployment strategies
- Build more complex projects

---

## Commands Cheat Sheet

```bash
# Environment
source fraud_env/Scripts/activate

# Data
python generate_data.py
python src/features.py

# Training
python src/train.py
mlflow ui

# API
python src/predict.py
curl http://localhost:5001/health

# Docker
docker build -t fraud-api:v1 .
docker run -p 5001:5001 fraud-api:v1

# DVC
dvc init
dvc add data/raw/creditcard.csv
git add data/raw/creditcard.csv.dvc
dvc push  # if remote configured

# Evidently (use evidently_env)
source evidently_env/Scripts/activate
python src/monitor.py
open reports/drift_report.html

# Azure ML
python create_compute.py
python submit_job.py
# View: portal.azure.com → ML workspace → Jobs
```

---

<img width="959" height="469" alt="image" src="https://github.com/user-attachments/assets/0c2c5da9-1230-4cff-b080-7bd8017c6985" />


<img width="959" height="470" alt="image" src="https://github.com/user-attachments/assets/3fff9623-867b-4ec5-a499-2a933a8b84eb" />

