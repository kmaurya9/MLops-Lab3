# Apache Airflow: Lab 1 - Advanced ML Clustering Pipeline

## Overview

This project implements an advanced Apache Airflow DAG pipeline for machine learning clustering, running in Docker. The pipeline demonstrates key MLOps concepts including workflow orchestration, parallel model training, data validation, branching logic, and automated model selection.

## Dataset

**Wine Dataset** (from scikit-learn) — 178 samples with 13 chemical analysis features from three different wine cultivars grown in Italy. Features include alcohol content, malic acid, ash, magnesium, flavanoids, color intensity, and more. This dataset was chosen instead of the base lab's `file.csv` customer data (which clusters on only 3 columns: BALANCE, PURCHASES, CREDIT_LIMIT) to work with higher-dimensional data (13 features vs 3), making clustering more challenging and realistic.

## Pipeline Architecture

```
load_data → validate_data → preprocess_data
                                  │
             ┌──────────────┬─────┴────────┬──────────────┐
             ↓              ↓              ↓              ↓
       train_kmeans   train_dbscan   train_agglom     train_gmm
             ↓              ↓              ↓              ↓
             └──────────────┴─────┬────────┴──────────────┘
                                  ↓
                          compare_models
                                  ↓
                          branch_on_quality
                           ↓             ↓
                     high_quality    low_quality
                           ↓             ↓
                           └──── notify ──┘
```

## Tasks Description

| Task                       | Type                 | Description                                                                            |
| -------------------------- | -------------------- | -------------------------------------------------------------------------------------- |
| `load_data_task`           | PythonOperator       | Loads Wine dataset from CSV file                                                       |
| `validate_data_task`       | PythonOperator       | Checks for missing values, duplicates, zero-variance columns, minimum row count        |
| `data_preprocessing_task`  | PythonOperator       | Drops target column, applies StandardScaler, then PCA keeping 95% of variance          |
| `train_kmeans_task`        | PythonOperator       | Trains K-Means (k=2-10), finds optimal k via elbow method, calculates silhouette score |
| `train_dbscan_task`        | PythonOperator       | Trains DBSCAN with multiple eps values (0.5-3.0), selects best configuration           |
| `train_agglomerative_task` | PythonOperator       | Trains Agglomerative Clustering (k=2-10), selects best k by silhouette score           |
| `train_gmm_task`           | PythonOperator       | Trains Gaussian Mixture Models (k=2-10), selects best k by BIC                         |
| `compare_models_task`      | PythonOperator       | Compares all four models by silhouette score, determines best model                    |
| `branch_quality_task`      | BranchPythonOperator | Routes to high or low quality path based on silhouette score threshold (0.3)           |
| `high_quality_task`        | PythonOperator       | Promotes best model to production                                                      |
| `low_quality_task`         | PythonOperator       | Retrains K-Means with extended range (k=2-20)                                          |
| `notify_task`              | PythonOperator       | Summarizes pipeline results                                                            |

## Customizations Beyond Base Lab

1. **Different Dataset**: Wine dataset (13 features, 178 samples) instead of the base lab's `file.csv` data (3 features used, 8,950 rows)
2. **Data Validation Task**: Automated quality checks before preprocessing
3. **PCA Dimensionality Reduction**: Scaled features are reduced with PCA (95% variance kept, 13 → 10 dimensions) before clustering
4. **Parallel Model Training**: Four clustering algorithms run simultaneously (K-Means, DBSCAN, Agglomerative, Gaussian Mixture)
5. **Probabilistic Model with BIC Selection**: Gaussian Mixture picks its number of components by BIC instead of silhouette/elbow
6. **Model Comparison**: Silhouette score-based comparison across all models
7. **Branching Logic**: BranchPythonOperator routes pipeline based on model quality
8. **Automated Retraining**: Low-quality models trigger retraining with expanded hyperparameter search
9. **Production Model Promotion**: Best model is automatically saved as the production model

## Project Structure

```
Lab_1/
├── dags/
│   ├── ml_pipeline_dag.py          # Main DAG definition
│   ├── rocket_tracker.py           # Bonus: Rocket launch tracker DAG
│   ├── .airflowignore              # Tells Airflow to skip src/ directory
│   └── src/
│       ├── __init__.py             # Package init
│       ├── lab.py                  # ML pipeline functions
│       ├── wine.csv                # Wine dataset
│       └── iris.csv                # Iris dataset (alternative, not used by the DAG)
├── config/
│   └── airflow.cfg                 # Airflow configuration
├── data/                           # Data directory
├── logs/                           # Airflow logs (gitignored)
├── plugins/                        # Airflow plugins
├── working_data/                   # Model output directory
├── docker-compose.yaml             # Docker Compose config
├── .env                            # Environment variables
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Prerequisites

- Docker Desktop (4GB+ memory allocated)
- Git

## Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/kmaurya9/MLops-Lab3.git
   cd MLops-Lab3/AirflowLabs/Lab1
   ```

2. **Create environment file**

   ```bash
   echo -e "AIRFLOW_UID=$(id -u)" > .env
   ```

3. **Create required directories**

   ```bash
   mkdir -p logs plugins working_data data
   ```

4. **Initialize Airflow database**

   ```bash
   docker compose up airflow-init
   ```

5. **Start Airflow**

   ```bash
   docker compose up
   ```

6. **Access the UI**
   - Open http://localhost:8080
   - Login with `airflow2` / `airflow2`

7. **Trigger the DAG**
   - Find `your_python_dag` in the DAGs list
   - Unpause it with the toggle switch (DAGs start paused; a run triggered while paused stays queued)
   - Click the play button to trigger manually
   - Trained models are written to `working_data/`

8. **Stop Airflow**

   ```bash
   docker compose down
   ```

## Docker Compose Changes

The following modifications were made to the default Airflow docker-compose.yaml:

1. **Disabled example DAGs**: `AIRFLOW__CORE__LOAD_EXAMPLES: 'false'`
2. **Added Python packages**: `pandas scikit-learn kneed requests`
3. **Mounted working_data volume**: For model persistence
4. **Updated admin credentials**: `airflow2` / `airflow2`

## Technologies Used

- **Apache Airflow 3.x** — Workflow orchestration
- **Docker & Docker Compose** — Containerized deployment
- **Python 3.12** — Programming language
- **scikit-learn** — Machine learning (KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture, PCA)
- **pandas** — Data manipulation
- **kneed** — Elbow method for optimal cluster detection
- **Pydantic** — Data serialization via JSON for XCom

## Key Learnings

- Airflow 3.x has breaking changes from 2.x (import paths, `schedule` vs `schedule_interval`, no `provide_context`, JSON-only XCom)
- XCom in Airflow 3.x requires JSON-serializable data — numpy types must be explicitly converted
- `.airflowignore` prevents Airflow from scanning utility modules as DAGs
- BranchPythonOperator enables conditional pipeline execution
- `trigger_rule='none_failed_min_one_success'` allows tasks to run after branching

## Author

Kshitij
