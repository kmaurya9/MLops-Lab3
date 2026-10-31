# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_data,
    validate_data,
    data_preprocessing,
    train_kmeans,
    train_dbscan,
    train_agglomerative,
    train_gmm,
    compare_models,
    decide_quality,
    high_quality_action,
    low_quality_action,
    notify_completion,
)

# Define default arguments for the DAG
default_args = {
    'owner': 'kshitij',
    'start_date': datetime(2026, 10, 30),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG instance
dag = DAG(
    dag_id='your_python_dag',
    default_args=default_args,
    description='Advanced ML Clustering Pipeline with Parallel Training & Branching',
    schedule=None,
    catchup=False,
)

# ============================================================
# Task 1: Load Data
# ============================================================
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# ============================================================
# Task 2: Validate Data
# ============================================================
validate_data_task = PythonOperator(
    task_id='validate_data_task',
    python_callable=validate_data,
    op_args=[load_data_task.output],
    dag=dag,
)

# ============================================================
# Task 3: Preprocess Data
# ============================================================
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[validate_data_task.output],
    dag=dag,
)

# ============================================================
# Task 4a: Train K-Means (Parallel)
# ============================================================
train_kmeans_task = PythonOperator(
    task_id='train_kmeans_task',
    python_callable=train_kmeans,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

# ============================================================
# Task 4b: Train DBSCAN (Parallel)
# ============================================================
train_dbscan_task = PythonOperator(
    task_id='train_dbscan_task',
    python_callable=train_dbscan,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

# ============================================================
# Task 4c: Train Agglomerative (Parallel)
# ============================================================
train_agglomerative_task = PythonOperator(
    task_id='train_agglomerative_task',
    python_callable=train_agglomerative,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

# ============================================================
# Task 4d: Train Gaussian Mixture (Parallel)
# ============================================================
train_gmm_task = PythonOperator(
    task_id='train_gmm_task',
    python_callable=train_gmm,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

# ============================================================
# Task 5: Compare Models (waits for all 4 training tasks)
# ============================================================
compare_models_task = PythonOperator(
    task_id='compare_models_task',
    python_callable=compare_models,
    op_args=[
        train_kmeans_task.output,
        train_dbscan_task.output,
        train_agglomerative_task.output,
        train_gmm_task.output,
    ],
    dag=dag,
)

# ============================================================
# Task 6: Branch based on model quality
# ============================================================
branch_task = BranchPythonOperator(
    task_id='branch_quality_task',
    python_callable=decide_quality,
    dag=dag,
)

# ============================================================
# Task 7a: High Quality Path
# ============================================================
high_quality_task = PythonOperator(
    task_id='high_quality_task',
    python_callable=high_quality_action,
    dag=dag,
)

# ============================================================
# Task 7b: Low Quality Path
# ============================================================
low_quality_task = PythonOperator(
    task_id='low_quality_task',
    python_callable=low_quality_action,
    dag=dag,
)

# ============================================================
# Task 8: Notify Completion (runs after either branch)
# ============================================================
notify_task = PythonOperator(
    task_id='notify_task',
    python_callable=notify_completion,
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# ============================================================
# Set Task Dependencies
# ============================================================

# Linear: load → validate → preprocess
load_data_task >> validate_data_task >> data_preprocessing_task

# Parallel: preprocess fans out to 4 training tasks
data_preprocessing_task >> [train_kmeans_task, train_dbscan_task, train_agglomerative_task, train_gmm_task]

# All training tasks converge into compare
[train_kmeans_task, train_dbscan_task, train_agglomerative_task, train_gmm_task] >> compare_models_task

# Compare → Branch → High/Low quality paths
compare_models_task >> branch_task >> [high_quality_task, low_quality_task]

# Both paths converge to notify
[high_quality_task, low_quality_task] >> notify_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()
