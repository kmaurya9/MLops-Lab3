import pandas as pd
import pickle
import json
import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


# ============================================================
# TASK 1: Load Data
# ============================================================
def load_data():
    """
    Loads the Wine dataset from a CSV file and returns it as JSON
    for XCom serialization.
    """
    data_path = '/opt/airflow/dags/src/wine.csv'

    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First 5 rows:\n{df.head()}")

    return df.to_json(orient='split')


# ============================================================
# TASK 2: Validate Data
# ============================================================
def validate_data(data):
    """
    Validates the loaded data by checking for missing values,
    duplicates, and basic statistical properties.
    Returns JSON data if validation passes.
    """
    df = pd.read_json(data, orient='split')

    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    # Check minimum row count
    min_rows = 50
    if len(df) < min_rows:
        raise ValueError(f"Dataset too small: {len(df)} rows (minimum: {min_rows})")

    # Check for numeric columns (excluding target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numeric columns: {len(numeric_cols)}")

    # Check for zero variance columns
    zero_var_cols = [col for col in numeric_cols if df[col].std() == 0]
    if zero_var_cols:
        print(f"WARNING: Zero variance columns found: {zero_var_cols}")

    # Remove duplicates if found
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows. New shape: {df.shape}")

    print("Data validation PASSED!")
    return df.to_json(orient='split')


# ============================================================
# TASK 3: Data Preprocessing
# ============================================================
def data_preprocessing(data):
    """
    Performs data preprocessing: drops target column, scales features,
    then reduces dimensions with PCA (keeping 95% of the variance).
    Returns preprocessed data as JSON.
    """
    df = pd.read_json(data, orient='split')

    # Drop target column
    columns_to_drop = ['target', 'Target', 'class', 'species', 'Species']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped column: {col}")

    print(f"Features used for clustering: {list(df.columns)}")

    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    print(f"Feature means after scaling: {scaled_data.mean(axis=0).round(2)}")

    # Reduce dimensions with PCA, keeping enough components for 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    pca_data = pca.fit_transform(scaled_data)
    pca_columns = [f'PC{i + 1}' for i in range(pca_data.shape[1])]
    pca_df = pd.DataFrame(pca_data, columns=pca_columns)

    print(f"PCA kept {pca.n_components_} of {df.shape[1]} dimensions "
          f"({pca.explained_variance_ratio_.sum() * 100:.1f}% variance explained)")
    print(f"Preprocessed data shape: {pca_df.shape}")

    return pca_df.to_json(orient='split')


# ============================================================
# TASK 4a: Train K-Means (Parallel Task 1)
# ============================================================
def train_kmeans(data):
    """
    Trains K-Means models for k=2 to k=10, finds optimal k using
    elbow method, trains final model, and returns results.
    """
    df = pd.read_json(data, orient='split')
    save_dir = '/opt/airflow/working_data'

    sse = []
    # Try K-Means for k = 2 to 10
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
        print(f"K={k}, SSE={kmeans.inertia_:.2f}")

    # Find optimal k using elbow method
    kl = KneeLocator(range(2, 11), sse, curve='convex', direction='decreasing')
    optimal_k = int(kl.elbow) if kl.elbow else 3
    print(f"Optimal K (elbow): {optimal_k}")

    # Train final model with optimal k
    final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(df)

    # Calculate silhouette score
    sil_score = silhouette_score(df, labels)
    print(f"K-Means Silhouette Score: {sil_score:.4f}")

    # Save model
    model_path = os.path.join(save_dir, 'kmeans_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"K-Means model saved to {model_path}")

    results = {
        'model_name': 'KMeans',
        'n_clusters': optimal_k,
        'silhouette_score': sil_score,
        'sse_values': sse,
        'model_path': model_path
    }
    return json.dumps(results)


# ============================================================
# TASK 4b: Train DBSCAN (Parallel Task 2)
# ============================================================
def train_dbscan(data):
    """
    Trains DBSCAN clustering model with multiple eps values,
    selects the best configuration, and returns results.
    """
    df = pd.read_json(data, orient='split')
    save_dir = '/opt/airflow/working_data'

    best_score = -1
    best_eps = None
    best_labels = None
    best_model = None

    # Try different eps values
    for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(df)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if n_clusters >= 2:
            score = silhouette_score(df, labels)
            print(f"DBSCAN eps={eps}: clusters={n_clusters}, noise={n_noise}, silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_eps = eps
                best_labels = labels
                best_model = dbscan
        else:
            print(f"DBSCAN eps={eps}: clusters={n_clusters}, noise={n_noise} (skipped - need >= 2 clusters)")

    if best_model is None:
        print("DBSCAN could not find a valid configuration")
        results = {
            'model_name': 'DBSCAN',
            'n_clusters': 0,
            'silhouette_score': -1,
            'best_eps': None,
            'model_path': None
        }
        return json.dumps(results)

    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    print(f"Best DBSCAN: eps={best_eps}, clusters={n_clusters}, silhouette={best_score:.4f}")

    # Save model
    model_path = os.path.join(save_dir, 'dbscan_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"DBSCAN model saved to {model_path}")

    results = {
        'model_name': 'DBSCAN',
        'n_clusters': n_clusters,
        'silhouette_score': best_score,
        'best_eps': best_eps,
        'model_path': model_path
    }
    return json.dumps(results)


# ============================================================
# TASK 4c: Train Agglomerative Clustering (Parallel Task 3)
# ============================================================
def train_agglomerative(data):
    """
    Trains Agglomerative Clustering models with different n_clusters,
    selects the best, and returns results.
    """
    df = pd.read_json(data, orient='split')
    save_dir = '/opt/airflow/working_data'

    best_score = -1
    best_k = None
    best_model = None

    # Try different cluster counts
    for k in range(2, 11):
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(df)
        score = silhouette_score(df, labels)
        print(f"Agglomerative k={k}: silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = agg

    print(f"Best Agglomerative: k={best_k}, silhouette={best_score:.4f}")

    # Save model
    model_path = os.path.join(save_dir, 'agglomerative_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Agglomerative model saved to {model_path}")

    results = {
        'model_name': 'Agglomerative',
        'n_clusters': best_k,
        'silhouette_score': best_score,
        'model_path': model_path
    }
    return json.dumps(results)


# ============================================================
# TASK 4d: Train Gaussian Mixture Model (Parallel Task 4)
# ============================================================
def train_gmm(data):
    """
    Trains Gaussian Mixture Models with different n_components,
    selects the best one by BIC (lower is better), and returns results.
    """
    df = pd.read_json(data, orient='split')
    save_dir = '/opt/airflow/working_data'

    best_bic = None
    best_k = None
    best_model = None

    # Try different component counts
    for k in range(2, 11):
        gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=3, random_state=42)
        gmm.fit(df)
        bic = gmm.bic(df)
        print(f"GMM k={k}: BIC={bic:.2f}")

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = gmm

    labels = best_model.predict(df)
    sil_score = silhouette_score(df, labels)
    print(f"Best GMM: k={best_k}, BIC={best_bic:.2f}, silhouette={sil_score:.4f}")

    # Save model
    model_path = os.path.join(save_dir, 'gmm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"GMM model saved to {model_path}")

    results = {
        'model_name': 'GaussianMixture',
        'n_clusters': best_k,
        'silhouette_score': float(sil_score),
        'bic': float(best_bic),
        'model_path': model_path
    }
    return json.dumps(results)


# ============================================================
# TASK 5: Compare Models
# ============================================================
def compare_models(kmeans_result, dbscan_result, agglomerative_result, gmm_result):
    """
    Compares all four clustering models based on silhouette scores
    and selects the best one. Returns comparison results.
    """
    kmeans = json.loads(kmeans_result)
    dbscan = json.loads(dbscan_result)
    agglomerative = json.loads(agglomerative_result)
    gmm = json.loads(gmm_result)

    models = [kmeans, dbscan, agglomerative, gmm]

    print("=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    for m in models:
        print(f"  {m['model_name']:20s} | Clusters: {m['n_clusters']:3d} | Silhouette: {m['silhouette_score']:.4f}")
    print("=" * 60)

    # Find best model (highest silhouette score)
    best = max(models, key=lambda x: x['silhouette_score'])
    print(f"\nBest Model: {best['model_name']} (silhouette: {best['silhouette_score']:.4f})")

    comparison = {
        'all_results': models,
        'best_model': best['model_name'],
        'best_score': best['silhouette_score'],
        'best_model_path': best['model_path'],
        'quality': 'high' if best['silhouette_score'] > 0.3 else 'low'
    }
    return json.dumps(comparison)


# ============================================================
# TASK 6: Branch Decision
# ============================================================
def decide_quality(**kwargs):
    """
    BranchPythonOperator callable: decides whether model quality
    is high or low based on the comparison results.
    """
    ti = kwargs['ti']
    comparison_json = ti.xcom_pull(task_ids='compare_models_task')
    comparison = json.loads(comparison_json)

    quality = comparison['quality']
    print(f"Model quality: {quality} (score: {comparison['best_score']:.4f})")

    if quality == 'high':
        return 'high_quality_task'
    else:
        return 'low_quality_task'


# ============================================================
# TASK 7a: High Quality Path
# ============================================================
def high_quality_action(**kwargs):
    """
    Executes when model quality is high.
    Saves the best model as the production model.
    """
    ti = kwargs['ti']
    comparison_json = ti.xcom_pull(task_ids='compare_models_task')
    comparison = json.loads(comparison_json)

    best_model_path = comparison['best_model_path']
    prod_path = '/opt/airflow/working_data/production_model.pkl'

    # Copy best model as production model
    with open(best_model_path, 'rb') as f:
        model = pickle.load(f)
    with open(prod_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"HIGH QUALITY: {comparison['best_model']} promoted to production!")
    print(f"Production model saved to: {prod_path}")
    print(f"Silhouette score: {comparison['best_score']:.4f}")


# ============================================================
# TASK 7b: Low Quality Path
# ============================================================
def low_quality_action(**kwargs):
    """
    Executes when model quality is low.
    Retrains K-Means with a wider range of clusters.
    """
    ti = kwargs['ti']
    comparison_json = ti.xcom_pull(task_ids='compare_models_task')
    comparison = json.loads(comparison_json)

    print(f"LOW QUALITY: Best score was only {comparison['best_score']:.4f}")
    print("Retraining K-Means with extended cluster range (2-20)...")

    # Load preprocessed data from preprocessing task
    data_json = ti.xcom_pull(task_ids='data_preprocessing_task')
    df = pd.read_json(data_json, orient='split')

    best_score = -1
    best_k = None
    best_model = None

    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df)
        score = silhouette_score(df, labels)
        print(f"  K={k}, silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans

    prod_path = '/opt/airflow/working_data/production_model.pkl'
    with open(prod_path, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"Retrained model (k={best_k}) saved as production model")
    print(f"New silhouette score: {best_score:.4f}")


# ============================================================
# TASK 8: Notify
# ============================================================
def notify_completion(**kwargs):
    """
    Final task that summarizes the pipeline results.
    """
    ti = kwargs['ti']
    comparison_json = ti.xcom_pull(task_ids='compare_models_task')
    comparison = json.loads(comparison_json)

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best Model: {comparison['best_model']}")
    print(f"Best Score: {comparison['best_score']:.4f}")
    print(f"Quality: {comparison['quality']}")
    print(f"Production model saved to: /opt/airflow/working_data/production_model.pkl")
    print("=" * 60)
