import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras import layers, models
import fasttext
import numpy as np
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# -----------------------------
# LOAD DATA
# -----------------------------
claims = pd.read_csv("output/claims.csv")
diseases = pd.read_csv("output/diseases.csv")
encounters = pd.read_csv("output/encounters.csv")
patients = pd.read_csv("output/patients.csv")
payers = pd.read_csv("output/payers.csv")

master_index = patients["patient_id"]
# -----------------------------
# CLINICAL EMBEDDINGS (FastText) #ClinicalBERT would be ideal but is resource intensive
# -----------------------------
import pandas as pd
import numpy as np
import fasttext
from collections import Counter
from sklearn.preprocessing import StandardScaler

# ================================
# SETUP
# ================================

diseases["start"] = pd.to_datetime(diseases["start"], errors="coerce", utc=True)
today = pd.Timestamp.today(tz="UTC")

# ================================
# DISEASE HIERARCHY
# ================================
DISEASE_HIERARCHY = {
    "diabetes": "metabolic",
    "hypertension": "cardiovascular",
    "asthma": "respiratory",
    "depression": "mental_health",
    "infection": "acute",
}

def map_category(disease):
    disease = str(disease).lower()
    for key in DISEASE_HIERARCHY:
        if key in disease:
            return DISEASE_HIERARCHY[key]
    return "other"

# ================================
# CHRONIC SCORE
# ================================
def compute_chronic_score(group):
    dates = group["start"].dropna().sort_values()

    if len(dates) == 0:
        return np.zeros(len(group))

    last_date = pd.to_datetime(dates.max())
    days_since_last = (today - last_date).days
    recency = np.exp(-0.01 * days_since_last)

    frequency = len(group)
    duration = (pd.to_datetime(dates.max()) - pd.to_datetime(dates.min())).days if len(dates) > 1 else 0

    persistence = duration / 365

    frequency = min(frequency / 10, 1)
    persistence = min(persistence, 1)

    chronic_score = 0.4 * recency + 0.3 * frequency + 0.3 * persistence

    return np.full(len(group), chronic_score)

# ================================
# FASTTEXT
# ================================
model = fasttext.load_model("cc.en.300.bin")
embedding_dim = model.get_dimension()

def embed_text(texts):
    return np.array([model.get_sentence_vector(str(t)) for t in texts])

# ================================
# MAIN PIPELINE
# ================================
def get_clinical_embeddings(diseases, patients):
    patient_vectors = {}
    for pid, group in diseases.groupby("patient_id"):
        group = group.copy()

        # ---- TEXT EMBEDDINGS ----
        disease_texts = group["disease"].astype(str).tolist()
        text_embs = embed_text(disease_texts) if disease_texts else np.zeros((1, embedding_dim))

        # ---- CHRONIC SCORE ----
        group["chronic_score"] = compute_chronic_score(group)

        weights = group["chronic_score"].values
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)

        weighted_emb = np.average(text_embs, axis=0, weights=weights)

        # ---- HIERARCHY ----
        group["category"] = group["disease"].apply(map_category)
        category_counts = Counter(group["category"])

        # ---- CHRONIC VS ACUTE ----
        chronic_mask = group["chronic_score"] > 0.5
        num_conditions = len(group)
        num_chronic = chronic_mask.sum()
        num_acute = num_conditions - num_chronic

        chronic_ratio = num_chronic / num_conditions if num_conditions else 0
        acute_ratio = num_acute / num_conditions if num_conditions else 0

        # ---- TEMPORAL ----
        dates = group["start"].dropna().sort_values()
        duration_days = ((pd.to_datetime(dates.max())) - pd.to_datetime(dates.min())).days if len(dates) > 1 else 0
        avg_duration = duration_days / max(num_conditions, 1)

        recent_6m = ((today - pd.to_datetime(group["start"], errors="coerce", utc= True)).dt.days <= 180).sum()

        # ---- STRUCTURED FEATURES ----
        structured_features = np.array([
            num_conditions,
            num_chronic,
            num_acute,
            chronic_ratio,
            acute_ratio,
            avg_duration,
            recent_6m,
            category_counts.get("cardiovascular", 0),
            category_counts.get("metabolic", 0),
            category_counts.get("respiratory", 0),
            category_counts.get("mental_health", 0),
            category_counts.get("acute", 0),
        ])

        # ---- FINAL VECTOR ----

        final_vector = np.concatenate([weighted_emb, structured_features])

        patient_vectors[pid] = final_vector

    # ================================
    # CONVERT TO DATAFRAME (KEY STEP)
    # ================================
    clinical_df = pd.DataFrame.from_dict(patient_vectors, orient="index")

    # 👉 IMPORTANT: align with patients table
    clinical_df = clinical_df.reindex(patients["patient_id"]).fillna(0)

    # ================================
    # NORMALIZE
    # ================================
    scaler = StandardScaler()
    clinical_emb = scaler.fit_transform(clinical_df)

    return clinical_emb
# -----------------------------
# NUMERIC FEATURES (Autoencoder)
# -----------------------------
# Example: combine financial + behavioural + demographics
def get_financial_embeddings(patients, encounters, claims):
    financial_df = pd.DataFrame(index=patients["patient_id"])

    financial_df["age"] = (
        (pd.Timestamp.today() - pd.to_datetime(patients["birth_date"], errors="coerce"))
        .dt.days // 365
    ).values

    financial_df["num_claims"] = claims.groupby("patient_id").size()
    financial_df["total_spend"] = claims.groupby("patient_id")["amount"].sum()
    financial_df["visits"] = encounters.groupby("patient_id").size()

    # Align everything
    financial_df = financial_df.fillna(0)

    financial_df = financial_df.reindex(patients["patient_id"]).fillna(0) # Financial features aligned to patients

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(financial_df)

    input_dim = X_scaled.shape[1]
    encoding_dim = 16

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
    decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)
    financial_emb = encoder.predict(X_scaled)

    return financial_emb

# -----------------------------
# 🔷 BEHAVIOURAL FEATURES
# -----------------------------
# Care coordination: number of distinct providers involved
def get_behavioural_embeddings(patients, encounters):
    care_coordination = encounters.groupby("patient_id")["provider_id"].nunique()

    # Over-utilization detection: flag patients with very high visit frequency
    visit_counts = encounters.groupby("patient_id").size()
    over_utilization = (visit_counts > visit_counts.mean() + 2*visit_counts.std()).astype(int)

    # Patient behavior patterns: average gap between visits, variability in gaps
    encounters["period_start"] = pd.to_datetime(encounters["period_start"], errors="coerce", utc= True)
    gap_features = []
    for pid, group in encounters.sort_values("period_start").groupby("patient_id"):
        times = group["period_start"].dropna().sort_values()
        gaps = times.diff().dt.days.dropna()
        gap_features.append({
            "patient_id": pid,
            "avg_gap": gaps.mean() if len(gaps) > 0 else 0,
            "std_gap": gaps.std() if len(gaps) > 0 else 0
        })
    gap_df = pd.DataFrame(gap_features).set_index("patient_id")

    behavioural_df = pd.DataFrame(index=patients["patient_id"])

    behavioural_df["care_coordination"] = care_coordination
    behavioural_df["over_utilization"] = over_utilization

    # Join gap features
    behavioural_df = behavioural_df.join(gap_df)

    # Fill missing patients (no encounters)
    behavioural_df = behavioural_df.fillna(0)

    behavioural_df = behavioural_df.reindex(patients["patient_id"]).fillna(0)
    # -----------------------------
    # 🔷 BEHAVIOURAL EMBEDDINGS (Autoencoder)
    # -----------------------------
    scaler_beh = StandardScaler()
    X_beh_scaled = scaler_beh.fit_transform(behavioural_df)

    input_dim_beh = X_beh_scaled.shape[1]
    encoding_dim_beh = 8

    input_layer_beh = layers.Input(shape=(input_dim_beh,))
    encoded_beh = layers.Dense(encoding_dim_beh, activation="relu")(input_layer_beh)
    decoded_beh = layers.Dense(input_dim_beh, activation="sigmoid")(encoded_beh)

    autoencoder_beh = models.Model(input_layer_beh, decoded_beh)
    encoder_beh = models.Model(input_layer_beh, encoded_beh)

    autoencoder_beh.compile(optimizer="adam", loss="mse")
    autoencoder_beh.fit(X_beh_scaled, X_beh_scaled, epochs=50, batch_size=32, verbose=0)
    
    behavioural_emb = encoder_beh.predict(X_beh_scaled)

    return behavioural_emb

# -----------------------------
# HYBRID EMBEDDINGS
# -----------------------------
def get_hybrid_embeddings(clinical_emb, financial_emb, behavioural_emb):
    hybrid_emb = np.hstack([clinical_emb, financial_emb, behavioural_emb])

    scaler_hybrid = StandardScaler()
    hybrid_emb_scaled = scaler_hybrid.fit_transform(hybrid_emb)

    return hybrid_emb_scaled
clinical_emb = get_clinical_embeddings(diseases, patients)
financial_emb = get_financial_embeddings(patients, encounters, claims)
behavioural_emb = get_behavioural_embeddings(patients, encounters)


hybrid_emb = get_hybrid_embeddings(clinical_emb, financial_emb, behavioural_emb)

# -----------------------------
# CLUSTERING EVALUATION
# -----------------------------
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(X, k_range=(2, 10)):
    results = []

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        inertia = kmeans.inertia_

        results.append({
            "k": k,
            "silhouette": sil_score,
            "davies_bouldin": db_score,
            "inertia": inertia
        })

    return results
# -----------------------------
# CLUSTERING
# -----------------------------
print(evaluate_clustering(behavioural_emb, k_range=(2, 7)))  # Example for financial embeddings
kmeans = KMeans(n_clusters=2, random_state=42)
clusters= kmeans.fit_predict(behavioural_emb)

print("Cluster distribution:", np.bincount(clusters))
# -----------------------------
# QDRANT SETUP
# -----------------------------
client = QdrantClient(host="localhost", port=6333)

collections = {
    "clinical": (clinical_emb, clinical_emb.shape[1]),
    "financial": (financial_emb, financial_emb.shape[1]),
    "behavioural": (behavioural_emb, behavioural_emb.shape[1]),
    "hybrid": (hybrid_emb, hybrid_emb.shape[1])
}

for name, (_, dim) in collections.items():
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

# -----------------------------
# INSERT INTO QDRANT
# -----------------------------
for name, (embeddings, _) in collections.items():
    points = []
    for idx, patient_id in enumerate(patients["patient_id"]):
        unique_id = hash(patient_id)
    
        points.append(
            PointStruct(
                id=unique_id if unique_id >= 0 else -unique_id,
                vector=embeddings[idx].tolist(),
                payload= {
                    "patient_id": patient_id,
                    "visits_cnt": int(encounters[encounters["patient_id"] == patient_id].shape[0])
                }
            )
        )
    client.upload_points(collection_name=name, points=points, batch_size=500)

print("✅ Clinical, financial, behavioural, and hybrid embeddings stored in Qdrant!")
