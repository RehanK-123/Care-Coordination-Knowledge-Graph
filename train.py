from pathlib import Path
import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from VectorDB import (
    get_clinical_embeddings,
    get_financial_embeddings,
    get_behavioural_embeddings,
    get_hybrid_embeddings,
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DATA_DIR = Path("output")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Load stored tables
# ─────────────────────────────────────────────

def load_tables():
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    diseases = pd.read_csv(DATA_DIR / "diseases.csv")
    encounters = pd.read_csv(DATA_DIR / "encounters.csv")
    claims = pd.read_csv(DATA_DIR / "claims.csv")

    return patients, diseases, encounters, claims


# ─────────────────────────────────────────────
# TRAIN FINANCIAL MODEL
# ─────────────────────────────────────────────

def train_financial_model():

    patients, diseases, encounters, claims = load_tables()

    fin_emb = get_financial_embeddings(patients, encounters, claims)

    scaler = StandardScaler()
    fin_scaled = scaler.fit_transform(fin_emb)

    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    model.fit(fin_scaled)

    # ✅ Save BOTH scaler + model
    pkl.dump(scaler, open(MODEL_DIR / "financial_scaler.pkl", "wb"))
    pkl.dump(model, open(MODEL_DIR / "financial_kmeans.pkl", "wb"))

    print("✅ Financial model trained & saved")


# ─────────────────────────────────────────────
# TRAIN CARE COORDINATION MODEL
# ─────────────────────────────────────────────

def train_care_model():

    patients, diseases, encounters, claims = load_tables()

    clin_emb = get_clinical_embeddings(diseases, patients)

    scaler = StandardScaler()
    clin_scaled = scaler.fit_transform(clin_emb)

    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    model.fit(clin_scaled)

    pkl.dump(scaler, open(MODEL_DIR / "clinical_scaler.pkl", "wb"))
    pkl.dump(model, open(MODEL_DIR / "clinical_kmeans.pkl", "wb"))

    print("✅ Care coordination model trained & saved")


# ─────────────────────────────────────────────
# TRAIN HYBRID MODEL (OPTIONAL)
# ─────────────────────────────────────────────

def train_hybrid_model():

    patients, diseases, encounters, claims = load_tables()
    diseases["start"] = pd.to_datetime(diseases["start"], errors="coerce", utc=True) # Ensure datetime format for clinical embeddings

    clinical = get_clinical_embeddings(diseases, patients)
    financial = get_financial_embeddings(patients, encounters, claims)
    behavioural = get_behavioural_embeddings(patients, encounters)

    hybrid = get_hybrid_embeddings(clinical, financial, behavioural)

    scaler = StandardScaler()
    clinical_scaled = scaler.fit_transform(clinical)

    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    model.fit(clinical_scaled)

    pkl.dump(scaler, open(MODEL_DIR / "care_coordination_scaler.pkl", "wb"))
    pkl.dump(model, open(MODEL_DIR / "care_coordination_kmeans.pkl", "wb"))

    scaler = StandardScaler()
    financial_scaled = scaler.fit_transform(financial)

    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    model.fit(financial_scaled)

    pkl.dump(scaler, open(MODEL_DIR / "financial_scaler.pkl", "wb"))
    pkl.dump(model, open(MODEL_DIR / "financial_kmeans.pkl", "wb"))




# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train_financial_model()
    train_care_model()
    train_hybrid_model()