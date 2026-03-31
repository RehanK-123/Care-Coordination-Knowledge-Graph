from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── your existing modules ────────────────────────────────────────────────────
from VectorDB import (
    get_clinical_embeddings,
    get_financial_embeddings,
    get_behavioural_embeddings,
    get_hybrid_embeddings,
)
from Ingestion import extract_and_save_data
import pickle as pkl

# ── Qdrant ───────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
INTERMEDIATE_DIR = Path("output/intermediate")
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
FHIR_JSON_PATH = "synthea_sample_data_fhir_latest/Patient_100000.json"  # example path
COLLECTIONS = {
    "clinical": None,        # dim set on first upsert
    "financial": None,
    "behavioural": None,
    "hybrid": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# Return types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Returned by extract_and_save_patient_data()."""
    patient_ids: list[str]
    newly_processed: list[str]       # patients that were NOT in Qdrant and got embedded
    already_cached: list[str]        # patients that were already in Qdrant (reused)
    embeddings: dict[str, np.ndarray]  # space → (n_patients, dim)
    saved_paths: dict[str, Path]


@dataclass
class FinancialRiskResult:
    """Returned by cluster_financial_risk()."""
    patient_ids: list[str]
    labels: dict[str, str]           # patient_id → "High Risk" | "Low Risk"
    cluster_summary: pd.DataFrame
    metrics: dict
    embeddings_used: np.ndarray


@dataclass
class CareCoordinationResult:
    """Returned by classify_care_coordination()."""
    patient_ids: list[str]
    labels: dict[str, str]           # patient_id → "Optimal Care" | "Non-Optimal Care"
    optimal_cluster: int
    cluster_summary: pd.DataFrame
    embeddings_used: np.ndarray


@dataclass
class SimilarityResult:
    """Returned by find_similar_patients()."""
    query_patient: str
    embedding_space: str
    similar_patients: list[dict]     # [{patient_id, similarity, rank}]
    metadata: dict


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _patients_in_qdrant(
    client: QdrantClient,
    collection: str,
    patient_ids: list[str],
) -> set[str]:
    """
    Return the subset of patient_ids already stored in the collection.
    Uses scroll with a payload filter; falls back gracefully if the
    collection doesn't exist yet.
    """
    try:
        existing = set()
        for pid in patient_ids:
            results, _ = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="patient_id", match=MatchValue(value=pid))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if results:
                existing.add(pid)
        return existing
    except Exception:
        return set()



def _fetch_embeddings_from_qdrant(
    client: QdrantClient,
    collection: str,
    patient_ids: list[str],
) -> dict[str, np.ndarray]:
    """Retrieve stored vectors for given patient_ids (returns {pid: vector})."""
    vectors = {}
    for pid in patient_ids:
        results, _ = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="patient_id", match=MatchValue(value=pid))]
            ),
            limit=1,
            with_vectors=True,
        )
        if results:
            vectors[pid] = np.array(results[0].vector)
    return vectors


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE EXTRACTION + QDRANT UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
def extract_and_save_patient_data(fhir_json=FHIR_JSON_PATH, output_dir=INTERMEDIATE_DIR):
    patients, providers, diseases, encounters, claims, payers = extract_and_save_data(fhir_json= fhir_json, output_dir=output_dir)
    return {"patients": patients, "providers": providers, "diseases": diseases, "encounters": encounters, "claims": claims, "payers": payers}

# ─────────────────────────────────────────────────────────────────────────────
# 2. FINANCIAL RISK CLUSTERING  (High Risk / Low Risk)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_financial_risk(
    patients: pd.DataFrame,
    claims: pd.DataFrame,
    encounters: pd.DataFrame,
    n_clusters: int = 2,
    dbscan_eps: float = 1.5,
    dbscan_min_samples: int = 5,
    outlier_z_threshold: float = 2.5,
    diseases: Optional[pd.DataFrame] = None,
    precomputed_embeddings: Optional[np.ndarray] = None,
) -> FinancialRiskResult:

    client = _get_qdrant_client()
    all_pids = patients["patient_id"].tolist()
    fin_emb = precomputed_embeddings if precomputed_embeddings is not None else get_financial_embeddings(patients, encounters, claims)
    # ── Scale ────────────────────────────────────────────────────────────────
    fin_scaler = pkl.load(open("models/financial_scaler.pkl", "rb"))
    fin_scaled = fin_scaler.fit_transform(fin_emb)

    # ── KMeans clustering ────────────────────────────────────────────────────
    km = pkl.load(open("models/financial_kmeans.pkl", "rb"))
    cluster_labels = km.predict(fin_scaled.astype(np.float32))
    # sil = silhouette_score(fin_scaled, cluster_labels) if n_clusters > 1 else None

    # ── Identify which KMeans cluster is "High Risk" ─────────────────────────
    # Build a lightweight financial feature table to compare clusters
    fin_features = pd.DataFrame({
        "patient_id": all_pids,
        "num_claims":   claims.groupby("patient_id").size().reindex(all_pids).fillna(0).values,
        "total_spend":  claims.groupby("patient_id")["amount"].sum().reindex(all_pids).fillna(0).values,
        "visits":       encounters.groupby("patient_id").size().reindex(all_pids).fillna(0).values,
        "cluster":      cluster_labels,
    })

    cluster_means = fin_features.groupby("cluster")[["num_claims", "total_spend", "visits"]].mean()
    risk_score_per_cluster = cluster_means["total_spend"] + cluster_means["num_claims"] * 100
    high_risk_cluster = int(risk_score_per_cluster.idxmax())
    # ── DBSCAN outlier boost ─────────────────────────────────────────────────
    db = pkl.load(open("models/financial_dbscan.pkl", "rb"))
    db_labels = db.fit_predict(fin_scaled)
    dbscan_outliers = set(np.array(all_pids)[db_labels == -1])

    # Fallback: z-score outliers if DBSCAN finds nothing
    if not dbscan_outliers:
        centroid = fin_scaled.mean(axis=0)
        dists = np.linalg.norm(fin_scaled - centroid, axis=1)
        z = (dists - dists.mean()) / (dists.std() + 1e-9)
        dbscan_outliers = set(np.array(all_pids)[z > outlier_z_threshold])

    # ── Assign labels ────────────────────────────────────────────────────────
    risk_labels: dict[str, str] = {}
    for pid, cl in zip(all_pids, cluster_labels):
        if cl == high_risk_cluster or pid in dbscan_outliers:
            risk_labels[pid] = "High Risk"
        else:
            risk_labels[pid] = "Low Risk"

    # ── Cluster summary ──────────────────────────────────────────────────────
    fin_features["risk_label"] = fin_features["patient_id"].map(risk_labels)
    summary = fin_features.groupby("risk_label")[["num_claims", "total_spend", "visits"]].agg(
        ["mean", "median", "std", "count"]
    ).round(2)

    # ── Store risk labels back to Qdrant payload ─────────────────────────────
    for pid, label in risk_labels.items():
        client.set_payload(
            collection_name="financial",
            payload={"risk_label": label},
            points=Filter(
                must=[FieldCondition(key="patient_id", match=MatchValue(value=pid))]
            ),
        )

    # print(f"[financial_risk] High Risk: {sum(v=='High Risk' for v in risk_labels.values())} | "
    #       f"Low Risk: {sum(v=='Low Risk' for v in risk_labels.values())} | "
    #       f"Silhouette: {round(sil, 4) if sil else 'N/A'}")

    return FinancialRiskResult(
        patient_ids=all_pids,
        labels=risk_labels,
        cluster_summary=summary,
        metrics={
            # "silhouette": round(sil, 4) if sil else None,
            "high_risk_cluster": high_risk_cluster,
            "dbscan_outliers_flagged": len(dbscan_outliers),
            "inertia": round(km.inertia_, 2),
        },
        embeddings_used=fin_emb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARE COORDINATION CLASSIFICATION  (Optimal / Non-Optimal)
# ─────────────────────────────────────────────────────────────────────────────

def classify_care_coordination(
    patients: pd.DataFrame,
    diseases: pd.DataFrame,
    encounters: pd.DataFrame,
    claims: Optional[pd.DataFrame] = None,
    precomputed_embeddings: Optional[np.ndarray] = None,
    n_clusters: int = 2
) -> CareCoordinationResult:
    
    client = _get_qdrant_client()
    all_pids = patients["patient_id"].tolist()

    clin_emb = precomputed_embeddings if precomputed_embeddings is not None else get_clinical_embeddings(patients, diseases, encounters, claims)
    # ── Scale + cluster ──────────────────────────────────────────────────────
    clin_scaler = pkl.load(open("models/clinical_scaler.pkl", "rb"))
    clin_scaled = clin_scaler.fit_transform(clin_emb)

    km = pkl.load(open("models/care_coordination_km.pkl", "rb"))
    cluster_labels = km.predict(clin_scaled.astype(np.float64))

    # ── Build behavioural attributes for cluster profiling ───────────────────
    encounters_copy = encounters.copy()
    encounters_copy["period_start"] = pd.to_datetime(encounters_copy["period_start"], errors="coerce", utc=True)

    care_coord = encounters_copy.groupby("patient_id")["provider_id"].nunique().reindex(all_pids).fillna(0)
    visit_cnt  = encounters_copy.groupby("patient_id").size().reindex(all_pids).fillna(0)
    over_util  = (visit_cnt > visit_cnt.mean() + 2 * visit_cnt.std()).astype(int)

    attr_df = pd.DataFrame({
        "patient_id":        all_pids,
        "care_coordination": care_coord.values,
        "over_utilization":  over_util.values,
        "visit_count":       visit_cnt.values,
        "cluster":           cluster_labels,
    })

    # ── Determine optimal cluster ────────────────────────────────────────────
    # Optimal = high care_coordination AND low over_utilization
    cluster_means = attr_df.groupby("cluster")[["care_coordination", "over_utilization", "visit_count"]].mean()
    # Score: maximise coordination, penalise over-utilization
    optimality_score = cluster_means["care_coordination"] - cluster_means["over_utilization"] * 5
    optimal_cluster = int(optimality_score.idxmax())

    # ── Assign labels ────────────────────────────────────────────────────────
    care_labels: dict[str, str] = {
        pid: ("Optimal Care" if cl == optimal_cluster else "Non-Optimal Care")
        for pid, cl in zip(all_pids, cluster_labels)
    }

    # ── Cluster summary ──────────────────────────────────────────────────────
    attr_df["care_label"] = attr_df["patient_id"].map(care_labels)
    summary = attr_df.groupby("care_label")[["care_coordination", "over_utilization", "visit_count"]].agg(
        ["mean", "median", "count"]
    ).round(3)

    # ── Persist labels to Qdrant ─────────────────────────────────────────────
    for pid, label in care_labels.items():
        client.set_payload(
            collection_name="clinical",
            payload={"care_label": label},
            points=Filter(
                must=[FieldCondition(key="patient_id", match=MatchValue(value=pid))]
            ),
        )

    # sil = silhouette_score(clin_scaled, cluster_labels) if n_clusters > 1 else None
    # opt_count = sum(v == "Optimal Care"     for v in care_labels.values())
    # non_opt_count = sum(v == "Non-Optimal Care" for v in care_labels.values())
    # print(f"[care_coordination] Optimal: {opt_count} | Non-Optimal: {non_opt_count} | "
    #       f"Silhouette: {round(sil, 4) if sil else 'N/A'}")

    return CareCoordinationResult(
        patient_ids=all_pids,
        labels=care_labels,
        optimal_cluster=optimal_cluster,
        cluster_summary=summary,
        embeddings_used=clin_emb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. PATIENT SIMILARITY  (cosine search, any embedding space)
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_patients(
    query_patient_id: str,
    patients: pd.DataFrame,
    diseases: Optional[pd.DataFrame] = None,
    claims: Optional[pd.DataFrame] = None,
    encounters: Optional[pd.DataFrame] = None,
    embedding_space: str = "hybrid",
    top_k: int = 10,
    precomputed_embeddings: Optional[np.ndarray] = None,
    similarity_threshold: float = 0.0,
) -> SimilarityResult:
    
    if embedding_space not in ["clinical", "financial", "behavioural", "hybrid"]:
        raise ValueError(f"embedding_space must be one of clinical/financial/behavioural/hybrid, got '{embedding_space}'")

    client = _get_qdrant_client()
    all_pids = patients["patient_id"].tolist()

    if query_patient_id not in all_pids:
        raise ValueError(f"query_patient_id '{query_patient_id}' not found in patients DataFrame.")
    
    emb = precomputed_embeddings if precomputed_embeddings is not None else get_hybrid_embeddings(
        get_clinical_embeddings(patients, diseases, encounters, claims),
        get_financial_embeddings(patients, claims, encounters),
        get_behavioural_embeddings(patients, encounters)
    )

    # ── Cosine similarity ────────────────────────────────────────────────────
    query_idx = all_pids.index(query_patient_id)
    query_vec = emb[query_idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, emb)[0]
    sims[query_idx] = -1  # exclude self

    # Apply threshold
    valid_mask = sims >= similarity_threshold
    valid_indices = np.where(valid_mask)[0]
    ranked = valid_indices[np.argsort(sims[valid_indices])[::-1]][:top_k]

    similar = [
        {
            "patient_id": all_pids[i],
            "similarity":  round(float(sims[i]), 4),
            "rank":        rank + 1,
        }
        for rank, i in enumerate(ranked)
    ]

    print(f"[similarity] Query: {query_patient_id} | Space: {embedding_space} | "
          f"Top-{top_k} results | Max sim: {similar[0]['similarity'] if similar else 'N/A'}")

    return SimilarityResult(
        query_patient=query_patient_id,
        embedding_space=embedding_space,
        similar_patients=similar,
        metadata={
            "n_corpus": len(all_pids),
            "embedding_dim": emb.shape[1],
            "similarity_threshold": similarity_threshold,
            "mean_similarity": round(float(np.mean([s["similarity"] for s in similar])), 4) if similar else 0,
        },
    )



