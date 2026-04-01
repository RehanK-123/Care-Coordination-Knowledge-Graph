"""
Vector Database Tools for Care Coordination Knowledge Graph
============================================================
Thin wrappers around teammate 2's VectorDB code, making them
LLM-callable with simple inputs (patient name/ID, numbers).

Tools:
    - get_financial_risk: Classify a patient as High/Low financial risk
    - get_care_coordination: Classify a patient's care as Optimal/Non-Optimal
    - get_similar_patients: Find patients similar to a given patient
    - get_cluster_summary: Get overall clustering statistics
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path

# Add project root to path so we can import teammate 2's modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Data directory
DATA_DIR = PROJECT_ROOT / "output"


class VectorTools:
    """Tool functions for querying the Vector DB (Qdrant + ML models)."""

    def __init__(self):
        """Initialize by loading CSV data and importing teammate 2's modules."""
        # Load CSVs once (reused by all tools)
        self.patients = pd.read_csv(DATA_DIR / "patients.csv")
        self.diseases = pd.read_csv(DATA_DIR / "diseases.csv")
        self.encounters = pd.read_csv(DATA_DIR / "encounters.csv")
        self.claims = pd.read_csv(DATA_DIR / "claims.csv")
        self.payers = pd.read_csv(DATA_DIR / "payers.csv")

        # Ensure datetime format for clinical embeddings
        self.diseases["start"] = pd.to_datetime(
            self.diseases["start"], errors="coerce", utc=True
        )

        # Import teammate 2's modules
        try:
            from VectorDB_Helper import (
                cluster_financial_risk,
                classify_care_coordination,
                find_similar_patients,
            )
            self._cluster_financial_risk = cluster_financial_risk
            self._classify_care_coordination = classify_care_coordination
            self._find_similar_patients = find_similar_patients
            self._available = True
        except ImportError as e:
            print(f"⚠️  VectorDB modules not available: {e}")
            self._available = False

    def _resolve_patient_id(self, patient_name: str) -> Optional[str]:
        """Convert a patient name to patient_id using fuzzy match."""
        matches = self.patients[
            self.patients["name"].str.lower().str.contains(patient_name.lower(), na=False)
        ]
        if matches.empty:
            return None
        return matches.iloc[0]["patient_id"]

    def _get_patient_name(self, patient_id: str) -> str:
        """Convert patient_id to name."""
        match = self.patients[self.patients["patient_id"] == patient_id]
        if match.empty:
            return patient_id
        return match.iloc[0]["name"]

    # ═══════════════════════════════════════════════════════════════
    # TOOL 1: Financial Risk Classification
    # ═══════════════════════════════════════════════════════════════

    def get_financial_risk(self, patient_name: str) -> dict:
        """Classify a patient's financial risk level (High Risk / Low Risk).

        Uses ML clustering on financial embeddings (claims, spend, visits)
        plus DBSCAN outlier detection for anomaly flagging.

        Args:
            patient_name: Full or partial name of the patient.

        Returns:
            Risk label, risk details, and comparison metrics.
        """
        if not self._available:
            return {"error": "VectorDB modules not available. Ensure VectorDB_Helper.py and dependencies are installed."}

        patient_id = self._resolve_patient_id(patient_name)
        if not patient_id:
            return {"error": f"No patient found matching '{patient_name}'"}

        try:
            result = self._cluster_financial_risk(
                patients=self.patients,
                claims=self.claims,
                encounters=self.encounters,
            )

            risk_label = result.labels.get(patient_id, "Unknown")

            # Get patient-specific financial stats
            patient_claims = self.claims[self.claims["patient_id"] == patient_id]
            total_spend = patient_claims["amount"].sum() if not patient_claims.empty else 0
            num_claims = len(patient_claims)

            # Count overall distribution
            high_risk_count = sum(1 for v in result.labels.values() if v == "High Risk")
            low_risk_count = sum(1 for v in result.labels.values() if v == "Low Risk")

            return {
                "patient": self._get_patient_name(patient_id),
                "patient_id": patient_id,
                "risk_level": risk_label,
                "total_spend": round(float(total_spend), 2),
                "num_claims": int(num_claims),
                "population": {
                    "high_risk_patients": high_risk_count,
                    "low_risk_patients": low_risk_count,
                    "total_patients": len(result.labels),
                },
                "metrics": result.metrics,
            }
        except Exception as e:
            return {"error": f"Financial risk analysis failed: {str(e)}"}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 2: Care Coordination Classification
    # ═══════════════════════════════════════════════════════════════

    def get_care_coordination(self, patient_name: str) -> dict:
        """Classify a patient's care coordination quality (Optimal / Non-Optimal).

        Analyzes clinical patterns, provider diversity, and visit patterns
        to determine if a patient is receiving well-coordinated care.

        Args:
            patient_name: Full or partial name of the patient.

        Returns:
            Care coordination label and supporting metrics.
        """
        if not self._available:
            return {"error": "VectorDB modules not available."}

        patient_id = self._resolve_patient_id(patient_name)
        if not patient_id:
            return {"error": f"No patient found matching '{patient_name}'"}

        try:
            result = self._classify_care_coordination(
                patients=self.patients,
                diseases=self.diseases,
                encounters=self.encounters,
            )

            care_label = result.labels.get(patient_id, "Unknown")

            # Get patient-specific care stats
            patient_encounters = self.encounters[self.encounters["patient_id"] == patient_id]
            num_providers = patient_encounters["provider_id"].nunique() if not patient_encounters.empty else 0
            num_visits = len(patient_encounters)

            # Count overall distribution
            optimal_count = sum(1 for v in result.labels.values() if v == "Optimal Care")
            non_optimal_count = sum(1 for v in result.labels.values() if v == "Non-Optimal Care")

            return {
                "patient": self._get_patient_name(patient_id),
                "patient_id": patient_id,
                "care_quality": care_label,
                "num_providers_seen": int(num_providers),
                "total_visits": int(num_visits),
                "population": {
                    "optimal_care_patients": optimal_count,
                    "non_optimal_care_patients": non_optimal_count,
                    "total_patients": len(result.labels),
                },
            }
        except Exception as e:
            return {"error": f"Care coordination analysis failed: {str(e)}"}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 3: Similar Patients
    # ═══════════════════════════════════════════════════════════════

    def get_similar_patients(self, patient_name: str, top_k: int = 5) -> dict:
        """Find patients most similar to a given patient based on combined
        clinical, financial, and behavioural embeddings.

        Args:
            patient_name: Full or partial name of the patient.
            top_k: Number of similar patients to return (default: 5).

        Returns:
            List of similar patients with similarity scores.
        """
        if not self._available:
            return {"error": "VectorDB modules not available."}

        patient_id = self._resolve_patient_id(patient_name)
        if not patient_id:
            return {"error": f"No patient found matching '{patient_name}'"}

        try:
            result = self._find_similar_patients(
                query_patient_id=patient_id,
                patients=self.patients,
                diseases=self.diseases,
                claims=self.claims,
                encounters=self.encounters,
                embedding_space="hybrid",
                top_k=top_k,
            )

            # Enrich results with patient names
            similar = []
            for entry in result.similar_patients:
                pid = entry["patient_id"]
                name = self._get_patient_name(pid)
                similar.append({
                    "name": name,
                    "patient_id": pid,
                    "similarity": entry["similarity"],
                    "rank": entry["rank"],
                })

            return {
                "query_patient": self._get_patient_name(patient_id),
                "similar_patients": similar,
                "embedding_space": result.embedding_space,
                "metadata": result.metadata,
            }
        except Exception as e:
            return {"error": f"Similarity search failed: {str(e)}"}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 4: Cluster Summary
    # ═══════════════════════════════════════════════════════════════

    def get_cluster_summary(self) -> dict:
        """Get an overall summary of patient clustering across
        financial risk and care coordination dimensions.

        Returns:
            Distribution of patients across risk and care quality clusters.
        """
        if not self._available:
            return {"error": "VectorDB modules not available."}

        try:
            # Financial risk clustering
            fin_result = self._cluster_financial_risk(
                patients=self.patients,
                claims=self.claims,
                encounters=self.encounters,
            )

            # Care coordination clustering
            care_result = self._classify_care_coordination(
                patients=self.patients,
                diseases=self.diseases,
                encounters=self.encounters,
            )

            fin_dist = {}
            for label in fin_result.labels.values():
                fin_dist[label] = fin_dist.get(label, 0) + 1

            care_dist = {}
            for label in care_result.labels.values():
                care_dist[label] = care_dist.get(label, 0) + 1

            return {
                "total_patients": len(self.patients),
                "financial_risk": {
                    "distribution": fin_dist,
                    "metrics": fin_result.metrics,
                },
                "care_coordination": {
                    "distribution": care_dist,
                    "optimal_cluster_id": care_result.optimal_cluster,
                },
            }
        except Exception as e:
            return {"error": f"Cluster summary failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════
# Quick Test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tools = VectorTools()

    print("=" * 60)
    print("  VECTOR TOOLS — QUICK TEST")
    print("=" * 60)

    print("\n💰 Financial Risk (Soledad):")
    print(json.dumps(tools.get_financial_risk("Soledad"), indent=2, default=str))

    print("\n🏥 Care Coordination (Soledad):")
    print(json.dumps(tools.get_care_coordination("Soledad"), indent=2, default=str))

    print("\n👥 Similar Patients (Soledad):")
    print(json.dumps(tools.get_similar_patients("Soledad", top_k=3), indent=2, default=str))

    print("\n📊 Cluster Summary:")
    print(json.dumps(tools.get_cluster_summary(), indent=2, default=str))

    print("\n✅ All tests complete!")
