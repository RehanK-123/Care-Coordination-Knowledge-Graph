"""
patient_analysis_mcp.py
======================

Integrated version with:
- LLM Input Router
- Qdrant-aware ingestion
- Embedding generation
- MCP-compatible tools
"""

from __future__ import annotations
import json
from typing import Union, Optional
import numpy as np
import pandas as pd

# ── Existing imports ─────────────────────────────────────
from VectorDB_Helper import (
    cluster_financial_risk,
    classify_care_coordination,
    find_similar_patients,
    _get_qdrant_client,
    _patients_in_qdrant
)

from VectorDB import (
    get_clinical_embeddings,
    get_financial_embeddings,
    get_behavioural_embeddings,
    get_hybrid_embeddings,
)

from Ingestion import extract

# ─────────────────────────────────────────────────────────
# 🧠 1. INPUT ROUTER (LLM ENTRY POINT)
# ─────────────────────────────────────────────────────────

def handle_llm_input(input_data: Union[str, dict]):
    """
    Handles:
    - FHIR JSON → ingestion + embeddings
    - Patient ID → fetch from Qdrant
    - Invalid → reject
    """

    # ── CASE 1: FHIR JSON ─────────────────────────────
    if isinstance(input_data, dict):

        if input_data.get("resourceType"):
            print("[Router] FHIR JSON detected")

            # Step 1: Ingestion
            extracted = extract(input_data)
            print("Extracted data:", extracted)

            patients = pd.DataFrame(extracted["patients"], columns= ["patient_id","name","gender","birth_date","marital_status","city","state","deceased"])
            diseases = pd.DataFrame(extracted["diseases"], columns= ["encounter_id","patient_id","disease","start","end"])
            claims = pd.DataFrame(extracted["claims"], columns= ["claim_id","patient_id","amount","provider_id","billable_period_start","billable_period_end"])
            encounters = pd.DataFrame(extracted["encounters"], columns= ["encounter_id","patient_id","provider_id","reason","period_start","period_end"])

            # Step 2: Embeddings
            # print(diseases.head(), patients.head(), encounters.head(), claims.head())
            diseases["start"] = pd.to_datetime(diseases["start"], errors="coerce", utc=True)
            clinical = get_clinical_embeddings(diseases, patients)
            financial = get_financial_embeddings(patients, encounters, claims)
            behavioural = get_behavioural_embeddings(patients, encounters)
            print(clinical.shape, financial.shape, behavioural.shape)
            hybrid = get_hybrid_embeddings(clinical, financial, behavioural)

            return {
            "status": "FHIR_processed",
            "data": {
                "patients": patients.to_dict(),
                "diseases": diseases.to_dict(),
                "claims": claims.to_dict(),
                "encounters": encounters.to_dict(),
            },
            "embeddings": {
                "clinical": clinical.tolist(),
                "financial": financial.tolist(),
                "behavioural": behavioural.tolist(),
                "hybrid": hybrid.tolist()
            },
            "available_tools": ["financial_risk", "care_coordination", "similarity"]
        }


        else:
            return {"error": "Invalid JSON: Not a FHIR resource"}

    # ── CASE 2: Patient ID ────────────────────────────
    elif isinstance(input_data, str):
        print("[Router] Patient ID detected")

        client = _get_qdrant_client()
        exists = _patients_in_qdrant(client, "hybrid", [input_data])

        if not exists:
            return {"error": "Patient not found in Qdrant"}

        return {
            "status": "patient_found",
            "patient_id": input_data,
            "available_tools": ["financial_risk", "care_coordination", "similarity"]
        }

    # ── CASE 3: Invalid ───────────────────────────────
    return {"error": "Invalid input type"}


# ─────────────────────────────────────────────────────────
# 🔧 2. MCP TOOL: FINANCIAL RISK
# ─────────────────────────────────────────────────────────

def mcp_cluster_financial_risk_tool(input_payload: dict):
    """
    MCP Tool: Financial Risk Clustering
    """

    required = ["patients", "claims", "encounters"]
    if not all(k in input_payload for k in required):
        return {"error": "Missing required inputs"}

    precomputed = None
    if "embeddings" in input_payload and "financial" in input_payload["embeddings"]:
        precomputed = np.array(input_payload["embeddings"]["financial"])

    result = cluster_financial_risk(
        patients=pd.DataFrame(input_payload["patients"]),
        claims=pd.DataFrame(input_payload["claims"]),
        encounters=pd.DataFrame(input_payload["encounters"]),
        precomputed_embeddings=precomputed  # ✅ FIX
    )

    return {
        "labels": result.labels,
        "metrics": result.metrics,
        "summary": result.cluster_summary.to_dict(),
    }


# ─────────────────────────────────────────────────────────
# 🔧 3. MCP TOOL: CARE COORDINATION
# ─────────────────────────────────────────────────────────

def mcp_care_coordination_tool(input_payload: dict):
    """
    MCP Tool: Care Coordination Classification
    """

    required = ["patients", "diseases", "encounters"]
    if not all(k in input_payload for k in required):
        return {"error": "Missing required inputs"}

    precomputed = None
    if "embeddings" in input_payload and "clinical" in input_payload["embeddings"]:
        precomputed = np.array(input_payload["embeddings"]["clinical"])

    result = classify_care_coordination(
        patients=pd.DataFrame(input_payload["patients"]),
        diseases=pd.DataFrame(input_payload["diseases"]),
        encounters=pd.DataFrame(input_payload["encounters"]),
        precomputed_embeddings=precomputed  # ✅ FIX
    )

    return {
        "labels": result.labels,
        "optimal_cluster": result.optimal_cluster,
        "summary": result.cluster_summary.to_dict(),
    }


# ─────────────────────────────────────────────────────────
# 🔧 4. MCP TOOL: SIMILARITY SEARCH
# ─────────────────────────────────────────────────────────

def mcp_similarity_tool(input_payload: dict):
    """
    MCP Tool: Patient Similarity Search
    """

    required = ["query_patient_id", "patients"]
    if not all(k in input_payload for k in required):
        return {"error": "Missing required inputs"}

    embedding_space = input_payload.get("embedding_space", "hybrid")

    precomputed = None
    if "embeddings" in input_payload and embedding_space in input_payload["embeddings"]:
        precomputed = np.array(input_payload["embeddings"][embedding_space])

    result = find_similar_patients(
        query_patient_id=input_payload["query_patient_id"],
        patients=pd.DataFrame(input_payload["patients"]),
        embedding_space=embedding_space,
        top_k=input_payload.get("top_k", 10),
        precomputed_embeddings=precomputed  # ✅ FIX
    )

    return {
        "query_patient": result.query_patient,
        "similar_patients": result.similar_patients,
        "metadata": result.metadata,
    }


# ─────────────────────────────────────────────────────────
# 🧩 5. MCP TOOL REGISTRY
# ─────────────────────────────────────────────────────────

MCP_TOOLS = {
    "financial_risk": mcp_cluster_financial_risk_tool,
    "care_coordination": mcp_care_coordination_tool,
    "similarity": mcp_similarity_tool,
}


# ─────────────────────────────────────────────────────────
# 🧪 6. EXAMPLE USAGE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_dic = json.load(open("fhir/Wilson960_Heathcote539_a8267e63-5207-4c6d-8f40-2e5ef1e0090d.json"))
    print(type(input_dic))
    response = handle_llm_input(input_dic)
    print("\nFHIR Response:\n", response)

    # Example 2: Patient ID input
    # response = handle_llm_input("P001")
    # print("\nPatient ID Response:\n", response)

    # Example 3: Tool call
    tool_payload = {
    **response["data"],          # patients, claims, etc.
    "embeddings": response["embeddings"],  # ✅ critical
}

    tool_output = mcp_cluster_financial_risk_tool(tool_payload)
    print("\nTool Output:\n", tool_output)