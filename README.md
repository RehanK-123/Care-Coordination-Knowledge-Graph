# Care Coordination Knowledge Graph
### Comprehensive Project Documentation

---

## 📌 1. Project Overview

The **Care Coordination Knowledge Graph** is an AI-powered healthcare analytics platform designed to analyze complex patient data. It combines two powerful database paradigms:
1. **Graph Database (Neo4j)** for deterministic relational queries (e.g., patient networks, financial cost aggregations, temporal disease journeys).
2. **Vector Database (Qdrant)** for probabilistic machine learning tasks (e.g., patient phenotype similarity, financial risk clustering, care coordination quality).

Sitting on top of these databases is a conversational **LLM Router Agent (OpenAI)** that translates natural language questions into precise database queries and tool executions.

---

## 🏗️ 2. Architectural Blueprint

The architecture is divided into three distinct layers:

### Layer 1: Data Ingestion & Storage
- **`Ingestion.py`**: Parses nested raw FHIR (Fast Healthcare Interoperability Resources) JSON files exported from Synthea into clean, relational CSV formats (`patients.csv`, `claims.csv`, `encounters.csv`, `diseases.csv`, etc.).
- **`graphdb.py`**: Ingests the unified CSV data into a Neo4j Knowledge Graph, creating nodes (`Patient`, `Disease`, `Encounter`, `Provider`, `Claim`) and semantic relationships (`HAS_CONDITION`, `WITH_PROVIDER`, `GENERATED`).

### Layer 2: Machine Learning & Vector Embeddings
- **`VectorDB_Helper.py` / `VectorDB.py`**: Contains the core Machine Learning pipeline.
  - Generates semantic clinical embeddings using raw text and a localized `FastText` language model.
  - Scales financial and encounter data using `StandardScaler`.
  - Performs multi-modal concatenation (Hybrid Embeddings) fed through a Keras `Autoencoder` to generate dense representations of patient phenotypes.
  - Pushes final complex tensors to `Qdrant` for fast similarity search.
- **`train.py`**: Handles offline K-Means and DBSCAN training regimens, saving serialized models to the `models/` directory.
- **`classification_tool.py`**: Determines if a specific patient's diseases are "Chronic" or "Non-Chronic" using algorithmic thresholding based on historical recency, frequency, and temporal persistence.

### Layer 3: Agentic Orchestration
- **`src/tools/`**: Holds abstracted wrapper functions (`graph_tools.py` and `vector_tools.py`) that provide safe interfaces between the LLM and the raw underlying database engines.
- **`src/agent/agent.py`**: Implements the OpenAI tool-calling loop. It relies heavily on prompt engineering and Level-2 reasoning to determine context. If a user asks *"Is Soledad high risk?"*, it triggers the Vector DB. If they ask *"How much did Soledad cost?"*, it routes to Neo4j.
- **`main.py`**: The interactive CLI entry point enabling users to converse with the system without touching code.

---

## 🛠️ 3. Tool Capabilities (Agent Interface)

When interacting with the agent via `main.py`, the LLM has autonomous access to the following deterministic and probabilistic functions:

### 🟢 Graph Knowledge Tools (Neo4j)
1. **`get_patient_info`**: Retrieves aggregate demographic and clinical summaries.
2. **`get_patient_providers`**: Maps all care providers visited by a given individual.
3. **`get_patient_cost`**: Sums total claims amounts and breaks down specific organizational payors.
4. **`get_comorbidities`**: Cross-references temporal occurrences to see what diseases most commonly co-occur (e.g., finding the overlap between Gingivitis and Chronic Pain).
5. **`get_provider_network`**: Identifies care coordination patterns (which doctors share the most patients).
6. **`run_cypher_query`**: A completely dynamic fallback allowing the LLM to write custom Neo4j syntax on the fly for queries without a predefined function. 

### 🟣 Vector Intelligence Tools (Qdrant & Local ML)
1. **`get_financial_risk`**: Accesses offline clustering models to assign a categorical "High Risk" or "Low Risk" label to a patient.
2. **`get_care_coordination`**: Analyzes provider diversity and visit patterns to determine if a patient has "Optimal Care".
3. **`get_similar_patients`**: Uses Autoencoder generated tensors and Qdrant cosine-similarity to find N mathematically similar patients based on mixed behavioral and clinical phenotypes.
4. **`chronic_classification`**: Calculates specific frequency matrices over a patient's historical diseases to determine chronicity likelihood. *(Newly implemented by User)*.

---

## 🚀 4. Deployment & Execution Instructions

### A. Environment Replication
Because of the heavy ML implementations (TensorFlow, FastText), this pipeline specifically requires **Python 3.11 or 3.12**.
```bash
# 1. Start Conda Environment
conda create -n vdb_env python=3.11
conda activate vdb_env

# 2. Install Standard and ML Requirements
pip install -r requirements.txt
```

### B. Pre-requisite Binaries & Containers
1. **Language Model**: The project utilizes the `cc.en.300.bin` (4.2GB) FastText word-vector model. This must be present in the root directory and explicitly ignored in `.gitignore` to prevent repository bloat.
2. **Databases**: 
   - Start **Neo4j** bound to `127.0.0.1:7687` (Default User: `neo4j` | Pass: `12345678`).
   - Start **Qdrant** via Docker mapping ports 6333 and 6334:
     ```bash
     docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
     ```

### C. Booting the Orchestrator
To converse with the finalized agent:
```bash
export OPENAI_API_KEY="sk-..."
python main.py
```

---

> **Database Mutation Risks:** If modifying `Ingestion.py`, be acutely aware of CSV structures. Missing columns (e.g. accidentally omitting `patient_id` during Claim parses) will permanently corrupt shifted tabular ingestion, crashing the VectorDB mapping pipelines due to mixed string-float casting errors in `StandardScaler`. Always use `git checkout output/` to safely restore deterministic states if ML mapping fails!