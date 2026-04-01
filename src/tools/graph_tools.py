"""
Graph Database Tools for Care Coordination Knowledge Graph
============================================================
Provides tool functions for querying the Neo4j knowledge graph.
Each function is designed to be exposed as an MCP tool for LLM agents.

Tools:
    - get_patient_info: Lookup patient demographics and conditions
    - get_patient_providers: Find all providers who treated a patient
    - get_patient_cost: Total healthcare cost for a patient
    - get_top_diseases: Most common diseases by patient count
    - get_comorbidities: Disease co-occurrence pairs
    - get_provider_network: Shared patients between providers
    - get_payer_breakdown: Cost breakdown by insurance payer
    - run_cypher_query: Execute custom Cypher queries (catch-all)
"""

from neo4j import GraphDatabase
from typing import Optional


class GraphTools:
    """Tool functions for querying the Neo4j Care Coordination Knowledge Graph."""

    # ── Graph Schema (provided to LLM for custom Cypher generation) ──
    GRAPH_SCHEMA = """
    NODES:
      - Patient (id, name, gender, birth_date, marital_status, city, state, deceased)
      - Disease (name)
      - Encounter (id, reason, period_start, period_end)
      - Provider (id, name, type, service_provider, service_provider_id)
      - Claim (id, amount, billable_period_start, billable_period_end)
      - Payer (name)

    RELATIONSHIPS:
      - (Patient)-[:HAS_ENCOUNTER]->(Encounter)
      - (Encounter)-[:WITH_PROVIDER]->(Provider)
      - (Encounter)-[:DIAGNOSED]->(Disease)
      - (Patient)-[:HAS_CONDITION]->(Disease)
      - (Encounter)-[:GENERATED_CLAIM]->(Claim)
      - (Claim)-[:FOR_PATIENT]->(Patient)
      - (Claim)-[:FULFILLED_BY]->(Payer)
    """

    def __init__(self, uri: str, user: str, password: str):
        """Initialize connection to Neo4j.

        Args:
            uri: Neo4j connection URI (e.g., "neo4j://127.0.0.1:7687")
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connection
        with self.driver.session() as session:
            session.run("RETURN 1")

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ═══════════════════════════════════════════════════════════════
    # TOOL 1: Patient Information
    # ═══════════════════════════════════════════════════════════════

    def get_patient_info(self, patient_name: str) -> dict:
        """Get detailed information about a patient including demographics and conditions.

        Args:
            patient_name: Full or partial name of the patient to search for.

        Returns:
            Dictionary with patient demographics, conditions list,
            encounter count, and provider count.
        """
        query = """
        MATCH (p:Patient)
        WHERE toLower(p.name) CONTAINS toLower($name)
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(d:Disease)
        OPTIONAL MATCH (p)-[:HAS_ENCOUNTER]->(e:Encounter)
        OPTIONAL MATCH (e)-[:WITH_PROVIDER]->(pr:Provider)
        WITH p,
             COLLECT(DISTINCT d.name) AS conditions,
             COUNT(DISTINCT e) AS encounter_count,
             COUNT(DISTINCT pr) AS provider_count
        RETURN p.id AS patient_id,
               p.name AS name,
               p.gender AS gender,
               p.birth_date AS birth_date,
               p.marital_status AS marital_status,
               p.city AS city,
               p.state AS state,
               p.deceased AS deceased,
               conditions,
               encounter_count,
               provider_count
        LIMIT 5
        """
        with self.driver.session() as session:
            result = session.run(query, name=patient_name)
            records = [dict(r) for r in result]

        if not records:
            return {"error": f"No patient found matching '{patient_name}'"}

        return {"patients": records, "count": len(records)}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 2: Patient Providers
    # ═══════════════════════════════════════════════════════════════

    def get_patient_providers(self, patient_name: str) -> dict:
        """Find all healthcare providers who have treated a specific patient.

        Args:
            patient_name: Full or partial name of the patient.

        Returns:
            List of providers with visit count and last visit date.
        """
        query = """
        MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)-[:WITH_PROVIDER]->(pr:Provider)
        WHERE toLower(p.name) CONTAINS toLower($name)
        WITH p, pr,
             COUNT(e) AS visits,
             MAX(e.period_start) AS last_visit
        RETURN p.name AS patient,
               pr.name AS provider,
               pr.type AS provider_type,
               pr.service_provider AS organization,
               visits,
               last_visit
        ORDER BY visits DESC
        """
        with self.driver.session() as session:
            result = session.run(query, name=patient_name)
            records = [dict(r) for r in result]

        if not records:
            return {"error": f"No providers found for patient matching '{patient_name}'"}

        return {"patient": records[0]["patient"], "providers": records}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 3: Patient Healthcare Cost
    # ═══════════════════════════════════════════════════════════════

    def get_patient_cost(self, patient_name: str) -> dict:
        """Get total healthcare cost and claims breakdown for a patient.

        Args:
            patient_name: Full or partial name of the patient.

        Returns:
            Total cost, number of claims, and cost breakdown by payer.
        """
        query = """
        MATCH (c:Claim)-[:FOR_PATIENT]->(p:Patient)
        WHERE toLower(p.name) CONTAINS toLower($name)
        OPTIONAL MATCH (c)-[:FULFILLED_BY]->(pay:Payer)
        WITH p,
             SUM(DISTINCT c.amount) AS total_cost,
             COUNT(DISTINCT c) AS num_claims,
             pay.name AS payer,
             SUM(c.amount) AS payer_cost
        RETURN p.name AS patient,
               total_cost,
               num_claims,
               COLLECT({payer: payer, amount: round(payer_cost, 2)}) AS payer_breakdown
        """
        with self.driver.session() as session:
            result = session.run(query, name=patient_name)
            records = [dict(r) for r in result]

        if not records:
            return {"error": f"No cost data found for patient matching '{patient_name}'"}

        return records[0]

    # ═══════════════════════════════════════════════════════════════
    # TOOL 4: Top Diseases
    # ═══════════════════════════════════════════════════════════════

    def get_top_diseases(self, top_n: int = 10) -> dict:
        """Find the most common diseases across all patients.

        Args:
            top_n: Number of top diseases to return (default: 10).

        Returns:
            List of diseases ranked by patient count.
        """
        query = """
        MATCH (p:Patient)-[:HAS_CONDITION]->(d:Disease)
        RETURN d.name AS disease,
               COUNT(DISTINCT p) AS patient_count
        ORDER BY patient_count DESC
        LIMIT $top_n
        """
        with self.driver.session() as session:
            result = session.run(query, top_n=top_n)
            records = [dict(r) for r in result]

        return {"diseases": records, "total_returned": len(records)}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 5: Disease Comorbidities
    # ═══════════════════════════════════════════════════════════════

    def get_comorbidities(self, disease_name: Optional[str] = None, top_n: int = 10) -> dict:
        """Find diseases that frequently co-occur in the same patients.

        Args:
            disease_name: Optional - filter co-occurrences for a specific disease.
                         If None, returns top co-occurring pairs across all diseases.
            top_n: Number of top pairs to return (default: 10).

        Returns:
            List of disease pairs with co-occurrence count.
        """
        if disease_name:
            query = """
            MATCH (p:Patient)-[:HAS_CONDITION]->(d1:Disease),
                  (p)-[:HAS_CONDITION]->(d2:Disease)
            WHERE toLower(d1.name) CONTAINS toLower($disease_name)
              AND d1 <> d2
            RETURN d1.name AS disease,
                   d2.name AS co_occurring_disease,
                   COUNT(DISTINCT p) AS shared_patients
            ORDER BY shared_patients DESC
            LIMIT $top_n
            """
            params = {"disease_name": disease_name, "top_n": top_n}
        else:
            query = """
            MATCH (p:Patient)-[:HAS_CONDITION]->(d1:Disease),
                  (p)-[:HAS_CONDITION]->(d2:Disease)
            WHERE elementId(d1) < elementId(d2)
            RETURN d1.name AS disease_1,
                   d2.name AS disease_2,
                   COUNT(DISTINCT p) AS co_occurrence
            ORDER BY co_occurrence DESC
            LIMIT $top_n
            """
            params = {"top_n": top_n}

        with self.driver.session() as session:
            result = session.run(query, **params)
            records = [dict(r) for r in result]

        return {"comorbidities": records, "total_returned": len(records)}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 6: Provider Network
    # ═══════════════════════════════════════════════════════════════

    def get_provider_network(self, provider_name: Optional[str] = None, top_n: int = 10) -> dict:
        """Find provider pairs that share the most patients (care coordination links).

        Args:
            provider_name: Optional - find providers connected to a specific provider.
                          If None, returns top provider pairs by shared patients.
            top_n: Number of top pairs to return (default: 10).

        Returns:
            List of provider pairs with shared patient count.
        """
        if provider_name:
            query = """
            MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e1:Encounter)-[:WITH_PROVIDER]->(pr1:Provider),
                  (p)-[:HAS_ENCOUNTER]->(e2:Encounter)-[:WITH_PROVIDER]->(pr2:Provider)
            WHERE toLower(pr1.name) CONTAINS toLower($provider_name)
              AND pr1 <> pr2
            WITH pr1, pr2, COUNT(DISTINCT p) AS shared_patients
            RETURN pr1.name AS provider,
                   pr2.name AS connected_provider,
                   pr1.service_provider AS organization_1,
                   pr2.service_provider AS organization_2,
                   shared_patients
            ORDER BY shared_patients DESC
            LIMIT $top_n
            """
            params = {"provider_name": provider_name, "top_n": top_n}
        else:
            query = """
            MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e1:Encounter)-[:WITH_PROVIDER]->(pr1:Provider),
                  (p)-[:HAS_ENCOUNTER]->(e2:Encounter)-[:WITH_PROVIDER]->(pr2:Provider)
            WHERE elementId(pr1) < elementId(pr2)
            WITH pr1, pr2, COUNT(DISTINCT p) AS shared_patients
            RETURN pr1.name AS provider_1,
                   pr2.name AS provider_2,
                   shared_patients
            ORDER BY shared_patients DESC
            LIMIT $top_n
            """
            params = {"top_n": top_n}

        with self.driver.session() as session:
            result = session.run(query, **params)
            records = [dict(r) for r in result]

        return {"provider_network": records, "total_returned": len(records)}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 7: Payer Breakdown
    # ═══════════════════════════════════════════════════════════════

    def get_payer_breakdown(self) -> dict:
        """Get total cost and claim count breakdown by insurance payer.

        Returns:
            List of payers ranked by total amount paid, including claim counts.
        """
        query = """
        MATCH (c:Claim)-[:FULFILLED_BY]->(pay:Payer)
        WITH pay.name AS payer,
             round(SUM(c.amount), 2) AS total_paid,
             COUNT(c) AS num_claims,
             round(AVG(c.amount), 2) AS avg_claim
        RETURN payer, total_paid, num_claims, avg_claim
        ORDER BY total_paid DESC
        """
        with self.driver.session() as session:
            result = session.run(query)
            records = [dict(r) for r in result]

        return {"payers": records, "total_payers": len(records)}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 8: Custom Cypher Query (Catch-All)
    # ═══════════════════════════════════════════════════════════════

    def run_cypher_query(self, cypher: str, params: Optional[dict] = None) -> dict:
        """Execute a custom Cypher query against the knowledge graph.

        Use this when no predefined tool matches the user's question.
        The LLM should generate a valid Cypher query based on the graph schema.

        Args:
            cypher: A valid Cypher query string.
            params: Optional dictionary of query parameters.

        Returns:
            Query results as a list of dictionaries.

        Graph Schema:
            NODES:
              Patient (id, name, gender, birth_date, marital_status, city, state, deceased)
              Disease (name)
              Encounter (id, reason, period_start, period_end)
              Provider (id, name, type, service_provider, service_provider_id)
              Claim (id, amount, billable_period_start, billable_period_end)
              Payer (name)

            RELATIONSHIPS:
              (Patient)-[:HAS_ENCOUNTER]->(Encounter)
              (Encounter)-[:WITH_PROVIDER]->(Provider)
              (Encounter)-[:DIAGNOSED]->(Disease)
              (Patient)-[:HAS_CONDITION]->(Disease)
              (Encounter)-[:GENERATED_CLAIM]->(Claim)
              (Claim)-[:FOR_PATIENT]->(Patient)
              (Claim)-[:FULFILLED_BY]->(Payer)
        """
        # Safety: block destructive operations
        dangerous_keywords = ["DELETE", "REMOVE", "DROP", "SET", "CREATE", "MERGE", "DETACH"]
        cypher_upper = cypher.upper().strip()
        for keyword in dangerous_keywords:
            if keyword in cypher_upper and "RETURN" not in cypher_upper:
                return {"error": f"Write operations are not allowed. Query contains '{keyword}'."}

        try:
            with self.driver.session() as session:
                result = session.run(cypher, **(params or {}))
                records = [dict(r) for r in result]

            return {"results": records, "count": len(records)}
        except Exception as e:
            return {"error": f"Cypher query failed: {str(e)}"}

    # ═══════════════════════════════════════════════════════════════
    # TOOL 9: Graph Statistics
    # ═══════════════════════════════════════════════════════════════

    def get_graph_stats(self) -> dict:
        """Get overall statistics of the knowledge graph.

        Returns:
            Node counts and relationship counts for all types.
        """
        node_queries = {
            "patients": "MATCH (p:Patient) RETURN COUNT(p) AS cnt",
            "diseases": "MATCH (d:Disease) RETURN COUNT(d) AS cnt",
            "encounters": "MATCH (e:Encounter) RETURN COUNT(e) AS cnt",
            "providers": "MATCH (pr:Provider) RETURN COUNT(pr) AS cnt",
            "claims": "MATCH (c:Claim) RETURN COUNT(c) AS cnt",
            "payers": "MATCH (pay:Payer) RETURN COUNT(pay) AS cnt",
        }

        rel_queries = {
            "has_encounter": "MATCH ()-[r:HAS_ENCOUNTER]->() RETURN COUNT(r) AS cnt",
            "with_provider": "MATCH ()-[r:WITH_PROVIDER]->() RETURN COUNT(r) AS cnt",
            "diagnosed": "MATCH ()-[r:DIAGNOSED]->() RETURN COUNT(r) AS cnt",
            "has_condition": "MATCH ()-[r:HAS_CONDITION]->() RETURN COUNT(r) AS cnt",
            "generated_claim": "MATCH ()-[r:GENERATED_CLAIM]->() RETURN COUNT(r) AS cnt",
            "for_patient": "MATCH ()-[r:FOR_PATIENT]->() RETURN COUNT(r) AS cnt",
            "fulfilled_by": "MATCH ()-[r:FULFILLED_BY]->() RETURN COUNT(r) AS cnt",
        }

        nodes = {}
        relationships = {}

        with self.driver.session() as session:
            for label, query in node_queries.items():
                nodes[label] = session.run(query).single()["cnt"]

            for label, query in rel_queries.items():
                relationships[label] = session.run(query).single()["cnt"]

        return {
            "nodes": nodes,
            "relationships": relationships,
            "total_nodes": sum(nodes.values()),
            "total_relationships": sum(relationships.values()),
        }


# ═══════════════════════════════════════════════════════════════
# Quick Test (run this file directly)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    # Connect to Neo4j
    tools = GraphTools(
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password="12345678",  # Change this
    )

    print("=" * 60)
    print("  GRAPH TOOLS — QUICK TEST")
    print("=" * 60)

    # Test 1: Graph Stats
    print("\n📊 Graph Statistics:")
    stats = tools.get_graph_stats()
    print(json.dumps(stats, indent=2))

    # Test 2: Patient Info
    print("\n👤 Patient Info (Soledad):")
    info = tools.get_patient_info("Soledad")
    print(json.dumps(info, indent=2, default=str))

    # Test 3: Patient Providers
    print("\n🏥 Patient Providers (Soledad):")
    providers = tools.get_patient_providers("Soledad")
    print(json.dumps(providers, indent=2, default=str))

    # Test 4: Patient Cost
    print("\n💰 Patient Cost (Giovanni):")
    cost = tools.get_patient_cost("Giovanni")
    print(json.dumps(cost, indent=2, default=str))

    # Test 5: Top Diseases
    print("\n🦠 Top 5 Diseases:")
    diseases = tools.get_top_diseases(top_n=5)
    print(json.dumps(diseases, indent=2))

    # Test 6: Comorbidities
    print("\n🔗 Top 5 Comorbidities:")
    comorbid = tools.get_comorbidities(top_n=5)
    print(json.dumps(comorbid, indent=2))

    # Test 7: Provider Network
    print("\n🌐 Provider Network (top 5):")
    network = tools.get_provider_network(top_n=5)
    print(json.dumps(network, indent=2))

    # Test 8: Payer Breakdown
    print("\n🏦 Payer Breakdown:")
    payers = tools.get_payer_breakdown()
    print(json.dumps(payers, indent=2))

    # Test 9: Custom Query
    print("\n🔍 Custom Query (female patients count):")
    custom = tools.run_cypher_query(
        "MATCH (p:Patient) WHERE p.gender = 'female' RETURN COUNT(p) AS female_count"
    )
    print(json.dumps(custom, indent=2))

    tools.close()
    print("\n✅ All tests complete!")
