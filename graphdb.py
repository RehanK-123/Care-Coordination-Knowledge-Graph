import pandas as pd
from neo4j import GraphDatabase

# ── Connection ───────────────────────────────────────────────
driver = GraphDatabase.driver(
    "neo4j://127.0.0.1:7687",
    auth=("neo4j", "12345678")
)

# ── Load CSVs ────────────────────────────────────────────────
patients  = pd.read_csv("output/patients.csv")
providers = pd.read_csv("output/providers.csv")
diseases  = pd.read_csv("output/diseases.csv")
encounters= pd.read_csv("output/encounters.csv")
claims    = pd.read_csv("output/claims.csv")
payers    = pd.read_csv("output/payers.csv")

# ── Clean NaN ────────────────────────────────────────────────
def clean(df):
    return df.where(pd.notnull(df), None)

patients   = clean(patients)
providers  = clean(providers).drop_duplicates(subset=["provider_id"]).dropna(subset=["provider_id"])
diseases   = clean(diseases).dropna(subset=["encounter_id", "patient_id", "disease"])
encounters = clean(encounters).dropna(subset=["encounter_id", "patient_id"])
claims     = clean(claims).dropna(subset=["claim_id", "encounter_id"])
payers     = clean(payers).dropna(subset=["claim_id", "payer"])

# ── Batch helper ─────────────────────────────────────────────
def batch_run(session, cypher, df, label="", size=500):
    rows = df.to_dict("records")
    total = 0
    for i in range(0, len(rows), size):
        session.run(cypher, rows=rows[i:i+size])
        total += len(rows[i:i+size])
    print(f"  ✓ {label}: {total} rows")

# ── Build graph ──────────────────────────────────────────────
with driver.session(database="neo4j") as s:

    s.run("MATCH (n) DETACH DELETE n")
    print("✓ Graph cleared")

    # Indexes
    for label, prop in [
        ("Patient", "id"),
        ("Provider", "id"),
        ("Encounter", "id"),
        ("Claim", "id"),
        ("Disease", "name"),
        ("Payer", "name"),
    ]:
        s.run(f"CREATE INDEX {label.lower()}_idx IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")

    print("✓ Indexes created\n")

    print("Creating nodes...")

    # ── NODES ─────────────────────────────────────────────

    batch_run(s, """
        UNWIND $rows AS r
        MERGE (p:Patient {id: toString(r.patient_id)})
        SET p.name = r.name,
            p.gender = r.gender,
            p.birth_date = r.birth_date,
            p.marital_status = r.marital_status,
            p.city = r.city,
            p.state = r.state,
            p.deceased = r.deceased
    """, patients, "Patients")

    batch_run(s, """
        UNWIND $rows AS r
        MERGE (pr:Provider {id: toString(r.provider_id)})
        SET pr.name = r.name,
            pr.type = r.type,
            pr.service_provider = r.service_provider,
            pr.service_provider_id = r.service_provider_id
    """, providers, "Providers")

    batch_run(s, """
        UNWIND $rows AS r
        MERGE (e:Encounter {id: toString(r.encounter_id)})
        SET e.reason = r.reason,
            e.period_start = r.period_start,
            e.period_end = r.period_end
    """, encounters, "Encounters")

    batch_run(s, """
        UNWIND $rows AS r
        MERGE (c:Claim {id: toString(r.claim_id)})
        SET c.amount = r.amount,
            c.billable_period_start = r.billable_period_start,
            c.billable_period_end = r.billable_period_end
    """, claims, "Claims")

    unique_payers = payers[["payer"]].drop_duplicates()

    batch_run(s, """
        UNWIND $rows AS r
        MERGE (py:Payer {name: r.payer})
    """, unique_payers, "Payers")

    batch_run(s, """
        UNWIND $rows AS r
        MERGE (d:Disease {name: r.disease})
    """, diseases, "Diseases")

    # ── RELATIONSHIPS ─────────────────────────────────────

    print("\nCreating relationships...")

    batch_run(s, """
        UNWIND $rows AS r
        MATCH (p:Patient {id: toString(r.patient_id)})
        MATCH (e:Encounter {id: toString(r.encounter_id)})
        MERGE (p)-[:HAS_ENCOUNTER]->(e)
    """, encounters, "HAS_ENCOUNTER")

    enc_with_prov = encounters.dropna(subset=["provider_id"])

    batch_run(s, """
        UNWIND $rows AS r
        MATCH (e:Encounter {id: toString(r.encounter_id)})
        MATCH (pr:Provider {id: toString(r.provider_id)})
        MERGE (e)-[:WITH_PROVIDER]->(pr)
    """, enc_with_prov, "WITH_PROVIDER")

    batch_run(s, """
        UNWIND $rows AS r
        MATCH (e:Encounter {id: toString(r.encounter_id)})
        MATCH (d:Disease {name: r.disease})
        MERGE (e)-[:DIAGNOSED]->(d)
    """, diseases, "DIAGNOSED")

    batch_run(s, """
        UNWIND $rows AS r
        MATCH (p:Patient {id: toString(r.patient_id)})
        MATCH (d:Disease {name: r.disease})
        MERGE (p)-[:HAS_CONDITION]->(d)
    """, diseases, "HAS_CONDITION")

    batch_run(s, """
        UNWIND $rows AS r
        MATCH (e:Encounter {id: toString(r.encounter_id)})
        MATCH (c:Claim {id: toString(r.claim_id)})
        MERGE (e)-[:GENERATED_CLAIM]->(c)
    """, claims, "GENERATED_CLAIM")

    batch_run(s, """
        UNWIND $rows AS r
        MATCH (c:Claim {id: toString(r.claim_id)})
        MATCH (e:Encounter {id: toString(r.encounter_id)})<-[:HAS_ENCOUNTER]-(p:Patient)
        MERGE (c)-[:FOR_PATIENT]->(p)
    """, claims, "FOR_PATIENT")

    claims_payers = claims[["claim_id"]].merge(payers, on="claim_id", how="inner")

    batch_run(s, """
        UNWIND $rows AS r
        MATCH (c:Claim {id: toString(r.claim_id)})
        MATCH (py:Payer {name: r.payer})
        MERGE (c)-[:FULFILLED_BY]->(py)
    """, claims_payers, "FULFILLED_BY")


# ── Verification ───────────────────────────────────────────
print("\n── Final counts ──────────────────")

with driver.session(database="neo4j") as s:
    for label in ["Patient","Provider","Encounter","Disease","Claim","Payer"]:
        n = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
        print(f"{label}: {n}")

    print()

    result = s.run("MATCH ()-[r]->() RETURN type(r), count(*)")
    for record in result:
        print(record)

driver.close()

print("\n✅ DONE — GRAPH READY")