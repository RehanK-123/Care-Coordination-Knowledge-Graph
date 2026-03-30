import json
import re

with open("synthea_sample_data_fhir_latest/Adrianne466_Jannie509_Simonis280_5b556a04-5899-ec90-f607-f125c7871e8f.json") as f:
    data = json.load(f)
patients = []
providers = []
diseases = []
encounters = []
claims = []
payers = []


def extract_provider_id(ref):
    if ref and "npi|" in ref:
        return ref.split("|")[-1]
    return None


for entry in data["entry"]:
    r = entry["resource"]
    rtype = r["resourceType"]
 
    # ---------------- PATIENT ---------------- #
    if rtype == "Patient":
        patients.append({
            "patient_id": r["id"],
            "name": r["name"][0]["given"][0] + " " + r["name"][0]["family"],
            "gender": r.get("gender"),
            "birth_date": r.get("birthDate"),
            "marital_status": r.get("maritalStatus", {}).get("coding", [{}])[0].get("display"),
            "city": r["address"][0].get("city"),
            "state": r["address"][0].get("state"),
            "deceased": r.get("deceasedBoolean", False)
        })

    # ---------------- ENCOUNTER ---------------- #
    elif rtype == "Encounter":
        patient_id = r["subject"]["reference"].split(":")[-1]

        provider_ref = r["participant"][0]["individual"]["reference"]
        provider_id = extract_provider_id(provider_ref)
        provider_name = r["participant"][0]["individual"]["display"]
        provider_type = r["participant"][0]["type"][0]["coding"][0]["display"]
        service_provider = r.get("serviceProvider", {}).get("display")
        service_provider = service_provider if service_provider else "Unknown"
        service_provider_id = r["serviceProvider"]["reference"].split("|")[-1] if r.get("serviceProvider") else None

        providers.append({"service_provider": service_provider, "service_provider_id": service_provider_id,"provider_id": provider_id, "name": provider_name, "type": provider_type})

        reason = r.get("reasonCode", [{}])[0].get("coding", [{}])[0].get("display")
        
        start = r["period"]["start"]
        end = r["period"]["end"]

        if reason:
            diseases.append({"encounter_id": r["id"], "patient_id": patient_id, "disease": reason})

        encounters.append({
            "encounter_id": r["id"],
            "patient_id": patient_id,
            "provider_id": provider_id,
            "reason": reason,
            "period_start": start,
            "period_end": end
        })

    # ---------------- CLAIM ---------------- #
    elif rtype == "Claim":
        patient_id = r["patient"]["reference"].split(":")[-1]
        billable_period_start = r["billablePeriod"]["start"]
        billable_period_end = r["billablePeriod"]["end"]
        claims.append({
            "claim_id": r["id"],
            "patient_id": patient_id,
            "amount": r["total"]["value"],
            "provider_id": r["provider"]["reference"].split("|")[-1] if r.get("provider") else None,
            "billable_period_start": billable_period_start,
            "billable_period_end": billable_period_end
        })

        payers.append({"claim_id": r["id"], "payer": r["insurance"][0]["coverage"]["display"]})

    # ---------------- EOB ---------------- #
    elif rtype == "ExplanationOfBenefit":
        payer = r.get("insurer", {}).get("display")
        if payer:
            payers.append({"claim_id": r["id"], "payer": payer})




import pandas as pd

# ---------------- CONVERT TO DATAFRAMES ---------------- #
patients_df = pd.DataFrame(patients)
providers_df = pd.DataFrame(providers)
diseases_df = pd.DataFrame(diseases)
encounters_df = pd.DataFrame(encounters)
claims_df = pd.DataFrame(claims)
payers_df = pd.DataFrame(payers)

# ---------------- SAVE AS CSV ---------------- #
patients_df.to_csv("output/patients.csv", index=False)
providers_df.to_csv("output/providers.csv", index=False)
diseases_df.to_csv("output/diseases.csv", index=False)
encounters_df.to_csv("output/encounters.csv", index=False)
claims_df.to_csv("output/claims.csv", index=False)
payers_df.to_csv("output/payers.csv", index=False)

print("✅ CSV files created successfully!")