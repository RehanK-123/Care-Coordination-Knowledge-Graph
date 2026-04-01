import json
import os
import pandas as pd 

def extract_provider_id(ref):
    if ref and "npi|" in ref:
        return ref.split("|")[-1]
    return None

def extract(data):
    patients = []
    providers = []
    diseases = []
    encounters = []
    claims = []
    payers = []
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
            provider_type = "primary_provider" if provider_id else "unknown"
            service_provider = r.get("serviceProvider", {}).get("display")
            service_provider = service_provider if service_provider else "Unknown"
            service_provider_id = r["serviceProvider"]["reference"].split("|")[-1] if r.get("serviceProvider") else None

            providers.append({"service_provider": service_provider, "service_provider_id": service_provider_id,"provider_id": provider_id, "name": provider_name, "type": provider_type})

            reason = r.get("reasonCode", [{}])[0].get("coding", [{}])[0].get("display")
            
            start = r["period"]["start"]
            end = r["period"]["end"]

            if reason:
                diseases.append({"encounter_id": r["id"], "patient_id": patient_id, "disease": reason, "start": start, "end": end})

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
                "encounter_id": r["item"][0]["encounter"][0]["reference"].split(":")[-1],
                "patient_id": patient_id,
                "amount": r["total"]["value"],
                "billable_period_start": billable_period_start,
                "billable_period_end": billable_period_end
            })

            payers.append({"claim_id": r["id"], "payer": r["insurance"][0]["coverage"]["display"]})
        
    return {"patients": patients, "providers": providers, "diseases": diseases, "encounters": encounters, "claims": claims, "payers": payers}

def extract_and_save_data(fhir_json, output_dir="output"):
    result = extract(fhir_json)

    patients = result["patients"]
    providers = result["providers"]
    diseases = result["diseases"]
    encounters = result["encounters"]
    claims = result["claims"]
    payers = result["payers"]
    print("Extracted data:", result)

    patients_df = pd.DataFrame(patients)
    providers_df = pd.DataFrame(providers)
    diseases_df = pd.DataFrame(diseases)
    encounters_df = pd.DataFrame(encounters)
    claims_df = pd.DataFrame(claims)
    payers_df = pd.DataFrame(payers)

    os.makedirs(output_dir, exist_ok=True)
    patients_df.to_csv(os.path.join(output_dir, "patients.csv"), index=False, mode="a+", header=not os.path.exists(os.path.join(output_dir, "patients.csv")))
    providers_df.to_csv(os.path.join(output_dir, "providers.csv"), index=False, mode="a+", header=not os.path.exists(os.path.join(output_dir, "providers.csv")))
    diseases_df.to_csv(os.path.join(output_dir, "diseases.csv"), index=False, mode="a+", header=not os.path.exists(os.path.join(output_dir, "diseases.csv")))
    encounters_df.to_csv(os.path.join(output_dir, "encounters.csv"), index=False, mode="a+", header=not os.path.exists(os.path.join(output_dir, "encounters.csv")))
    claims_df.to_csv(os.path.join(output_dir, "claims.csv"), index=False, mode="a+", header=not os.path.exists(os.path.join(output_dir, "claims.csv")))
    payers_df.to_csv(os.path.join(output_dir, "payers.csv"), index=False, mode="a+", header=not os.path.exists(os.path.join(output_dir, "payers.csv")))

    print("✅ CSV files created successfully!")


if __name__ == "__main__":
    print("Ingestion")

    files = [file for file in os.listdir("synthea_sample_data_fhir_latest") if file.endswith(".json")]

    for file in files[:1]:
        with open(f"synthea_sample_data_fhir_latest/{file}") as f:
            data = json.load(f)
        print(f"Processing {file}...")
        extract_and_save_data(data)
