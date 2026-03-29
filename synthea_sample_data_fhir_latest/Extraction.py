# Let's parse the uploaded FHIR JSON and export key tables as CSV files
import json
import pandas as pd

file_path = "/mnt/data/Adrianne466_Jannie509_Simonis280_5b556a04-5899-ec90-f607-f125c7871e8f.json"

with open(file_path) as f:
    data = json.load(f)

patients, encounters, conditions, claims, allergies = [], [], [], [], []

def clean_ref(ref):
    return ref.split(":")[-1] if ref else None

for entry in data["entry"]:
    resource = entry["resource"]
    rtype = resource["resourceType"]

    if rtype == "Patient":
        patients.append({
            "patient_id": resource["id"],
            "name": resource["name"][0]["given"][0] + " " + resource["name"][0]["family"],
            "gender": resource.get("gender"),
            "birth_date": resource.get("birthDate"),
            "city": resource["address"][0].get("city"),
            "state": resource["address"][0].get("state")
        })

    elif rtype == "Encounter":
        encounters.append({
            "encounter_id": resource["id"],
            "patient_id": clean_ref(resource["subject"]["reference"]),
            "provider": resource["participant"][0]["individual"]["display"],
            "start": resource["period"]["start"],
            "end": resource["period"]["end"],
            "reason": resource.get("reasonCode", [{}])[0].get("coding", [{}])[0].get("display")
        })

    elif rtype == "Condition":
        conditions.append({
            "condition_id": resource["id"],
            "patient_id": clean_ref(resource["subject"]["reference"]),
            "encounter_id": clean_ref(resource["encounter"]["reference"]),
            "condition": resource["code"]["coding"][0]["display"]
        })

    elif rtype == "AllergyIntolerance":
        allergies.append({
            "allergy_id": resource["id"],
            "patient_id": clean_ref(resource["patient"]["reference"]),
            "type": resource["type"],
            "substance": resource["code"]["coding"][0]["display"],
            "severity": resource.get("reaction", [{}])[0].get("severity")
        })

    elif rtype == "Claim":
        claims.append({
            "claim_id": resource["id"],
            "patient_id": clean_ref(resource["patient"]["reference"]),
            "amount": resource["total"]["value"],
            "provider": resource["provider"]["display"],
            "encounter_id": clean_ref(resource["item"][0]["encounter"][0]["reference"])
        })

# Convert to DataFrames
dfs = {
    "patients": pd.DataFrame(patients),
    "encounters": pd.DataFrame(encounters),
    "conditions": pd.DataFrame(conditions),
    "allergies": pd.DataFrame(allergies),
    "claims": pd.DataFrame(claims),
}

# Save CSVs
paths = {}
for name, df in dfs.items():
    path = f"/mnt/data/{name}.csv"
    df.to_csv(path, index=False)
    paths[name] = path

