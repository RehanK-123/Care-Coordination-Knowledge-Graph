import json
import subprocess
import tempfile
import os

# ── CONFIG ───────────────────────────────────────────────
VALIDATOR_JAR = "./validator_cli.jar"
FHIR_VERSION = "4.0.1"


# ── VALIDATE FULL FILE (OPTIMIZED) ───────────────────────
def validate_fhir_file(fhir_json):
    """
    Validate full FHIR file (Bundle or resource) using HL7 validator
    """

    # Write entire JSON to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(fhir_json, tmp)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                "java",
                
                "-Xmx2G",
                "-jar",
                VALIDATOR_JAR,
                tmp_path,
                "-version",
                FHIR_VERSION,
                "-ig",
                "hl7.fhir.us.core",   # 🔥 US CORE VALIDATION
                "-tx",
                "n/a"                 # skip terminology server
            ],
            capture_output=True,
            text=True
        )

        output = result.stdout + result.stderr

        # ✅ ONLY treat real "errors" as invalid
        if "error" in output.lower():
            return False, output

        return True, output

    finally:
        os.remove(tmp_path)


# ── RUN VALIDATION ───────────────────────────────────────
DATA_FOLDER = "synthea_sample_data_fhir_latest"

files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]

valid_count = 0
invalid_count = 0

for file in files[:5]:   # test first 5 files
    file_path = os.path.join(DATA_FOLDER, file)

    with open(file_path) as f:
        data = json.load(f)

    is_valid, output = validate_fhir_file(data)

    if is_valid:
        print(f"✅ {file} is valid FHIR.")
        valid_count += 1
    else:
        print(f"❌ {file} is invalid FHIR.")
        invalid_count += 1

        # 🔥 Print only summary (not full huge logs)
        for line in output.split("\n"):
            if "error" in line.lower():
                print("   →", line.strip())

    print("-" * 50)


# ── SUMMARY ──────────────────────────────────────────────
print("\n📊 Validation Summary")
print(f"Valid Files   : {valid_count}")
print(f"Invalid Files : {invalid_count}")
print("\n✅ Validation complete")