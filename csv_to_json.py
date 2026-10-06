import csv
import json
import ast

def csv_to_json(csv_file, json_file):
    # Fonti da considerare per la conversione in JSON
    allowed_sources = [
        "p0f",
        "Blazor:2.0.0.2:AuthAudit",
        "Blazor:2.0.0.2:ApplicationAudit",
        "captiveportal"
    ]

    data = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Controllo se il source è tra quelli consentiti
            if row["source"] not in allowed_sources:
                continue  # salta questo log

            # Copia dei campi standard
            item = {
                "time": row["time"],
                "host": row["host"],
                "source": row["source"],
                "sourcetype": row["sourcetype"]
            }

            raw_message = row["message"]

            # Prova a convertire il message in JSON
            try:
                try:
                    # Primo tentativo: json.loads
                    item["message"] = json.loads(raw_message)
                except json.JSONDecodeError:
                    # Fallback: valuta come dizionario python "quasi JSON"
                    item["message"] = ast.literal_eval(raw_message)
            except Exception as e:
                print("[ERRORE JSON]", e)
                print("Message fallito:", raw_message)
                item["message"] = raw_message

            data.append(item)

    # Salvataggio JSON formattato
    with open(json_file, "w", encoding='utf-8') as out:
        json.dump(data, out, indent=4, ensure_ascii=False)

# ---------- ESEMPIO UTILIZZO ----------
csv_to_json("final.csv", "logsprova.json")