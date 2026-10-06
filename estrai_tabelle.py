import re

FILE_JSON = "logsprova.json"

tabelle = set()
dentro_resource_context = False

with open(FILE_JSON, "r", encoding="utf-8", errors="ignore") as f:
    for riga in f:

        if '"ResourceContext"' in riga:
            dentro_resource_context = True
            continue

        if dentro_resource_context:
            match = re.search(r'"table"\s*:\s*"([^"]+)"', riga)

            if match:
                tabelle.add(match.group(1))

            # ResourceContext nel tuo JSON è un oggetto semplice
            if "}" in riga:
                dentro_resource_context = False

print(f"Tabelle uniche trovate: {len(tabelle)}")
print()

for tabella in sorted(tabelle):
    print(tabella)