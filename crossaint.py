import json

import requests
headers = {"Authorization": f"Bearer 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"}
API_URL = "https://huggingface.co/api/datasets/nabinpakka07/Foliagen/croissant"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()

with open("duorc_croissant_metadata.json", "w") as f:
    json.dump(data, f, indent=4)

print("Data saved to duorc_croissant_metadata.json")