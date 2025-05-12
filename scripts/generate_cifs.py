from mp_api.client import MPRester
import os
import random

# Replace with your Materials Project API key
API_KEY = "yvC2UCUjpLKJgwZ4Vhx5xFKHvVfOiF7k"

# Folder to save CIFs
SAVE_DIR = "cifs"
os.makedirs(SAVE_DIR, exist_ok=True)

with MPRester(API_KEY) as mpr:
    # Fetch 100 materials with 0.0 energy above hull
    docs = mpr.summary.search(
        energy_above_hull=(0.0, 0.0),
        num_chunks=None,
    )
    random_sample = random.sample(docs, 100)

    print(f"Found {len(docs)} stable materials. Downloading CIFs...")

    for idx, doc in enumerate(docs, start=1):
        try:
            structure = mpr.get_structure_by_material_id(doc.material_id)
            cif_path = os.path.join(SAVE_DIR, f"{doc.material_id}.cif")
            structure.to(fmt="cif", filename=cif_path)
            print(f"[{idx:03}] Saved {doc.material_id}.cif")
        except Exception as e:
            print(f"Error saving {doc.material_id}: {e}")
