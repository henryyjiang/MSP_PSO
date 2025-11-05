from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
import random

# Replace with your Materials Project API key
API_KEY = "yvC2UCUjpLKJgwZ4Vhx5xFKHvVfOiF7k"

# Folder to save CIFs
SAVE_DIR = "cifs"
os.makedirs(SAVE_DIR, exist_ok=True)

total_structs = 10
max_atoms = 20

with MPRester(API_KEY) as mpr:
    # Fetch 100 materials with 0.0 energy above hull
    docs = mpr.summary.search(
        energy_above_hull=(0.0, 0.01),
        num_sites=(1,max_atoms),
        num_chunks=None,
    )

    docs = [
        d for d in docs
        if not getattr(d, "is_disordered", False)
    ]

    random_sample = random.sample(docs, total_structs)

    print(f"Found {len(random_sample)} stable materials. Downloading CIFs...")

    for idx, doc in enumerate(random_sample, start=1):
        try:
            structure = mpr.get_structure_by_material_id(doc.material_id)
            sga = SpacegroupAnalyzer(structure)
            structure_std = sga.get_conventional_standard_structure()

            cif_path = os.path.join(SAVE_DIR, f"{doc.material_id}.cif")
            structure_std.to(fmt="cif", filename=cif_path)

            print(f"[{idx:03}] Saved standardized {doc.material_id}.cif")
        except Exception as e:
            print(f"Error saving {doc.material_id}: {e}")
