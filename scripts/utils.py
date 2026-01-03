import numpy as np
from collections import Counter
from pyxtal import pyxtal
import periodictable
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from pymatgen.core import Structure
from pathlib import Path
from pymatgen.io.ase import AseAtomsAdaptor

def extract_cell(cif_path):
    structure = Structure.from_file(cif_path)
    cell = structure.lattice.matrix
    return cell.tolist()

def extract_composition(cif_path):
    structure = Structure.from_file(cif_path)
    composition = [site.specie.Z for site in structure]
    return composition

def composition_to_zs(composition):
    counter = Counter(composition)
    zs = [periodictable.elements[z].symbol for z, _ in counter.items()]
    zcounts = [count for _, count in counter.items()]
    return zs, zcounts

def generate_sgs(zs, zcounts, seed=0):
    sg_dist = [146, 2720, 14, 407, 178, 7, 145, 97, 351, 39, 768, 1688, 286,
               5736, 2345, 1, 5, 66, 596, 89, 13, 2, 11, 0, 10, 64, 3, 9, 196, 11,
               185, 19, 471, 26, 8, 309, 6, 91, 15, 70, 47, 18, 164, 64, 14, 60, 30,
               0, 1, 3, 111, 69, 27, 33, 459, 76, 248, 265, 245, 327, 634, 4129, 1480,
               342, 237, 34, 21, 23, 59, 220, 449, 205, 36, 206, 5, 29, 3, 10, 12, 5,
               12, 157, 12, 31, 64, 95, 221, 204, 0, 7, 8, 107, 0, 1, 5, 41, 5, 5, 17,
               28, 0, 13, 1, 1, 5, 2, 105, 14, 45, 20, 14, 5, 94, 46, 15, 10, 9, 18,
               26, 9, 116, 126, 405, 31, 51, 15, 434, 113, 880, 75, 44, 7, 8, 11, 35,
               289, 79, 26, 1575, 530, 281, 133, 19, 23, 14, 73, 98, 633, 9, 71, 10,
               83, 1, 28, 57, 215, 22, 6, 48, 158, 112, 48, 77, 571, 73, 1047, 405, 0,
               0, 0, 0, 1, 306, 125, 3, 297, 0, 0, 0, 32, 17, 40, 3, 4, 69, 412, 108,
               20, 644, 51, 528, 8, 418, 1567, 0, 2, 16, 245, 27, 31, 47, 22, 11, 170,
               234, 68, 0, 2, 0, 0, 0, 23, 44, 17, 58, 509, 118, 71, 18, 176, 1116, 2,
               210, 34, 1563, 45, 788, 9, 210, 105]

    rng = np.random.default_rng(seed)
    possible_sgs = []
    for i in range(230):
        try:
            xtal = pyxtal()
            xtal.from_random(3, i + 1, zs, zcounts,
                             random_state=rng)
            possible_sgs.append(i + 1)
        except:
            continue
    possible_sgs = np.array(possible_sgs)
    sg_probs = np.array([sg_dist[i - 1] + 1.0 for i in possible_sgs])
    sg_probs /= np.sum(sg_probs)

    return possible_sgs, sg_probs

def get_z(el_symbols, site):
    return np.argmax(el_symbols == site.species.elements[0].symbol)

def lj_reject(el_symbols, lj_rmins, structure):
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            if structure.sites[i].distance(structure.sites[j]) < lj_rmins[get_z(el_symbols,
                    structure.sites[i]) - 1][get_z(el_symbols, structure.sites[j]) - 1]:
                return True
    return False

def initialize_atoms(el_symbols, lj_rmins, zs, zcounts, possible_sgs, sg_probs, density=0.2):
    seed = 0
    rng = np.random.default_rng(seed)
    rejected = True
    while rejected:
        try:
            xtal = pyxtal()
            xtal.from_random(3, np.random.choice(possible_sgs,
                                                 p=sg_probs), zs, zcounts, random_state=rng)
            new_structure = xtal.to_pymatgen()
            rejected = lj_reject(el_symbols, lj_rmins, new_structure)
        except:
            rejected = True

    atoms = xtal.to_ase()
    return atoms

def dimensions_to_atoms(params, i, composition, cell, calculator, cell_perturb):
    if not cell_perturb:
        frac_positions = params.reshape(-1, 3)
        frac_positions = frac_positions % 1.0
        atoms = Atoms(composition, cell=cell[i], pbc=(True, True, True),
                      scaled_positions=frac_positions)
    else:
        cell = params[:9].reshape(-1, 3)
        positions = params[9:].reshape(-1, 3)
        atoms = Atoms(composition, cell=cell, pbc=(True, True, True), positions=positions)

    if not hasattr(atoms, 'calc') or atoms.calc is None:
        atoms.set_calculator(calculator)
    return atoms

def final_dimensions(params, best_cell, composition):
    frac_positions = params.reshape(-1, 3)
    atoms = Atoms(composition, cell=best_cell, pbc=(True, True, True),
                 scaled_positions=frac_positions)
    return atoms


def atoms_to_dimensions(atoms, cell_perturb):
    if not cell_perturb:
        pos = atoms.get_scaled_positions().flatten()
    else:
        cell_flat = atoms.cell.array.flatten()[:9]
        pos_flat = atoms.positions.flatten()
        pos = np.concatenate([cell_flat, pos_flat])

    return pos

def lj_repulsion_pymatgen(structure, scale = 40, buffer = 0.85):
  lj_rmins = np.genfromtxt(str(Path(__file__).parent / "lj_rmins.csv"),
                             delimiter=",")
  repulsions = []

  def get_z_site(site):
      el_symbols = np.array([periodictable.elements[i].symbol for i in range(95)])
      return np.argmax(el_symbols == site.species.elements[0].symbol)

  for i in range(len(structure)):
    for j in range(i, len(structure)):
      rmin = lj_rmins[get_z_site(structure.sites[i]) - 1, get_z_site(
        structure.sites[j]) - 1] * buffer
      r = np.min([structure.lattice.a, structure.lattice.b,
        structure.lattice.c]) if i == j else structure.sites[i].distance(
        structure.sites[j])
      repulsions.append(max(0, (rmin / r) ** 12 - 1))
  return np.mean(repulsions) / scale


def calculate_lj_forces(atoms, lj_rmins, cutoff_factor=1.5, epsilon=1.0, min_distance=0.5):
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    atomic_numbers = atoms.get_atomic_numbers()
    n_atoms = len(atoms)
    forces = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            z_i = atomic_numbers[i] - 1
            z_j = atomic_numbers[j] - 1
            sigma = lj_rmins[z_i, z_j]  # Use as sigma parameter

            delta = positions[j] - positions[i]
            delta = delta - np.round(delta @ np.linalg.inv(cell)) @ cell
            r = np.linalg.norm(delta)

            r = max(r, min_distance)

            if r < sigma * cutoff_factor:
                # Full LJ force: F = 24ε/r * [2(σ/r)^12 - (σ/r)^6]
                sr6 = (sigma / r) ** 6
                sr12 = sr6 ** 2
                force_magnitude = 24 * epsilon * (2 * sr12 - sr6) / r

                # Cap extreme forces
                max_force = 50.0
                force_magnitude = np.clip(force_magnitude, -max_force, max_force)

                force_vector = force_magnitude * (delta / r)

                forces[i] -= force_vector
                forces[j] += force_vector

    return forces


def separate_close_atoms(atoms, max_iters=5, min_dist=1.0, step_scale=0.1):
    from ase.neighborlist import neighbor_list

    lj_rmins = np.genfromtxt(str(Path(__file__).parent / "lj_rmins.csv"),
                             delimiter=",")

    for iteration in range(max_iters):
        atomic_numbers = atoms.get_atomic_numbers()
        n_atoms = len(atoms)

        try:
            indices_i, indices_j, distances, vecs = neighbor_list(
                'ijdD', atoms, cutoff=min_dist * 2.0
            )
        except Exception as e:
            print(f"Neighbor list failed: {e}")
            break

        if len(distances) == 0:
            break

        forces = np.zeros((n_atoms, 3))
        max_violation = 0.0

        # Process each neighbor pair
        for idx in range(len(distances)):
            i = indices_i[idx]
            j = indices_j[idx]
            r = distances[idx]
            delta = vecs[idx]

            if i == j:
                continue

            # Safety check
            if r < 0.01 or not np.isfinite(r):
                # Emergency separation
                rand_dir = np.random.randn(3)
                rand_dir /= np.linalg.norm(rand_dir)
                forces[i] -= 10.0 * rand_dir
                forces[j] += 10.0 * rand_dir
                max_violation = max(max_violation, 5.0)
                continue

            z_i = atomic_numbers[i] - 1
            z_j = atomic_numbers[j] - 1
            sigma = lj_rmins[z_i, z_j]
            target_dist = max(sigma, min_dist)

            if r < target_dist:
                violation = target_dist - r
                max_violation = max(max_violation, violation)

                force_magnitude = min(violation / r, 10.0)
                direction = delta / r
                force_vector = force_magnitude * direction

                forces[i] -= force_vector
                forces[j] += force_vector

        if max_violation < 0.01:
            break

        total_force = np.linalg.norm(forces)
        if total_force > 0:
            step_size = min(step_scale, step_scale * 10.0 / (total_force + 1.0))
        else:
            break

        new_positions = atoms.get_positions() + step_size * forces

        if np.any(~np.isfinite(new_positions)):
            print(f"Warning: Non-finite positions at iteration {iteration}")
            break

        atoms.set_positions(new_positions)

    return atoms

def separate_close_atoms2(atoms, min_dist=1.0, max_iterations=5):
    cutoff = 4.0

    for iteration in range(max_iterations):
        indices_i, indices_j, distances = neighbor_list('ijd', atoms, cutoff=cutoff)

        if len(distances) == 0:
            return True

        min_d = np.min(distances)
        if min_d >= min_dist:
            return True

        elem_nums_i = atoms.numbers[indices_i]
        elem_nums_j = atoms.numbers[indices_j]
        min_allowed = 0.75 * (covalent_radii[elem_nums_i] + covalent_radii[elem_nums_j])

        min_allowed = np.maximum(min_allowed, min_dist)

        needs_adjustment = distances < min_allowed

        if not np.any(needs_adjustment):
            return True

        for idx in np.where(needs_adjustment)[0]:
            i = indices_i[idx]
            j = indices_j[idx]
            d = distances[idx]
            target = min_allowed[idx]

            vec = atoms.positions[j] - atoms.positions[i]
            vec_norm = np.linalg.norm(vec)

            if vec_norm < 1e-6:
                vec = np.random.randn(3)
                vec_norm = np.linalg.norm(vec)

            vec = vec / vec_norm

            shift = 0.3 * (target - d) * vec
            atoms.positions[i] -= shift
            atoms.positions[j] += shift

    indices_i, indices_j, distances = neighbor_list('ijd', atoms, cutoff=cutoff)

    if len(distances) > 0 and np.min(distances) < min_dist:
        #print("reject1")
        return False

    return True


def separate_close_atoms_batch(atoms_list, min_dist=1.0):
    """Vectorized version for multiple structures"""
    # Process all at once using numpy broadcasting
    for atoms in atoms_list:  # Still sequential but optimized
        # Use faster neighbor list
        from matscipy.neighbours import neighbour_list
        i, j, d = neighbour_list('ijd', atoms, cutoff=4.0)

        # Vectorized distance checks
        mask = d < min_dist
        if not np.any(mask):
            continue

        # Vectorized position adjustments
        vecs = atoms.positions[j[mask]] - atoms.positions[i[mask]]
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        shift = 0.3 * (min_dist - d[mask])[:, None] * vecs

        # Update positions (use bincount for multiple atoms)
        pos_delta = np.zeros_like(atoms.positions)
        np.add.at(pos_delta, i[mask], -shift)
        np.add.at(pos_delta, j[mask], shift)

        atoms.positions += pos_delta


def validate_structure_distances(atoms, min_dist=1.0):
    cutoff = 3.0
    indices_i, indices_j, distances = neighbor_list('ijd', atoms, cutoff=cutoff)
    if len(distances) == 0:
        return True

    min_d = np.min(distances)
    if min_d < min_dist:
        #print("reject2")
        return False

    return True