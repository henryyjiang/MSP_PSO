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
        atoms = Atoms(composition, cell=cell[i], pbc=(True, True, True), positions=frac_positions)
        # frac_positions = frac_positions % 1.0
        # atoms = Atoms(composition, cell=cell[i], pbc=(True, True, True), scaled_positions=frac_positions)
    else:
        cell = params[:9].reshape(-1, 3)
        positions = params[9:].reshape(-1, 3)
        atoms = Atoms(composition, cell=cell, pbc=(True, True, True), positions=positions)

    if not hasattr(atoms, 'calc') or atoms.calc is None:
        atoms.set_calculator(calculator)
    return atoms


def atoms_to_dimensions(atoms, cell_perturb):
    if not cell_perturb:
        pos = [float(i) for l in atoms.positions for i in l]
        # pos = atoms.get_scaled_positions().flatten()
    else:
        pos = [float(i) for l in atoms.cell for i in l][:9] + [float(i) for l in atoms.positions for i in l]

    return pos


def final_dimensions(params, best_cell, composition, cell_perturb=True):
    if cell_perturb:
        actual_cell = params[:9].reshape(3, 3)
        coords = params[9:].reshape(-1, 3)
        atoms = Atoms(composition, cell=actual_cell, pbc=(True, True, True), positions=coords)

    else:
        actual_cell = best_cell
        coords = params.reshape(-1, 3)

        # atoms = Atoms(composition, cell=actual_cell, pbc=(True, True, True), scaled_positions=coords)
        atoms = Atoms(composition, cell=actual_cell, pbc=(True, True, True), positions=coords)
    return atoms


def calculate_lj_forces(atoms, lj_rmins, cutoff_factor=1.5, epsilon=1.0, min_distance=0.5, step_scale=0.1):
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)
    atomic_numbers = atoms.get_atomic_numbers()
    n_atoms = len(atoms)
    forces = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            z_i = atomic_numbers[i] - 1
            z_j = atomic_numbers[j] - 1
            sigma = lj_rmins[z_i, z_j]

            delta = positions[j] - positions[i]
            delta = delta - np.round(delta @ inv_cell) @ cell
            r = np.linalg.norm(delta)

            r = max(r, min_distance)

            if r < sigma * cutoff_factor:
                sr6 = (sigma / r) ** 6
                sr12 = sr6 ** 2
                force_magnitude = 24 * epsilon * (2 * sr12 - sr6) / r

                max_force = 50.0
                force_magnitude = np.clip(force_magnitude, -max_force, max_force)

                force_vector = force_magnitude * (delta / r)

                forces[i] -= force_vector
                forces[j] += force_vector

    atoms.positions += step_scale * forces
    atoms.wrap()

    return atoms


def separate_close_atoms(atoms, lj_rmins, max_iters=10, min_dist=1.0, step_scale=0.2):
    epsilon = 1.0

    numbers = atoms.numbers - 1

    for iteration in range(max_iters):
        i, j, dists, vecs = neighbor_list('ijdD', atoms, cutoff=4.0)

        if len(dists) == 0:
            break

        sigmas = lj_rmins[numbers[i], numbers[j]]
        targets = np.maximum(sigmas, min_dist)
        mask = (i < j) & (dists < targets)
        if not np.any(mask):
            break

        idx_i, idx_j = i[mask], j[mask]
        r = dists[mask][:, np.newaxis]
        delta = vecs[mask]
        target_r = targets[mask][:, np.newaxis]

        # sr6 = (target_r / (r + 1e-9)) ** 6
        # sr12 = sr6 ** 2
        # repulsion_mag = (24 * epsilon * sr12) / (r + 1e-9)
        #
        # max_f = 50.0
        # repulsion_mag = np.clip(repulsion_mag, 0, max_f)


        repulsion_mag = (target_r / (r + 1e-6)) ** 2 - 1.0
        repulsion_mag = np.minimum(repulsion_mag, 5.0)

        force_vecs = repulsion_mag * (delta / r)

        total_forces = np.zeros((len(atoms), 3))
        np.add.at(total_forces, idx_i, -force_vecs)
        np.add.at(total_forces, idx_j, force_vecs)

        critical = (dists < 0.1) & (i < j)
        if np.any(critical):
            rand_kicks = np.random.randn(len(i[critical]), 3) * 0.5
            np.add.at(total_forces, i[critical], -rand_kicks)
            np.add.at(total_forces, j[critical], rand_kicks)

        atoms.positions += step_scale * total_forces
        atoms.wrap()

    return atoms


def separate_close_atoms2(atoms, min_dist=1.0, max_iterations=10):
    cutoff = 3.0

    for _ in range(max_iterations):
        i, j, d, D = neighbor_list('ijdD', atoms, cutoff=cutoff)

        if len(d) == 0:
            break
        radii_sum = covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]]
        min_allowed = np.maximum(0.75 * radii_sum, min_dist)

        mask = (i < j) & (d < min_allowed) & (d > 1e-9)

        if not np.any(mask):
            return True

        idx_i, idx_j = i[mask], j[mask]
        actual_dist = d[mask][:, np.newaxis]
        unit_vecs = D[mask] / actual_dist
        target_dist = min_allowed[mask][:, np.newaxis]
        shift_mag = 0.5 * (target_dist - actual_dist)
        shift_vecs = shift_mag * unit_vecs

        pos_delta = np.zeros_like(atoms.positions)
        np.add.at(pos_delta, idx_i, -shift_vecs)
        np.add.at(pos_delta, idx_j, shift_vecs)

        atoms.positions += pos_delta

    final_i, final_j, final_d = neighbor_list('ijd', atoms, cutoff=min_dist)
    return not (len(final_d) > 0 and np.min(final_d) < min_dist)


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