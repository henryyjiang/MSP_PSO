from ase.geometry import cell_to_cellpar, cellpar_to_cell
import ase
import numpy as np
import matplotlib.pyplot as plt
from mattersim.forcefield.potential import Potential, MatterSimCalculator
from mace.calculators import mace_mp
from ase.optimize import BFGS, FIRE
import torch
import torch_sim as ts
from torch_sim.models.mace import MaceModel
from torch_sim.models.mattersim import MatterSimModel
from mattersim.forcefield.m3gnet import m3gnet
import json
import logging
import sys
import os
import time
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pathlib import Path

from utils import *

logging.getLogger("mattertune").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("pandas").setLevel(logging.CRITICAL)

device = "cuda" if torch.cuda.is_available() else "cpu"


class BasinHopping():
    def __init__(self, cif_name, model, composition, cell, calc, iters, local_steps,
                 step_size=0.5, temperature=1.0, cell_perturb=True,
                 adapt_interval=20, restart_interval=50):
        self.cif_name = cif_name
        self.cell_perturb = cell_perturb
        self.composition = composition
        self.cell = cell

        self.iters = iters
        self.local_steps = local_steps
        self.step_size = step_size
        self.temperature = temperature

        self.adapt_interval = adapt_interval
        self.accept_count = 0
        self.attempt_count = 0
        self.step_size_min = 0.05
        self.step_size_max = 2.0

        self.restart_interval = restart_interval

        self.zs, self.zcounts = composition_to_zs(self.composition)
        self.possible_sgs, self.sg_probs = generate_sgs(self.zs, self.zcounts)
        self.el_symbols = np.array([periodictable.elements[i].symbol for i in range(95)])
        self.lj_rmins = np.genfromtxt(str(Path(__file__).parent / "lj_rmins.csv"),
                                      delimiter=",")

        self.best_loss = float('inf')
        self.current_loss = float('inf')
        self.loss_history = []
        self.best_losses = []

        self.model = model

        if calc == "mace":
            self.calculator = mace_mp(model="large", device=device)
        else:
            ckpt = torch.load("mattersim-v1.0.0-5M.pth", map_location=device)
            m3gnet_model = m3gnet.M3Gnet(**ckpt["model_args"])
            state_dict = ckpt["model"]
            m3gnet_model.load_state_dict(state_dict=state_dict, strict=False)
            self.calculator = MatterSimCalculator(potential=Potential(m3gnet_model), device=device)

    def adapt_step_size(self):
        if self.attempt_count > 0 and self.attempt_count % self.adapt_interval == 0:
            rate = self.accept_count / self.adapt_interval
            if rate > 0.5:
                self.step_size *= 1.1
            elif rate < 0.3:
                self.step_size *= 0.9
            self.step_size = np.clip(self.step_size, self.step_size_min, self.step_size_max)
            print(f"  [Adapt] Acceptance rate: {rate:.2f}, new step size: {self.step_size:.4f}")
            self.accept_count = 0
            self.attempt_count = 0

    def perturb_structure(self, atoms):
        new_atoms = atoms.copy()


        cell_lengths = np.linalg.norm(new_atoms.cell[:], axis=1)  # (3,)
        frac_step = self.step_size / cell_lengths 

        frac = new_atoms.get_scaled_positions()
        perturbation = np.random.randn(*frac.shape) * frac_step[np.newaxis, :]
        frac += perturbation
        frac %= 1.0
        new_atoms.set_scaled_positions(frac)

        if self.cell_perturb:
            cell = new_atoms.get_cell()
            cell_perturbation = np.random.randn(3, 3) * self.step_size * 0.1
            new_cell = cell + cell_perturbation

            cellpar = cell_to_cellpar(new_cell)
            cellpar[:3] = np.clip(cellpar[:3], 3.0, 100.0)
            cellpar[3:] = np.clip(cellpar[3:], 30.0, 150.0) 

            alpha, beta, gamma = cellpar[3], cellpar[4], cellpar[5]
            if not (alpha + beta <= gamma or alpha + gamma <= beta or beta + gamma <= alpha):
                try:
                    new_cell = cellpar_to_cell(cellpar)
                    new_atoms.set_cell(new_cell, scale_atoms=False)
                except (AssertionError, ValueError):
                    pass  # keep original cell on failure

        return new_atoms

    def sanitize_atoms(self, atoms):
        try:
            cellpar = cell_to_cellpar(atoms.cell)
            cellpar[3:] = np.clip(cellpar[3:], 30.0, 150.0)
            cell = cellpar_to_cell(cellpar)

            lengths = np.linalg.norm(cell, axis=1)
            if np.any(lengths < 3.0) or np.any(lengths > 100.0):
                lengths = np.clip(lengths, 3.0, 100.0)
                for i in range(3):
                    cell[i] = cell[i] / np.linalg.norm(cell[i]) * lengths[i]

            atoms.set_cell(cell, scale_atoms=True)

            separate_close_atoms2(atoms)

            if not np.all(np.isfinite(atoms.get_forces())):
                atoms.positions += 1e-3 * np.random.randn(*atoms.positions.shape)

        except Exception:
            pass

        return atoms

    def accept_move(self, new_energy, old_energy):
        if new_energy < old_energy:
            return True

        delta_e = new_energy - old_energy
        probability = np.exp(-delta_e / self.temperature)
        return np.random.rand() < probability

    def local_optimize(self, atoms):
        atoms = self.sanitize_atoms(atoms)
        atoms.calc = self.calculator

        try:
            optimized_state = ts.optimize(
                system=atoms,
                model=self.model,
                optimizer=ts.frechet_cell_fire,
                autobatcher=False,
                max_steps=self.local_steps
            )
            optimized_atoms_list = optimized_state.to_atoms()
            optimized_atoms = (optimized_atoms_list[0]
                               if isinstance(optimized_atoms_list, list)
                               else optimized_atoms_list)
            optimized_atoms.calc = self.calculator

            if validate_structure_distances(optimized_atoms):
                energy = optimized_atoms.get_potential_energy()
            else:
                separate_close_atoms2(optimized_atoms)
                energy = optimized_atoms.get_potential_energy()

            return optimized_atoms, energy

        except Exception as e:
            print(f"Optimization failed: {e}")
            atoms.calc = self.calculator
            try:
                energy = atoms.get_potential_energy()
                return atoms, energy
            except Exception:
                return atoms, float('inf')

    def run(self):
        os.makedirs("plots", exist_ok=True)
        os.makedirs("matches", exist_ok=True)
        os.makedirs("fails", exist_ok=True)
        os.makedirs("lower_energy", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

        matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8)

        original_cif = os.path.join("cifs", self.cif_name + ".cif")
        ground_truth = Structure.from_file(original_cif)

        atoms_gt = AseAtomsAdaptor.get_atoms(ground_truth)
        atoms_gt.calc = self.calculator
        ground_truth_energy = atoms_gt.get_potential_energy()

        init_atoms = initialize_atoms(self.el_symbols, self.lj_rmins, self.zs,
                                      self.zcounts, self.possible_sgs, self.sg_probs)

        # Initial local optimization
        current_atoms, self.current_loss = self.local_optimize(init_atoms)
        best_atoms = current_atoms.copy()
        self.best_loss = self.current_loss

        print(f"Initial energy: {self.current_loss:.4f} eV  "
              f"(ground truth: {ground_truth_energy:.4f} eV)")

        # Basin hopping main loop
        for i in range(self.iters):
            start_time = time.time()

            if i > 0 and i % self.restart_interval == 0:
                current_atoms = best_atoms.copy()
                self.current_loss = self.best_loss
                print(f"  [Restart] Returning to best: {self.best_loss:.4f} eV")

            # Perturb
            perturbed_atoms = self.perturb_structure(current_atoms)

            # Local optimization
            optimized_atoms, new_energy = self.local_optimize(perturbed_atoms)

            # Metropolis acceptance
            self.attempt_count += 1
            if self.accept_move(new_energy, self.current_loss):
                self.accept_count += 1
                current_atoms = optimized_atoms
                self.current_loss = new_energy

                if new_energy < self.best_loss:
                    self.best_loss = new_energy
                    best_atoms = optimized_atoms.copy()
                    print(f"  *** New best: {self.best_loss:.4f} eV")

            self.adapt_step_size()

            self.loss_history.append(self.current_loss)
            self.best_losses.append(self.best_loss)

            print(f"Iteration {i + 1}/{self.iters}: "
                  f"GT: {ground_truth_energy:.4f}  "
                  f"Best: {self.best_loss:.4f}  "
                  f"Current: {self.current_loss:.4f}  "
                  f"Step: {self.step_size:.3f}  "
                  f"Time: {time.time() - start_time:.2f}s")

            # Checkpoint every 50 iterations so long runs survive crashes
            if (i + 1) % 50 == 0:
                ase.io.write(
                    f"checkpoints/{self.cif_name}_iter{i + 1}.cif", best_atoms
                )

        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Current Energy', alpha=0.7)
        plt.plot(self.best_losses, label='Best Energy', linewidth=2)
        plt.axhline(y=ground_truth_energy, color='r', linestyle='--', label='Ground Truth')
        plt.xlabel('Iteration')
        plt.ylabel('Energy (eV)')
        plt.title(f'Basin Hopping: {self.cif_name}')
        plt.legend()
        plt.savefig(f'plots/basin_hopping_{self.cif_name}.png')
        plt.close()

        # Evaluate final structure
        try:
            optimized_structure = AseAtomsAdaptor.get_structure(best_atoms)

            best_atoms.calc = self.calculator
            optimized_energy = best_atoms.get_potential_energy()

            energy_tolerance = 0.1
            distance_threshold = 2.5

            energy_diff = abs(optimized_energy - ground_truth_energy)
            try:
                distance = matcher.get_rms_dist(ground_truth, optimized_structure)[0]
            except Exception:
                distance = np.linalg.norm(
                    optimized_structure.frac_coords - ground_truth.frac_coords
                ).mean()

            if distance < distance_threshold and energy_diff <= energy_tolerance:
                result_type = "match"
                print(f"Matched for {self.cif_name}: RMSD = {distance:.3f}, ΔE = {energy_diff:.3f} eV")
            elif optimized_energy - ground_truth_energy < 0:
                result_type = "lower_energy"
                print(f"Lower energy for {self.cif_name}: RMSD = {distance:.3f}, ΔE = {energy_diff:.3f} eV")
            else:
                result_type = "fail"
                print(f"No match for {self.cif_name}: RMSD = {distance:.3f}, ΔE = {energy_diff:.3f} eV")

            out_dir = {"match": "matches", "lower_energy": "lower_energy", "fail": "fails"}[result_type]
            ase.io.write(os.path.join(out_dir, f"best_structure_{self.cif_name}.cif"), best_atoms)

            match = result_type in ("match", "lower_energy")

        except ValueError:
            print(f"{self.cif_name}: invalid structure, cannot match")
            match = False
            distance, energy_diff = float('inf'), float('inf')
            ase.io.write(os.path.join("fails", f"best_structure_{self.cif_name}.cif"), best_atoms)

        # Save energy history
        with open(f"plots/{self.cif_name}_costs.txt", "w") as f:
            f.write(f"Ground Truth: {ground_truth_energy}\n")
            f.write(f"Best Energy: {self.best_loss}\n")

        return match, (distance, energy_diff)


if __name__ == "__main__":
    calc = "mattersim"

    if calc == "mace":
        mace = mace_mp(model="large", return_raw_model=True)
        model = MaceModel(model=mace)
    else:
        ckpt = torch.load("mattersim-v1.0.0-5M.pth", map_location=device)
        m3gnet_model = m3gnet.M3Gnet(**ckpt["model_args"])
        state_dict = ckpt["model"]
        m3gnet_model.load_state_dict(state_dict=state_dict, strict=False)
        model = MatterSimModel(model=Potential(m3gnet_model))

    all_matches = []
    all_dist_energy = []

    for filename in os.listdir("cifs"):
        cif = os.path.join("cifs", filename)
        cif_name = os.path.splitext(filename)[0]

        composition = extract_composition(cif)
        cell = extract_cell(cif)

        iters = 500
        local_steps = 100
        step_size = 0.5
        temperature = 1.0
        cell_perturb = True
        adapt_interval = 20
        restart_interval = 50

        bh = BasinHopping(cif_name, model, composition, cell, calc, iters,
                          local_steps, step_size, temperature, cell_perturb,
                          adapt_interval, restart_interval)
        match, dist_energy = bh.run()
        all_matches.append(match)
        all_dist_energy.append(dist_energy)
        print(f"{cif_name} match: {match}")

    num_true = sum(all_matches)
    total = len(all_matches)
    match_ratio = num_true / total if total > 0 else 0

    print("\nDistance and Energy Differences:")
    print(all_dist_energy)

    print(f"\nMatched {num_true}/{total} structures.")
    print(f"Match ratio: {match_ratio:.2f}")