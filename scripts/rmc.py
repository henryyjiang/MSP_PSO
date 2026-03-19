from ase.geometry import cell_to_cellpar, cellpar_to_cell
import ase
import numpy as np
import matplotlib.pyplot as plt
from mattersim.forcefield.potential import Potential, MatterSimCalculator
from mace.calculators import mace_mp
import torch
import torch_sim as ts
from torch_sim.models.mace import MaceModel
from torch_sim.models.mattersim import MatterSimModel
from mattersim.forcefield.m3gnet import m3gnet
import logging
import os
import time
import copy
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pathlib import Path

from utils import *

logging.getLogger("mattertune").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("pandas").setLevel(logging.CRITICAL)

device = "cuda" if torch.cuda.is_available() else "cpu"


class RMC():
    def __init__(self, cif_name, model, composition, cell, calc, iters, local_steps,
                 step_size=0.3, temperature=1.0, cell_perturb=True,
                 adapt_interval=50, restart_interval=200,
                 energy_opt_threshold=0.5,
                 calculator=None):
        self.cif_name = cif_name
        self.cell_perturb = cell_perturb
        self.composition = composition
        self.cell = cell

        self.iters = iters
        self.local_steps = local_steps
        self.temperature = temperature

        self.pos_step = step_size
        self.len_step = step_size * 0.5
        self.ang_step = step_size * 3.0

        self.adapt_interval = adapt_interval
        self._reset_accept_counters()

        self.step_min = 0.01
        self.step_max = 3.0

        self.restart_interval = restart_interval

        self.energy_opt_threshold = energy_opt_threshold

        self.zs, self.zcounts = composition_to_zs(self.composition)
        self.possible_sgs, self.sg_probs = generate_sgs(self.zs, self.zcounts)
        self.el_symbols = np.array([periodictable.elements[i].symbol for i in range(95)])
        self.lj_rmins = np.genfromtxt(str(Path(__file__).parent / "lj_rmins.csv"),
                                      delimiter=",")

        self.best_loss = float('inf')
        self.current_loss = float('inf')
        self.loss_history = []
        self.best_losses = []
        self.acceptance_history = []

        self.model = model

        if calculator is not None:
            self.calculator = calculator
        elif calc == "mace":
            self.calculator = mace_mp(model="large", device=device)
        else:
            ckpt = torch.load("mattersim-v1.0.0-5M.pth", map_location=device)
            m3gnet_model = m3gnet.M3Gnet(**ckpt["model_args"])
            state_dict = ckpt["model"]
            m3gnet_model.load_state_dict(state_dict=state_dict, strict=False)
            self.calculator = MatterSimCalculator(potential=Potential(m3gnet_model),
                                                  device=device)

    def _reset_accept_counters(self):
        self._pos_accept = self._pos_attempt = 0
        self._len_accept = self._len_attempt = 0
        self._ang_accept = self._ang_attempt = 0

    def _adapt_one(self, step, n_accept, n_attempt, label):
        if n_attempt == 0:
            return step
        rate = n_accept / n_attempt
        if rate > 0.5:
            step *= 1.1
        elif rate < 0.3:
            step *= 0.9
        step = float(np.clip(step, self.step_min, self.step_max))
        return step

    def adapt_step_sizes(self, iteration):
        if iteration > 0 and iteration % self.adapt_interval == 0:
            self.pos_step = self._adapt_one(
                self.pos_step, self._pos_accept, self._pos_attempt, "pos")
            self.len_step = self._adapt_one(
                self.len_step, self._len_accept, self._len_attempt, "len")
            self.ang_step = self._adapt_one(
                self.ang_step, self._ang_accept, self._ang_attempt, "ang")
            print(f"  [Adapt] pos={self.pos_step:.4f} Å  "
                  f"len={self.len_step:.4f} Å  ang={self.ang_step:.2f}°")
            self._reset_accept_counters()

    def perturb_atom(self, atoms):
        new_atoms = atoms.copy()

        weights = self._repulsion_weights(atoms)
        atom_idx = np.random.choice(len(atoms), p=weights)

        cell_lengths = np.linalg.norm(new_atoms.cell[:], axis=1)
        frac_step = self.pos_step / cell_lengths
        frac = new_atoms.get_scaled_positions()
        frac[atom_idx] += np.random.randn(3) * frac_step
        frac %= 1.0
        new_atoms.set_scaled_positions(frac)
        return new_atoms, "pos"

    def perturb_cell_length(self, atoms):
        new_atoms = atoms.copy()
        cellpar = cell_to_cellpar(new_atoms.get_cell())
        idx = np.random.randint(0, 3)
        cellpar[idx] += np.random.randn() * self.len_step
        cellpar[idx] = float(np.clip(cellpar[idx], 3.0, 100.0))
        try:
            new_atoms.set_cell(cellpar_to_cell(cellpar), scale_atoms=True)
            return new_atoms, "len"
        except (AssertionError, ValueError):
            return atoms, "len"

    def perturb_cell_angle(self, atoms):
        new_atoms = atoms.copy()
        cellpar = cell_to_cellpar(new_atoms.get_cell())
        idx = np.random.randint(3, 6)
        cellpar[idx] += np.random.randn() * self.ang_step
        cellpar[idx] = float(np.clip(cellpar[idx], 30.0, 150.0))
        a, b, g = cellpar[3], cellpar[4], cellpar[5]
        if a + b <= g or a + g <= b or b + g <= a:
            return atoms, "ang"
        try:
            new_atoms.set_cell(cellpar_to_cell(cellpar), scale_atoms=False)
            return new_atoms, "ang"
        except (AssertionError, ValueError):
            return atoms, "ang"

    def _repulsion_weights(self, atoms):
        n = len(atoms)
        try:
            dists = atoms.get_all_distances(mic=True)  # (n, n)
            np.fill_diagonal(dists, np.inf)
            nums = atoms.get_atomic_numbers()
            scores = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    rmin = (self.lj_rmins[nums[i]] + self.lj_rmins[nums[j]]) / 2.0
                    overlap = max(0.0, rmin - dists[i, j])
                    scores[i] += overlap
            total = scores.sum()
            if total > 0:
                return scores / total
        except Exception:
            pass
        return np.ones(n) / n

    def accept_move(self, new_energy, old_energy):
        if new_energy < old_energy:
            return True
        delta_e = new_energy - old_energy
        return np.random.rand() < np.exp(-delta_e / self.temperature)

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

    def get_energy(self, atoms):
        atoms = self.sanitize_atoms(atoms)
        atoms.calc = self.calculator
        try:
            return atoms.get_potential_energy()
        except Exception:
            return float('inf')

    def local_optimize(self, atoms):
        if self.local_steps == 0:
            return atoms, self.get_energy(atoms)
        atoms = self.sanitize_atoms(atoms)
        atoms.calc = self.calculator
        try:
            opt_state = ts.optimize(
                system=atoms,
                model=self.model,
                optimizer=ts.frechet_cell_fire,
                autobatcher=False,
                max_steps=self.local_steps
            )
            opt_list = opt_state.to_atoms()
            opt_atoms = opt_list[0] if isinstance(opt_list, list) else opt_list
            opt_atoms.calc = self.calculator
            if not validate_structure_distances(opt_atoms):
                separate_close_atoms2(opt_atoms)
            energy = opt_atoms.get_potential_energy()
            return opt_atoms, energy
        except Exception as e:
            return atoms, self.get_energy(atoms)

    def run(self, ground_truth_energy=None, init_atoms=None, _plot=True, _save=True):
        if _save:
            for d in ("plots", "matches", "fails", "lower_energy", "checkpoints"):
                os.makedirs(d, exist_ok=True)

        matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8)

        original_cif = os.path.join("cifs", self.cif_name + ".cif")
        ground_truth = Structure.from_file(original_cif)

        if ground_truth_energy is None:
            atoms_gt = AseAtomsAdaptor.get_atoms(ground_truth)
            atoms_gt.calc = self.calculator
            ground_truth_energy = atoms_gt.get_potential_energy()

        if init_atoms is None:
            init_atoms = initialize_atoms(self.el_symbols, self.lj_rmins, self.zs,
                                          self.zcounts, self.possible_sgs, self.sg_probs)

        current_atoms = init_atoms.copy()
        self.current_loss = self.get_energy(current_atoms)
        best_atoms = current_atoms.copy()
        self.best_loss = self.current_loss

        print(f"[T={self.temperature:.2f}] Initial energy: {self.current_loss:.4f} eV  "
              f"(GT: {ground_truth_energy:.4f} eV)")

        accepted_moves = 0

        for i in range(self.iters):
            start_time = time.time()

            if i > 0 and i % self.restart_interval == 0:
                current_atoms = best_atoms.copy()
                self.current_loss = self.best_loss
                print(f"  [Restart T={self.temperature:.2f}] "
                      f"Returning to best: {self.best_loss:.4f} eV")

            # Choose move type: 60% pos, 20% cell-length, 20% cell-angle
            r = np.random.rand()
            if self.cell_perturb and r > 0.8:
                perturbed_atoms, move_type = self.perturb_cell_angle(current_atoms)
            elif self.cell_perturb and r > 0.6:
                perturbed_atoms, move_type = self.perturb_cell_length(current_atoms)
            else:
                perturbed_atoms, move_type = self.perturb_atom(current_atoms)

            raw_energy = self.get_energy(perturbed_atoms)
            run_local = (
                self.local_steps > 0 and (
                    (raw_energy - self.best_loss) < self.energy_opt_threshold
                    or i % 10 == 0
                )
            )
            if run_local:
                candidate_atoms, new_energy = self.local_optimize(perturbed_atoms)
            else:
                candidate_atoms, new_energy = perturbed_atoms, raw_energy

            # Metropolis acceptance
            if move_type == "pos":
                self._pos_attempt += 1
            elif move_type == "len":
                self._len_attempt += 1
            else:
                self._ang_attempt += 1

            if self.accept_move(new_energy, self.current_loss):
                current_atoms = candidate_atoms
                self.current_loss = new_energy
                accepted_moves += 1
                self.acceptance_history.append(1)

                if move_type == "pos":
                    self._pos_accept += 1
                elif move_type == "len":
                    self._len_accept += 1
                else:
                    self._ang_accept += 1

                if new_energy < self.best_loss:
                    self.best_loss = new_energy
                    best_atoms = candidate_atoms.copy()
                    print(f"  *** [T={self.temperature:.2f}] "
                          f"New best: {self.best_loss:.4f} eV")
            else:
                self.acceptance_history.append(0)

            self.loss_history.append(self.current_loss)
            self.best_losses.append(self.best_loss)

            self.adapt_step_sizes(i + 1)

            if (i + 1) % 10 == 0:
                rate = accepted_moves / (i + 1)
                print(f"[T={self.temperature:.2f}] Iter {i+1}/{self.iters}: "
                      f"GT: {ground_truth_energy:.4f}  "
                      f"Best: {self.best_loss:.4f}  "
                      f"Current: {self.current_loss:.4f}  "
                      f"Acc: {rate:.2%}  "
                      f"Time: {time.time() - start_time:.2f}s")

            if _save and (i + 1) % 100 == 0:
                ase.io.write(
                    f"checkpoints/{self.cif_name}_T{self.temperature:.2f}_iter{i+1}.cif",
                    best_atoms
                )

        if _plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(self.loss_history, label='Current Energy', alpha=0.7)
            ax1.plot(self.best_losses, label='Best Energy', linewidth=2)
            ax1.axhline(y=ground_truth_energy, color='r', linestyle='--',
                        label='Ground Truth')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Energy (eV)')
            ax1.set_title(f'RMC T={self.temperature:.2f}: {self.cif_name}')
            ax1.legend()

            window = min(100, len(self.acceptance_history))
            if window > 0:
                running_acc = np.convolve(
                    self.acceptance_history, np.ones(window) / window, mode='valid')
                ax2.plot(running_acc)
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel(f'Acceptance Rate (window={window})')
                ax2.set_ylim([0, 1])

            plt.tight_layout()
            plt.savefig(f'plots/rmc_{self.cif_name}_T{self.temperature:.2f}.png')
            plt.close()

        if not _save:
            return best_atoms, self.best_loss

        return self._evaluate_and_save(
            best_atoms, ground_truth, ground_truth_energy, matcher)

    def _evaluate_and_save(self, best_atoms, ground_truth, ground_truth_energy, matcher):
        for d in ("plots", "matches", "fails", "lower_energy"):
            os.makedirs(d, exist_ok=True)

        try:
            optimized_structure = AseAtomsAdaptor.get_structure(best_atoms)
            best_atoms.calc = self.calculator
            optimized_energy = best_atoms.get_potential_energy()

            energy_diff = abs(optimized_energy - ground_truth_energy)
            try:
                distance = matcher.get_rms_dist(ground_truth, optimized_structure)[0]
            except Exception:
                distance = np.linalg.norm(
                    optimized_structure.frac_coords - ground_truth.frac_coords).mean()

            if distance < 2.5 and energy_diff <= 0.1:
                result_type = "match"
                print(f"Matched {self.cif_name}: RMSD={distance:.3f}  ΔE={energy_diff:.3f} eV")
            elif optimized_energy < ground_truth_energy:
                result_type = "lower_energy"
                print(f"Lower energy {self.cif_name}: RMSD={distance:.3f}  ΔE={energy_diff:.3f} eV")
            else:
                result_type = "fail"
                print(f"No match {self.cif_name}: RMSD={distance:.3f}  ΔE={energy_diff:.3f} eV")

            out_dir = {"match": "matches", "lower_energy": "lower_energy",
                       "fail": "fails"}[result_type]
            ase.io.write(
                os.path.join(out_dir, f"best_structure_{self.cif_name}.cif"), best_atoms)

            with open(f"plots/{self.cif_name}_rmc_costs.txt", "w") as f:
                f.write(f"Ground Truth: {ground_truth_energy}\n")
                f.write(f"Best Energy: {self.best_loss}\n")

            return result_type in ("match", "lower_energy"), (distance, energy_diff)

        except ValueError:
            print(f"{self.cif_name}: invalid structure, cannot match")
            ase.io.write(
                os.path.join("fails", f"best_structure_{self.cif_name}.cif"), best_atoms)
            return False, (float('inf'), float('inf'))

class ParallelTemperingRMC():
    def __init__(self, cif_name, model, composition, cell, calc, iters, local_steps,
                 step_size=0.3, cell_perturb=True,
                 adapt_interval=50, restart_interval=200,
                 energy_opt_threshold=0.5,
                 temperatures=None, swap_interval=50):

        if temperatures is None:
            temperatures = [0.5, 1.0, 2.0, 5.0]
        assert temperatures == sorted(temperatures), \
            "temperatures must be sorted ascending"

        self.temperatures = temperatures
        self.swap_interval = swap_interval
        self.cif_name = cif_name
        self.iters = iters

        if calc == "mace":
            shared_calc = mace_mp(model="large", device=device)
        else:
            ckpt = torch.load("mattersim-v1.0.0-5M.pth", map_location=device)
            m3gnet_model = m3gnet.M3Gnet(**ckpt["model_args"])
            state_dict = ckpt["model"]
            m3gnet_model.load_state_dict(state_dict=state_dict, strict=False)
            shared_calc = MatterSimCalculator(potential=Potential(m3gnet_model),
                                              device=device)

        self.replicas = [
            RMC(
                cif_name=cif_name,
                model=model,
                composition=composition,
                cell=cell,
                calc=calc,
                iters=iters,
                local_steps=local_steps,
                step_size=step_size,
                temperature=T,
                cell_perturb=cell_perturb,
                adapt_interval=adapt_interval,
                restart_interval=restart_interval,
                energy_opt_threshold=energy_opt_threshold,
                calculator=shared_calc,
            )
            for T in temperatures
        ]

        self.swap_history = []

    def _attempt_swap(self, i, j):
        """
        Metropolis swap between replicas i and j.
        Δ = (1/T_i - 1/T_j) * (E_j - E_i)
        """
        Ti, Tj = self.replicas[i].temperature, self.replicas[j].temperature
        Ei, Ej = self.replicas[i].current_loss, self.replicas[j].current_loss
        delta = (1.0 / Ti - 1.0 / Tj) * (Ej - Ei)
        if delta <= 0 or np.random.rand() < np.exp(-delta):
            self.replicas[i].current_loss, self.replicas[j].current_loss = Ej, Ei
            self.replicas[i].current_atoms, self.replicas[j].current_atoms = \
                self.replicas[j].current_atoms, self.replicas[i].current_atoms
            return True
        return False

    def run(self):
        for d in ("plots", "matches", "fails", "lower_energy", "checkpoints"):
            os.makedirs(d, exist_ok=True)

        matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8)

        original_cif = os.path.join("cifs", self.cif_name + ".cif")
        ground_truth = Structure.from_file(original_cif)

        r0 = self.replicas[0]
        atoms_gt = AseAtomsAdaptor.get_atoms(ground_truth)
        atoms_gt.calc = r0.calculator
        ground_truth_energy = atoms_gt.get_potential_energy()

        for rep in self.replicas:
            init_atoms = initialize_atoms(
                rep.el_symbols, rep.lj_rmins, rep.zs, rep.zcounts,
                rep.possible_sgs, rep.sg_probs)
            rep.current_atoms = init_atoms.copy()
            rep.current_loss = rep.get_energy(rep.current_atoms)
            rep.best_atoms = rep.current_atoms.copy()
            rep.best_loss = rep.current_loss
            print(f"[T={rep.temperature:.2f}] Initial energy: {rep.current_loss:.4f}")

        # Main PT loop — each "iteration" is one MC step per replica
        for i in range(self.iters):

            for rep in self.replicas:
                # Single-step MC (inline — avoids re-running rep.run())
                if i > 0 and i % rep.restart_interval == 0:
                    rep.current_atoms = rep.best_atoms.copy()
                    rep.current_loss = rep.best_loss

                r = np.random.rand()
                if rep.cell_perturb and r > 0.8:
                    perturbed, move_type = rep.perturb_cell_angle(rep.current_atoms)
                elif rep.cell_perturb and r > 0.6:
                    perturbed, move_type = rep.perturb_cell_length(rep.current_atoms)
                else:
                    perturbed, move_type = rep.perturb_atom(rep.current_atoms)

                raw_energy = rep.get_energy(perturbed)
                run_local = (
                    rep.local_steps > 0 and (
                        (raw_energy - rep.best_loss) < rep.energy_opt_threshold
                        or i % 10 == 0
                    )
                )
                if run_local:
                    candidate, new_energy = rep.local_optimize(perturbed)
                else:
                    candidate, new_energy = perturbed, raw_energy

                if move_type == "pos":
                    rep._pos_attempt += 1
                elif move_type == "len":
                    rep._len_attempt += 1
                else:
                    rep._ang_attempt += 1

                if rep.accept_move(new_energy, rep.current_loss):
                    rep.current_atoms = candidate
                    rep.current_loss = new_energy
                    rep.acceptance_history.append(1)
                    if move_type == "pos":
                        rep._pos_accept += 1
                    elif move_type == "len":
                        rep._len_accept += 1
                    else:
                        rep._ang_accept += 1
                    if new_energy < rep.best_loss:
                        rep.best_loss = new_energy
                        rep.best_atoms = candidate.copy()
                        print(f"  *** [T={rep.temperature:.2f}] "
                              f"New best: {rep.best_loss:.4f} eV")
                else:
                    rep.acceptance_history.append(0)

                rep.loss_history.append(rep.current_loss)
                rep.best_losses.append(rep.best_loss)
                rep.adapt_step_sizes(i + 1)

            # Attempt replica swaps between all adjacent pairs
            if (i + 1) % self.swap_interval == 0:
                for k in range(len(self.replicas) - 1):
                    accepted = self._attempt_swap(k, k + 1)
                    self.swap_history.append((i, k, k + 1, accepted))
                    if accepted:
                        print(f"  [Swap] T={self.temperatures[k]:.2f} ↔ "
                              f"T={self.temperatures[k+1]:.2f}  accepted")

            if (i + 1) % 100 == 0:
                print(f"\n--- Iteration {i+1}/{self.iters} summary ---")
                for rep in self.replicas:
                    print(f"  T={rep.temperature:.2f}: "
                          f"best={rep.best_loss:.4f}  current={rep.current_loss:.4f}")
                print()

        best_rep = min(self.replicas, key=lambda r: r.best_loss)
        best_atoms = best_rep.best_atoms

        # Plot all replicas on one figure
        fig, axes = plt.subplots(len(self.replicas), 1,
                                 figsize=(12, 4 * len(self.replicas)), sharex=True)
        if len(self.replicas) == 1:
            axes = [axes]
        for ax, rep in zip(axes, self.replicas):
            ax.plot(rep.loss_history, alpha=0.6, label='Current')
            ax.plot(rep.best_losses, linewidth=2, label='Best')
            ax.axhline(y=ground_truth_energy, color='r', linestyle='--',
                       label='GT')
            ax.set_ylabel('Energy (eV)')
            ax.set_title(f'T={rep.temperature:.2f}')
            ax.legend(fontsize=7)
        axes[-1].set_xlabel('Iteration')
        plt.suptitle(f'Parallel Tempering RMC: {self.cif_name}')
        plt.tight_layout()
        plt.savefig(f'plots/pt_rmc_{self.cif_name}.png')
        plt.close()

        return r0._evaluate_and_save(
            best_atoms, ground_truth, ground_truth_energy, matcher)


if __name__ == "__main__":
    calc = "mattersim"

    if calc == "mace":
        mace_raw = mace_mp(model="large", return_raw_model=True)
        model = MaceModel(model=mace_raw)
    else:
        ckpt = torch.load("mattersim-v1.0.0-5M.pth", map_location=device)
        m3gnet_model = m3gnet.M3Gnet(**ckpt["model_args"])
        state_dict = ckpt["model"]
        m3gnet_model.load_state_dict(state_dict=state_dict, strict=False)
        model = MatterSimModel(model=Potential(m3gnet_model))

    all_matches = []
    all_dist_energy = []

    USE_PARALLEL_TEMPERING = True 

    for filename in os.listdir("cifs"):
        cif = os.path.join("cifs", filename)
        cif_name = os.path.splitext(filename)[0]

        composition = extract_composition(cif)
        cell = extract_cell(cif)

        iters = 10000
        local_steps = 100
        step_size = 0.3
        cell_perturb = True
        adapt_interval = 50
        restart_interval = 200
        energy_opt_threshold = 0.5

        if USE_PARALLEL_TEMPERING:
            runner = ParallelTemperingRMC(
                cif_name=cif_name,
                model=model,
                composition=composition,
                cell=cell,
                calc=calc,
                iters=iters,
                local_steps=local_steps,
                step_size=step_size,
                cell_perturb=cell_perturb,
                adapt_interval=adapt_interval,
                restart_interval=restart_interval,
                energy_opt_threshold=energy_opt_threshold,
                temperatures=[0.3, 0.7, 1.5, 4.0],
                swap_interval=50,
            )
        else:
            runner = RMC(
                cif_name=cif_name,
                model=model,
                composition=composition,
                cell=cell,
                calc=calc,
                iters=iters,
                local_steps=local_steps,
                step_size=step_size,
                temperature=1.0,
                cell_perturb=cell_perturb,
                adapt_interval=adapt_interval,
                restart_interval=restart_interval,
                energy_opt_threshold=energy_opt_threshold,
            )

        match, dist_energy = runner.run()
        all_matches.append(match)
        all_dist_energy.append(dist_energy)
        print(f"{cif_name} match: {match}")

    num_true = sum(all_matches)
    total = len(all_matches)

    print("\nDistance and Energy Differences:")
    print(all_dist_energy)
    print(f"\nMatched {num_true}/{total} structures.")
    print(f"Match ratio: {num_true / total if total > 0 else 0:.2f}")