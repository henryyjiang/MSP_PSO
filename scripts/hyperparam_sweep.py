from ase.neighborlist import neighbor_list
from ase.filters import ExpCellFilter, UnitCellFilter, FrechetCellFilter
from ase.geometry import cell_to_cellpar, cellpar_to_cell
import ase
import numpy as np
import pyswarms as ps
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
import pandas as pd
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("../.."))))
from msp.utils.objectives import Energy
from msp.forcefield import MDL_FF
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from pathlib import Path

from utils import *

logging.getLogger("mattertune").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("pandas").setLevel(logging.CRITICAL)

device = "cuda" if torch.cuda.is_available() else "cpu"


class PSO():
    def __init__(self, cif_name, model, composition, cell, calc, options, particles, iters, local_steps,
                 cell_perturb=True):
        self.cif_name = cif_name
        self.cell_perturb = cell_perturb
        self.composition = composition
        self.cell = [cell] * particles
        self.best_cell = []

        self.options = options
        self.particles = particles
        self.iters = iters
        self.local_steps = local_steps

        self.zs, self.zcounts = composition_to_zs(self.composition)
        self.possible_sgs, self.sg_probs = generate_sgs(self.zs, self.zcounts)
        self.el_symbols = np.array([periodictable.elements[i].symbol for i in range(95)])
        self.lj_rmins = np.genfromtxt(str(Path(__file__).parent / "lj_rmins.csv"),
                                      delimiter=",")

        self.best_losses = []
        self.best_loss = float('inf')
        self.avg_losses = []

        my_dataset = json.load(open("../data/data_subset_msp.json", "r"))
        train_config = 'mdl_config.yml'
        self.forcefield = MDL_FF(train_config, my_dataset)
        self.energy = Energy(normalize=True, ljr_ratio=1)

        self.optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=54,
                                                 options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
        self.model = model

        if calc == "mace":
            self.calculator = mace_mp(model="large", device=device)
        else:
            ckpt = torch.load("mattersim-v1.0.0-5M.pth", map_location=device)
            m3gnet_model = m3gnet.M3Gnet(**ckpt["model_args"])
            state_dict = ckpt["model"]
            m3gnet_model.load_state_dict(state_dict=state_dict, strict=False)

            self.calculator = MatterSimCalculator(potential=Potential(m3gnet_model), device=device)

    def obj_func(self, params, i):
        atoms = dimensions_to_atoms(params, i, self.composition, self.cell, self.calculator, self.cell_perturb)

        atoms.calc = self.calculator
        try:
            loss = atoms.get_potential_energy()
        except:
            loss = float('inf')

        if loss < self.best_loss:
            self.best_loss = loss

        return loss

    def f(self, x):
        n_particles = x.shape[0]
        j = [self.obj_func(x[i], i) for i in range(n_particles)]

        self.best_losses.append(self.best_loss)
        self.avg_losses.append(np.mean(j))

        return np.array(j)

    def run(self, run_id=""):
        costs = []
        matches = []
        dist_energy = []

        os.makedirs(f"plots/{run_id}", exist_ok=True)
        os.makedirs(f"matches/{run_id}", exist_ok=True)
        os.makedirs(f"fails/{run_id}", exist_ok=True)
        os.makedirs(f"lower_energy/{run_id}", exist_ok=True)

        matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8)

        original_cif = os.path.join("cifs", self.cif_name + ".cif")
        ground_truth = Structure.from_file(original_cif)

        atoms_gt = AseAtomsAdaptor.get_atoms(ground_truth)
        atoms_gt.calc = self.calculator
        ground_truth_energy = atoms_gt.get_potential_energy()

        for iteration in range(1):
            self.best_losses = []
            self.best_loss = float('inf')
            self.avg_losses = []

            options = self.options
            particles = self.particles
            iters = self.iters
            if not self.cell_perturb:
                dimensions = len(self.composition) * 3
            else:
                dimensions = 9 + len(self.composition) * 3

            init_positions = np.empty((particles, dimensions))
            for i in range(particles):
                init_atoms = initialize_atoms(self.el_symbols, self.lj_rmins, self.zs, self.zcounts, self.possible_sgs,
                                              self.sg_probs)
                if self.cell_perturb:
                    flattened_cell = [i for l in init_atoms.get_cell().tolist() for i in l]
                    flattened_pos = [float(i) for l in init_atoms.positions for i in l]
                    init_pos = flattened_cell + flattened_pos
                else:
                    self.cell[i] = init_atoms.get_cell()
                    init_pos = [float(i) for l in init_atoms.positions for i in l]
                init_positions[i] = np.array(init_pos)

            self.optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options,
                                                     init_pos=init_positions)

            for i in range(iters):
                start_time = time.time()

                cost = self.f(self.optimizer.swarm.position)
                self.optimizer.swarm.current_cost = cost

                if self.optimizer.swarm.pbest_cost.size == 0:
                    self.optimizer.swarm.pbest_cost = np.full(self.optimizer.swarm.position.shape[0], np.inf)

                improved = cost < self.optimizer.swarm.pbest_cost
                self.optimizer.swarm.pbest_pos[improved] = self.optimizer.swarm.position[improved]
                self.optimizer.swarm.pbest_cost[improved] = cost[improved]

                min_idx = np.argmin(self.optimizer.swarm.pbest_cost)
                if self.optimizer.swarm.pbest_cost[min_idx] < self.optimizer.swarm.best_cost:
                    self.optimizer.swarm.best_cost = self.optimizer.swarm.pbest_cost[min_idx]
                    self.optimizer.swarm.best_pos = self.optimizer.swarm.pbest_pos[min_idx]
                    self.best_cell = self.cell[min_idx]

                n_particles, dimensions = self.optimizer.swarm.position.shape
                r1, r2 = np.random.rand(n_particles, dimensions), np.random.rand(n_particles, dimensions)
                cognitive = options["c1"] * r1 * (self.optimizer.swarm.pbest_pos - self.optimizer.swarm.position)
                social = options["c2"] * r2 * (self.optimizer.swarm.best_pos - self.optimizer.swarm.position)
                self.optimizer.swarm.velocity = options["w"] * self.optimizer.swarm.velocity + cognitive + social

                self.optimizer.swarm.position += self.optimizer.swarm.velocity
                if not self.cell_perturb:
                    lower_bound = np.full(self.optimizer.swarm.position.shape[1], -50)
                    upper_bound = np.full(self.optimizer.swarm.position.shape[1], 50)
                else:
                    cell_dims = 9
                    lower_bound = np.concatenate([
                        np.full(cell_dims, 2.0),
                        np.full(dimensions - cell_dims, -50)
                    ])
                    upper_bound = np.concatenate([
                        np.full(cell_dims, 100.0),
                        np.full(dimensions - cell_dims, 50)
                    ])

                self.optimizer.swarm.position = np.clip(self.optimizer.swarm.position, lower_bound, upper_bound)

                positions = self.optimizer.swarm.position
                new_atoms = [dimensions_to_atoms(positions[i], i, self.composition, self.cell, self.calculator,
                                                 self.cell_perturb) for i in range(len(positions))]

                sanitized_atoms = []
                for atoms in new_atoms:
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

                        sanitized_atoms.append(atoms)
                    except Exception as e:
                        sanitized_atoms.append(atoms)
                try:
                    optimized_state = ts.optimize(
                        system=sanitized_atoms,
                        model=self.model,
                        optimizer=ts.frechet_cell_fire,
                        autobatcher=False,
                        max_steps=self.local_steps)
                    optimized_atoms = optimized_state.to_atoms()

                    final_atoms = []
                    for i, atoms in enumerate(optimized_atoms):
                        atoms.calc = self.calculator
                        if validate_structure_distances(atoms):
                            final_atoms.append(atoms)
                        else:
                            separate_close_atoms2(atoms)
                            final_atoms.append(atoms)
                except Exception as e:
                    final_atoms = sanitized_atoms

                if not self.cell_perturb:
                    self.cell = [opt.cell if hasattr(opt, "cell") else None for opt in final_atoms]

                final_costs = []
                final_positions = []
                for i, atoms in enumerate(final_atoms):
                    try:
                        final_costs.append(atoms.get_potential_energy())
                        final_positions.append(atoms_to_dimensions(final_atoms[i], self.cell_perturb))
                    except:
                        final_costs.append(self.optimizer.swarm.current_cost[i])
                        final_positions.append(self.optimizer.swarm.position[i])

                self.optimizer.swarm.current_cost = np.array(final_costs)
                self.optimizer.swarm.position = np.array(final_positions)

                print(
                    f"Iteration {i + 1}: Ground Truth: {ground_truth_energy}, Best Cost = {self.optimizer.swarm.best_cost}, Current Cost = {self.optimizer.swarm.current_cost[0]}, Time Taken: {(time.time() - start_time):.2f} s")

            cost = self.optimizer.swarm.best_cost
            costs.append(cost)
            pos = self.optimizer.swarm.best_pos

            plt.plot(self.best_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Best Loss')
            plt.title('Best Losses')
            plt.savefig(f'plots/{run_id}/best_losses_{self.cif_name}_{iteration}.png')
            plt.close()

            plt.plot(self.avg_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title('Average Losses')
            plt.savefig(f'plots/{run_id}/avg_losses_{self.cif_name}_{iteration}.png')
            plt.close()

            final_atoms = final_dimensions(pos, self.best_cell, self.composition, self.cell_perturb)
            try:
                optimized_structure = AseAtomsAdaptor.get_structure(final_atoms)

                atoms_opt = AseAtomsAdaptor.get_atoms(optimized_structure)
                atoms_opt.calc = self.calculator
                optimized_energy = atoms_opt.get_potential_energy()

                energy_tolerance = 0.1
                distance_threshold = 2.5

                energy_diff = abs(optimized_energy - ground_truth_energy)
                try:
                    distance = matcher.get_rms_dist(ground_truth, optimized_structure)[0]
                except Exception:
                    distance = np.linalg.norm(optimized_structure.frac_coords - ground_truth.frac_coords).mean()

                if distance < distance_threshold and energy_diff <= energy_tolerance:
                    result_type = "match"
                    print(f"Matched for {self.cif_name}: RMSD = {distance:.3f}, ΔE = {energy_diff:.3f} eV")
                elif optimized_energy - ground_truth_energy < 0:
                    result_type = "lower_energy"
                    print(f"Lower energy for {self.cif_name}: RMSD = {distance:.3f}, ΔE = {energy_diff:.3f} eV")
                else:
                    result_type = "fail"
                    print(f"No match for {self.cif_name}: RMSD = {distance:.3f}, ΔE = {energy_diff:.3f} eV")

                if result_type == "match":
                    out_dir = f"matches/{run_id}"
                elif result_type == "lower_energy":
                    out_dir = f"lower_energy/{run_id}"
                else:
                    out_dir = f"fails/{run_id}"

                filename = os.path.join(out_dir, f"best_structure_{self.cif_name}_{iteration}.cif")
                ase.io.write(filename, final_atoms)

                matches.append(result_type == "match" or result_type == "lower_energy")
                dist_energy.append((distance, energy_diff))

            except ValueError:
                print(f"{self.cif_name}: invalid structure, cannot match")
                matches.append(False)
                dist_energy.append((np.nan, np.nan))
                filename = os.path.join(f"fails/{run_id}", f"best_structure_{self.cif_name}_{iteration}.cif")
                ase.io.write(filename, final_atoms)

        costs_filename = f"plots/{run_id}/{self.cif_name}_costs.txt"
        with open(costs_filename, "w") as f:
            f.write(f"Ground Truth: {ground_truth_energy}\n")
            for cost in costs:
                f.write(f"{cost}\n")

        return matches, dist_energy


def run_hyperparameter_sweep():
    """Run hyperparameter sweep and collect results"""

    calc = "mattersim"

    # Initialize model once
    if calc == "mace":
        mace = mace_mp(model="large", return_raw_model=True)
        model = MaceModel(model=mace)
    else:
        ckpt = torch.load("mattersim-v1.0.0-5M.pth", map_location=device)
        m3gnet_model = m3gnet.M3Gnet(**ckpt["model_args"])
        state_dict = ckpt["model"]
        m3gnet_model.load_state_dict(state_dict=state_dict, strict=False)
        model = MatterSimModel(model=Potential(m3gnet_model))

    # Define hyperparameter grid
    w_values = [0.5, 0.7]
    c1_values = [0.7, 1.0, 1.2, 1.5]
    c2_values = [0.9, 1.2, 1.5, 1.8]

    # Fixed parameters
    particles = 10
    iters = 50
    local_steps = 25
    cell_perturb = False

    # Get list of CIF files
    cif_files = [f for f in os.listdir("cifs") if f.endswith(".cif")]

    # Store results for each hyperparameter combination
    results = []

    # Iterate through all combinations
    for w, c1, c2 in product(w_values, c1_values, c2_values):
        options = {'c1': c1, 'c2': c2, 'w': w}
        run_id = f"w{w}_c1{c1}_c2{c2}"

        print(f"\n{'=' * 80}")
        print(f"Running hyperparameter combination: w={w}, c1={c1}, c2={c2}")
        print(f"{'=' * 80}\n")

        all_matches = []
        all_distances = []
        all_energy_diffs = []

        for filename in cif_files:
            cif = os.path.join("cifs", filename)
            cif_name = os.path.splitext(filename)[0]

            composition = extract_composition(cif)
            cell = extract_cell(cif)

            if cell_perturb:
                pso = PSO(cif_name, model, composition, None, calc, options, particles, iters, local_steps,
                          cell_perturb)
            else:
                pso = PSO(cif_name, model, composition, cell, calc, options, particles, iters, local_steps,
                          cell_perturb)

            matches, dist_energy = pso.run(run_id=run_id)
            all_matches.extend(matches)

            for distance, energy_diff in dist_energy:
                if not np.isnan(distance):
                    all_distances.append(distance)
                    all_energy_diffs.append(energy_diff)

            print(f"{cif_name} match: {matches}")

        # Calculate metrics for this hyperparameter combination
        num_matches = sum(all_matches)
        total = len(all_matches)
        match_rate = num_matches / total if total > 0 else 0

        avg_distance = np.mean(all_distances) if all_distances else np.nan
        std_distance = np.std(all_distances) if all_distances else np.nan
        median_distance = np.median(all_distances) if all_distances else np.nan

        avg_energy_diff = np.mean(all_energy_diffs) if all_energy_diffs else np.nan
        std_energy_diff = np.std(all_energy_diffs) if all_energy_diffs else np.nan
        median_energy_diff = np.median(all_energy_diffs) if all_energy_diffs else np.nan

        results.append({
            'w': w,
            'c1': c1,
            'c2': c2,
            'match_rate': match_rate,
            'num_matches': num_matches,
            'total_structures': total,
            'avg_rmsd': avg_distance,
            'std_rmsd': std_distance,
            'median_rmsd': median_distance,
            'avg_energy_diff': avg_energy_diff,
            'std_energy_diff': std_energy_diff,
            'median_energy_diff': median_energy_diff
        })

        print(f"\nResults for w={w}, c1={c1}, c2={c2}:")
        print(f"  Match rate: {match_rate:.2%} ({num_matches}/{total})")
        print(f"  Avg RMSD: {avg_distance:.3f} ± {std_distance:.3f}")
        print(f"  Median RMSD: {median_distance:.3f}")
        print(f"  Avg Energy Diff: {avg_energy_diff:.3f} ± {std_energy_diff:.3f} eV")
        print(f"  Median Energy Diff: {median_energy_diff:.3f} eV")

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Sort by match rate (descending), then by avg RMSD (ascending)
    df_results_sorted = df_results.sort_values(
        by=['match_rate', 'avg_rmsd'],
        ascending=[False, True]
    )

    # Save results to CSV
    os.makedirs("hyperparameter_results", exist_ok=True)
    df_results_sorted.to_csv("hyperparameter_results/sweep_results.csv", index=False)

    # Print summary
    print(f"\n{'=' * 80}")
    print("HYPERPARAMETER SWEEP SUMMARY")
    print(f"{'=' * 80}\n")
    print(df_results_sorted.to_string(index=False))

    # Print top 5 configurations
    print(f"\n{'=' * 80}")
    print("TOP 5 CONFIGURATIONS BY MATCH RATE")
    print(f"{'=' * 80}\n")
    top5 = df_results_sorted.head(5)
    for idx, row in top5.iterrows():
        print(f"Rank {idx + 1}: w={row['w']}, c1={row['c1']}, c2={row['c2']}")
        print(f"  Match Rate: {row['match_rate']:.2%}")
        print(f"  Avg RMSD: {row['avg_rmsd']:.3f}, Avg Energy Diff: {row['avg_energy_diff']:.3f} eV")
        print()

    # Create visualization
    create_results_visualization(df_results)

    return df_results_sorted


def create_results_visualization(df_results):
    """Create visualizations comparing hyperparameter performance"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Match rate heatmap for different w values
    for idx, w in enumerate([0.5, 0.7]):
        ax = axes[idx, 0]
        df_w = df_results[df_results['w'] == w]
        pivot = df_w.pivot(index='c1', columns='c2', values='match_rate')
        im = ax.imshow(pivot, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('c2')
        ax.set_ylabel('c1')
        ax.set_title(f'Match Rate (w={w})')

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax)

    # 2. Average RMSD comparison
    ax = axes[0, 1]
    df_results_sorted = df_results.sort_values('avg_rmsd')
    x = range(len(df_results_sorted))
    labels = [f"w={row['w']}\nc1={row['c1']}\nc2={row['c2']}"
              for _, row in df_results_sorted.iterrows()]
    ax.bar(x, df_results_sorted['avg_rmsd'])
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Average RMSD')
    ax.set_title('Average RMSD by Configuration')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(labels[::2], rotation=45, ha='right', fontsize=6)

    # 3. Average energy difference comparison
    ax = axes[1, 1]
    df_results_sorted = df_results.sort_values('avg_energy_diff')
    x = range(len(df_results_sorted))
    labels = [f"w={row['w']}\nc1={row['c1']}\nc2={row['c2']}"
              for _, row in df_results_sorted.iterrows()]
    ax.bar(x, df_results_sorted['avg_energy_diff'])
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Average Energy Difference (eV)')
    ax.set_title('Average Energy Difference by Configuration')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(labels[::2], rotation=45, ha='right', fontsize=6)

    plt.tight_layout()
    plt.savefig('hyperparameter_results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nVisualization saved to: hyperparameter_results/performance_comparison.png")


if __name__ == "__main__":
    results = run_hyperparameter_sweep()