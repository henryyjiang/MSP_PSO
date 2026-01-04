from ase.neighborlist import neighbor_list
from ase.filters import ExpCellFilter, UnitCellFilter, FrechetCellFilter
from ase.geometry import cell_to_cellpar, cellpar_to_cell
import ase
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
# from mattertune.backbones import MatterSimM3GNetBackboneModule, MatterSimBackboneConfig
# from mattertune import configs as MC
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("../.."))))
#from matdeeplearn.common.ase_utils import MDLCalculator
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
    def __init__(self, cif_name, model, composition, cell, calc, options, particles, iters, local_steps, cell_perturb=True):
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
                                      delimiter=",") * 0.85

        self.best_losses = []
        self.best_loss = float('inf')
        self.avg_losses = []

        my_dataset = json.load(open("../data/data_subset_msp.json", "r"))
        train_config = 'mdl_config.yml'
        self.forcefield = MDL_FF(train_config, my_dataset)
        self.energy = Energy(normalize=True, ljr_ratio=1)

        self.optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=54, options={'c1': 0.5, 'c2': 0.3, 'w':0.9})
        self.model = model

        if calc == "mace":
            self.calculator = mace_mp(model="large", device=device)
        else:
            ckpt = torch.load("mattersim-v1.0.0-5M.pth", map_location=device)
            m3gnet_model = m3gnet.M3Gnet(**ckpt["model_args"])
            state_dict = ckpt["model"]
            m3gnet_model.load_state_dict(state_dict=state_dict, strict=False)

            self.calculator = MatterSimCalculator(potential=Potential(m3gnet_model), device=device)

        #self.calculator = MDLCalculator(config=train_config)


    def obj_func(self, params, i):
        atoms = dimensions_to_atoms(params, i, self.composition, self.cell, self.calculator, self.cell_perturb)

        atoms.calc = self.calculator
        loss = atoms.get_potential_energy()

        if loss < self.best_loss:
            self.best_loss = loss

        return loss

    def f(self, x):
        n_particles = x.shape[0]
        j = [self.obj_func(x[i], i) for i in range(n_particles)]

        self.best_losses.append(self.best_loss)
        self.avg_losses.append(np.mean(j))

        return np.array(j)

    def run(self):
        costs = []
        matches = []
        dist_energy = []

        os.makedirs("plots", exist_ok=True)
        os.makedirs("matches", exist_ok=True)
        os.makedirs("fails", exist_ok=True)
        os.makedirs("lower_energy", exist_ok=True)

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

            options = self.options  # cognitive, social, inertia
            particles = self.particles  # number of particles in system
            iters = self.iters
            if not self.cell_perturb:
                dimensions = len(self.composition)*3
            else:
                dimensions = 9 + len(self.composition)*3  # first 9 are cell, rest are atom positions

            init_positions = np.empty((particles, dimensions))
            for i in range(particles):
                init_atoms = initialize_atoms(self.el_symbols, self.lj_rmins, self.zs, self.zcounts, self.possible_sgs, self.sg_probs)
                if self.cell_perturb:
                    flattened_cell = [i for l in init_atoms.get_cell().tolist() for i in l]
                    flattened_pos = [float(i) for l in init_atoms.positions for i in l]
                    init_pos = flattened_cell + flattened_pos
                else:
                    self.cell[i] = init_atoms.get_cell()
                    init_pos = [float(i) for l in init_atoms.positions for i in l]
                init_positions[i] = np.array(init_pos)

            self.optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options, init_pos=init_positions)
            #self.optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options, init_pos=init_positions, oh_strategy={'w':'exp_decay'})

            #cost, pos = self.optimizer.optimize(self.f, iters=10)

            for i in range(iters):
                start_time = time.time()

                cost = self.f(self.optimizer.swarm.position)
                self.optimizer.swarm.current_cost = cost

                if self.optimizer.swarm.pbest_cost.size == 0:
                    self.optimizer.swarm.pbest_cost = np.full(self.optimizer.swarm.position.shape[0], np.inf)

                #update pbest
                improved = cost < self.optimizer.swarm.pbest_cost
                self.optimizer.swarm.pbest_pos[improved] = self.optimizer.swarm.position[improved]
                self.optimizer.swarm.pbest_cost[improved] = cost[improved]

                # update gbest
                min_idx = np.argmin(self.optimizer.swarm.pbest_cost)
                if self.optimizer.swarm.pbest_cost[min_idx] < self.optimizer.swarm.best_cost:
                    self.optimizer.swarm.best_cost = self.optimizer.swarm.pbest_cost[min_idx]
                    self.optimizer.swarm.best_pos = self.optimizer.swarm.pbest_pos[min_idx]
                    self.best_cell = self.cell[min_idx]

                #compute velocity
                n_particles, dimensions = self.optimizer.swarm.position.shape
                r1, r2 = np.random.rand(n_particles, dimensions), np.random.rand(n_particles,dimensions)
                cognitive_component = options["c1"] * r1 * (self.optimizer.swarm.pbest_pos - self.optimizer.swarm.position)
                social_component = options["c2"] * r2 * (self.optimizer.swarm.best_pos - self.optimizer.swarm.position)
                self.optimizer.swarm.velocity = options["w"] * self.optimizer.swarm.velocity + cognitive_component + social_component

                # Update positions
                self.optimizer.swarm.position += self.optimizer.swarm.velocity
                lower_bound = np.full(self.optimizer.swarm.position.shape[1], -5)
                upper_bound = np.full(self.optimizer.swarm.position.shape[1], 5)

                self.optimizer.swarm.position = np.clip(self.optimizer.swarm.position, lower_bound, upper_bound)

                #local optimization
                positions = self.optimizer.swarm.position
                new_atoms = [dimensions_to_atoms(positions[i], i, self.composition, self.cell, self.calculator, self.cell_perturb) for i in range(len(positions))]

                sanitized_atoms = []
                for atoms in new_atoms:
                    atoms.calc = self.calculator

                    cellpar = cell_to_cellpar(atoms.cell)
                    cellpar[3:] = np.clip(cellpar[3:], 30.0, 150.0)

                    atoms.set_cell(cellpar_to_cell(cellpar), scale_atoms=True)

                    cell = atoms.get_cell().array
                    lengths = np.linalg.norm(cell, axis=1)
                    if np.any(lengths < 3.0) or np.any(lengths > 150.0):
                        #print("cell lengths out of bounds")
                        lengths = np.clip(lengths, 3.0, 150.0)
                        cell = atoms.get_cell()
                        for i in range(3):
                            cell[i] = cell[i] / np.linalg.norm(cell[i]) * lengths[i]
                        atoms.set_cell(cell, scale_atoms=True)

                    separate_close_atoms(atoms)

                    if not np.all(np.isfinite(atoms.get_forces())):
                        #print("forces are infinite")
                        atoms.positions += 1e-3 * np.random.randn(*atoms.positions.shape)

                    sanitized_atoms.append(atoms)

                optimized_state = ts.optimize(
                        system=sanitized_atoms,
                        model=self.model,
                        optimizer=ts.frechet_cell_fire,
                        autobatcher=False,
                        max_steps =self.local_steps)
                optimized_atoms = optimized_state.to_atoms()
                for atom in optimized_atoms:
                    atom.calc = self.calculator

                if not self.cell_perturb:
                    self.cell = [opt.cell if hasattr(opt, "cell") else None for opt in optimized_atoms]

                self.optimizer.swarm.current_cost = np.array([atoms.get_potential_energy() for atoms in optimized_atoms])
                self.optimizer.swarm.position = np.array([atoms_to_dimensions(optimized_atoms[i], self.cell_perturb) for i in range(len(optimized_atoms))])

                print(f"Iteration {i + 1}: Ground Truth: {ground_truth_energy}, Best Cost = {self.optimizer.swarm.best_cost}, Time Taken: {(time.time() - start_time):.2f} s")

            cost = self.optimizer.swarm.best_cost
            costs.append(cost)
            pos = self.optimizer.swarm.best_pos

            plt.plot(self.best_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Best Loss')
            plt.title('Best Losses')
            plt.savefig(f'plots/best_losses_{self.cif_name}_{iteration}.png')
            plt.close()

            plt.plot(self.avg_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title('Average Losses')
            plt.savefig(f'plots/avg_losses_{self.cif_name}_{iteration}.png')
            plt.close()

            final_atoms = final_dimensions(pos, self.best_cell, self.composition)
            try:
                optimized_structure = AseAtomsAdaptor.get_structure(final_atoms)

                atoms_opt = AseAtomsAdaptor.get_atoms(optimized_structure)
                atoms_opt.calc = self.calculator
                optimized_energy = atoms_opt.get_potential_energy()

                energy_tolerance = 0.05
                distance_threshold = 0.25

                energy_diff = abs(optimized_energy - ground_truth_energy)
                try:
                    # RMSD / distance-like metric from pymatgen
                    distance = matcher.get_rms_dist(ground_truth, optimized_structure)[0]
                except Exception:
                    distance = np.linalg.norm(optimized_structure.frac_coords - ground_truth.frac_coords).mean()

                if distance < distance_threshold and energy_diff <= energy_tolerance:
                    result_type = "match"
                    print(f"Matched for {self.cif_name}: RMSD = {distance:.3f}, Î”E = {energy_diff:.3f} eV")
                elif optimized_energy - ground_truth_energy < 0:
                    result_type = "lower_energy"
                    print(f"Lower energy for {self.cif_name}: RMSD = {distance:.3f}, Î”E = {energy_diff:.3f} eV")
                else:
                    result_type = "fail"
                    print(f"No match for {self.cif_name}: RMSD = {distance:.3f}, Î”E = {energy_diff:.3f} eV")

                # Save CIF accordingly
                if result_type == "match":
                    out_dir = "matches"
                elif result_type == "lower_energy":
                    out_dir = "lower_energy"
                else:
                    out_dir = "fails"

                filename = os.path.join(out_dir, f"best_structure_{self.cif_name}_{iteration}.cif")
                ase.io.write(filename, final_atoms)

                # Record metrics
                matches.append(result_type == "match" or result_type == "lower_energy")
                dist_energy.append((distance, energy_diff))

            except ValueError:
                print(f"{self.cif_name}: invalid structure, cannot match")
                matches.append(False)
                filename = os.path.join("fails", f"best_structure_{self.cif_name}_{iteration}.cif")
                ase.io.write(filename, final_atoms)

        costs_filename = f"plots/{self.cif_name}_costs.txt"
        with open(costs_filename, "w") as f:
            f.write(f"Ground Truth: {ground_truth_energy}\n")
            for cost in costs:
                f.write(f"{cost}\n")

        return matches, dist_energy



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

        options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}  # cognitive, social, inertia
        particles = 30  # number of particles in system
        iters = 100
        local_steps = 100

        cell_perturb = False
        if cell_perturb:
                pso = PSO(cif_name, model, composition, None, calc, options, particles, iters, local_steps, cell_perturb)
        else:
                pso = PSO(cif_name, model, composition, cell, calc, options, particles, iters, local_steps, cell_perturb)
        matches, dist_energy = pso.run()
        all_matches.extend(matches)
        all_dist_energy.extend(dist_energy)
        print(f"{cif_name} match: {matches}")

    num_true = sum(all_matches)
    total = len(all_matches)
    match_ratio = num_true / total if total > 0 else 0

    print("Distance and Energy Differences:")
    print(all_dist_energy)

    print(f"\nMatched {num_true}/{total} structures.")
    print(f"Match ratio: {match_ratio:.2f}")