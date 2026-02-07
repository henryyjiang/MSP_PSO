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


class RMC():
    def __init__(self, cif_name, model, composition, cell, calc, iters, local_steps,
                 step_size=0.3, temperature=1.0, cell_perturb=True):
        self.cif_name = cif_name
        self.cell_perturb = cell_perturb
        self.composition = composition
        self.cell = cell
        
        self.iters = iters
        self.local_steps = local_steps
        self.step_size = step_size
        self.temperature = temperature
        
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
        
        if calc == "mace":
            self.calculator = mace_mp(model="large", device=device)
        else:
            ckpt = torch.load("mattersim-v1.0.0-5M.pth", map_location=device)
            m3gnet_model = m3gnet.M3Gnet(**ckpt["model_args"])
            state_dict = ckpt["model"]
            m3gnet_model.load_state_dict(state_dict=state_dict, strict=False)
            self.calculator = MatterSimCalculator(potential=Potential(m3gnet_model), device=device)
    
    def perturb_atom(self, atoms, atom_idx):
        """Perturb a single atom's position"""
        new_atoms = atoms.copy()
        perturbation = np.random.randn(3) * self.step_size
        new_atoms.positions[atom_idx] += perturbation
        return new_atoms
    
    def perturb_cell_parameter(self, atoms):
        """Perturb a random cell parameter"""
        new_atoms = atoms.copy()
        cell = new_atoms.get_cell()
        
        # Choose random cell vector or angle to perturb
        cellpar = cell_to_cellpar(cell)
        
        # Perturb one parameter
        param_idx = np.random.randint(0, 6)
        if param_idx < 3:  # lengths
            cellpar[param_idx] += np.random.randn() * self.step_size
            cellpar[param_idx] = np.clip(cellpar[param_idx], 3.0, 100.0)
        else:  # angles
            cellpar[param_idx] += np.random.randn() * self.step_size * 5  # larger steps for angles
            cellpar[param_idx] = np.clip(cellpar[param_idx], 30.0, 150.0)
        
        new_cell = cellpar_to_cell(cellpar)
        new_atoms.set_cell(new_cell, scale_atoms=False)
        
        return new_atoms
    
    def accept_move(self, new_energy, old_energy):
        """Metropolis acceptance criterion"""
        if new_energy < old_energy:
            return True
        
        delta_e = new_energy - old_energy
        probability = np.exp(-delta_e / self.temperature)
        return np.random.rand() < probability
    
    def get_energy(self, atoms):
        """Calculate potential energy"""
        atoms.calc = self.calculator
        try:
            return atoms.get_potential_energy()
        except:
            return float('inf')
    
    def local_optimize(self, atoms):
        """Optional local optimization using torch_sim"""
        if self.local_steps == 0:
            return atoms, self.get_energy(atoms)
        
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
            # ts.optimize returns a list even for single atoms object
            optimized_atoms = optimized_atoms_list[0] if isinstance(optimized_atoms_list, list) else optimized_atoms_list
            optimized_atoms.calc = self.calculator
            energy = optimized_atoms.get_potential_energy()
            return optimized_atoms, energy
        except Exception as e:
            return atoms, self.get_energy(atoms)
    
    def run(self):
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
        
        # Initialize structure
        init_atoms = initialize_atoms(self.el_symbols, self.lj_rmins, self.zs,
                                      self.zcounts, self.possible_sgs, self.sg_probs)
        
        current_atoms = init_atoms.copy()
        self.current_loss = self.get_energy(current_atoms)
        best_atoms = current_atoms.copy()
        self.best_loss = self.current_loss
        
        print(f"Initial energy: {self.current_loss}")
        
        accepted_moves = 0
        
        # RMC iterations
        for i in range(self.iters):
            start_time = time.time()
            
            # Randomly choose to perturb atom or cell
            if self.cell_perturb and np.random.rand() < 0.3:  # 30% chance to perturb cell
                perturbed_atoms = self.perturb_cell_parameter(current_atoms)
            else:
                # Perturb random atom
                atom_idx = np.random.randint(0, len(current_atoms))
                perturbed_atoms = self.perturb_atom(current_atoms, atom_idx)
            
            # Calculate new energy (with optional local optimization)
            if self.local_steps > 0 and i % 10 == 0:  # Periodic local optimization
                optimized_atoms, new_energy = self.local_optimize(perturbed_atoms)
            else:
                optimized_atoms = perturbed_atoms
                new_energy = self.get_energy(optimized_atoms)
            
            # Accept or reject move
            if self.accept_move(new_energy, self.current_loss):
                current_atoms = optimized_atoms
                self.current_loss = new_energy
                accepted_moves += 1
                self.acceptance_history.append(1)
                
                # Update best if improved
                if new_energy < self.best_loss:
                    self.best_loss = new_energy
                    best_atoms = optimized_atoms.copy()
                    print(f"  *** New best: {self.best_loss:.4f} eV")
            else:
                self.acceptance_history.append(0)
            
            self.loss_history.append(self.current_loss)
            self.best_losses.append(self.best_loss)
            
            if (i + 1) % 10 == 0:
                acceptance_rate = accepted_moves / (i + 1)
                print(f"Iteration {i + 1}/{self.iters}: Ground Truth: {ground_truth_energy:.4f}, "
                      f"Best: {self.best_loss:.4f}, Current: {self.current_loss:.4f}, "
                      f"Acceptance: {acceptance_rate:.2%}, Time: {(time.time() - start_time):.2f} s")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(self.loss_history, label='Current Energy', alpha=0.7)
        ax1.plot(self.best_losses, label='Best Energy', linewidth=2)
        ax1.axhline(y=ground_truth_energy, color='r', linestyle='--', label='Ground Truth')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title(f'RMC: {self.cif_name}')
        ax1.legend()
        
        # Running acceptance rate
        window = 100
        running_acceptance = np.convolve(self.acceptance_history, 
                                         np.ones(window)/window, mode='valid')
        ax2.plot(running_acceptance)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel(f'Acceptance Rate (window={window})')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'plots/rmc_{self.cif_name}.png')
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
            
            # Save CIF
            if result_type == "match":
                out_dir = "matches"
            elif result_type == "lower_energy":
                out_dir = "lower_energy"
            else:
                out_dir = "fails"
            
            filename = os.path.join(out_dir, f"best_structure_{self.cif_name}.cif")
            ase.io.write(filename, best_atoms)
            
            match = result_type == "match" or result_type == "lower_energy"
            
        except ValueError:
            print(f"{self.cif_name}: invalid structure, cannot match")
            match = False
            distance, energy_diff = float('inf'), float('inf')
            filename = os.path.join("fails", f"best_structure_{self.cif_name}.cif")
            ase.io.write(filename, best_atoms)
        
        # Save energy history
        costs_filename = f"plots/{self.cif_name}_rmc_costs.txt"
        with open(costs_filename, "w") as f:
            f.write(f"Ground Truth: {ground_truth_energy}\n")
            f.write(f"Best Energy: {self.best_loss}\n")
            f.write(f"Final Acceptance Rate: {accepted_moves / self.iters:.2%}\n")
        
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
        
        iters = 5000  # RMC typically needs more iterations
        local_steps = 25  # Can set to 50 for periodic local optimization
        step_size = 0.3
        temperature = 1.0
        cell_perturb = True
        
        rmc = RMC(cif_name, model, composition, cell, calc, iters,
                  local_steps, step_size, temperature, cell_perturb)
        match, dist_energy = rmc.run()
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
