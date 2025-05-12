from ase import Atoms
import ase
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from mattertune.backbones import MatterSimM3GNetBackboneModule, MatterSimBackboneConfig
from mattertune import configs as MC
#from matdeeplearn.common.ase_utils import MDLCalculator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("../.."))))
from msp.utils.objectives import Energy
from msp.forcefield import MDL_FF
from ase.optimize import BFGS
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
import json
import logging
from pyxtal import pyxtal
import periodictable
from pathlib import Path
from collections import Counter
from loss_calculator import calculate_loss

logging.getLogger("mattertune").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("pandas").setLevel(logging.CRITICAL)

def extract_cell(cif_path):
    structure = Structure.from_file(cif_path)
    cell = structure.lattice.matrix
    return cell.tolist()

def extract_composition(cif_path):
    structure = Structure.from_file(cif_path)
    composition = [site.specie.Z for site in structure]
    return composition

class PSO():
    def __init__(self, cif_name, model, composition, cell, options, particles, iters, local_steps, cell_perturb=True):
        self.cif_name = cif_name
        self.cell_perturb = cell_perturb
        self.composition = composition
        if cell is not None:
            self.cell = cell

        self.options = options
        self.particles = particles
        self.iters = iters
        self.local_steps = local_steps

        self.zs, self.zcounts = self.composition_to_zs()
        self.possible_sgs, self.sg_probs = self.generate_sgs(self.zs, self.zcounts)

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

        self.calculator = model.ase_calculator()
        # self.calculator = MDLCalculator(config=train_config)

    def composition_to_zs(self):
        counter = Counter(self.composition)
        zs = [periodictable.elements[z].symbol for z, _ in counter.items()]
        zcounts = [count for _, count in counter.items()]
        return zs, zcounts

    def generate_sgs(self, zs, zcounts, seed=0):
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

    def get_z(self, site):
        return np.argmax(self.el_symbols == site.species.elements[0].symbol)

    def lj_reject(self, structure):
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                if structure.sites[i].distance(structure.sites[j]) < self.lj_rmins[self.get_z(
                        structure.sites[i]) - 1][self.get_z(structure.sites[j]) - 1]:
                    return True
        return False

    def init_atoms(self, density=0.2):
        # if not self.cell_perturb:
        #     atoms = Atoms(self.composition, cell=self.cell, pbc=(True, True, True))
        #
        #     scaled_positions = np.random.uniform(0., 1., (len(atoms), 3))
        #     atoms.set_scaled_positions(scaled_positions)
        # else:
        #     beta = np.random.uniform(0, 180)
        #     gamma = np.random.uniform(0, 180)
        #     minCosA = - np.sin(gamma * np.pi/180) * np.sqrt(1 - np.cos(beta* np.pi/180) ** 2) + np.cos(beta * np.pi/180) * np.cos(gamma * np.pi/180)
        #     maxCosA = np.sin(gamma * np.pi/180) * np.sqrt(1 - np.cos(beta* np.pi/180) ** 2) + np.cos(beta * np.pi/180) * np.cos(gamma * np.pi/180)
        #     alpha = np.random.uniform(minCosA, maxCosA)
        #     alpha = np.arccos(alpha) * 180 / np.pi
        #     a = np.random.rand() + .000001
        #     b = np.random.rand() + .000001
        #     c = np.random.rand() + .000001
        #     cell=[a, b, c, alpha, beta, gamma]
        #
        #     atoms = Atoms(self.composition, cell=cell, pbc=(True, True, True))
        #     vol = atoms.get_cell().volume
        #
        #     ideal_vol = len(self.composition) / density
        #     scale = (ideal_vol / vol) ** (1/3)
        #     cell = [scale * a, scale * b, scale * c, alpha, beta, gamma]
        #     atoms.set_cell(cell)
        #     scaled_positions = np.random.uniform(0., 1., (len(atoms), 3))
        #     atoms.set_scaled_positions(scaled_positions)

        """
        pyxtal
        """
        seed = 0
        rng = np.random.default_rng(seed)
        rejected = True
        while rejected:
            try:
                xtal = pyxtal()
                xtal.from_random(3, np.random.choice(self.possible_sgs,
                                                     p=self.sg_probs), self.zs, self.zcounts, random_state=rng)
                new_structure = xtal.to_pymatgen()
                rejected = self.lj_reject(new_structure)
            except:
                rejected = True

        atoms = xtal.to_ase()
        return atoms

    def dimensions_to_atoms(self, params):
        if not self.cell_perturb:
            positions = params.reshape(-1, 3)
            atoms = Atoms(self.composition, cell=self.cell, pbc=(True, True, True), positions=positions)
        else:
            cell = params[:9].reshape(-1, 3)
            positions = params[9:].reshape(-1, 3)
            atoms = Atoms(self.composition, cell=cell, pbc=(True, True, True), positions=positions)

        atoms.set_calculator(self.calculator)
        return atoms

    def atoms_to_dimensions(self, atoms):
        if not self.cell_perturb:
            pos = [float(i) for l in atoms.positions for i in l]
        else:
            pos = [float(i) for l in atoms.cell for i in l][:9] + [float(i) for l in atoms.positions for i in l]

        return np.array(pos)

    def obj_func(self, params):
        atoms = self.dimensions_to_atoms(params)

        atoms.set_calculator(self.calculator)
        loss = atoms.get_potential_energy()

        if loss < self.best_loss:
            self.best_loss = loss

        return loss

    def f(self, x):
        n_particles = x.shape[0]
        j = [self.obj_func(x[i]) for i in range(n_particles)]

        self.best_losses.append(self.best_loss)
        self.avg_losses.append(np.mean(j))

        return np.array(j)

    def run(self):
        costs = []
        matches = []
        matcher = StructureMatcher()

        os.makedirs("plots", exist_ok=True)
        os.makedirs("matches", exist_ok=True)
        os.makedirs("fails", exist_ok=True)

        ground_truth_energy = calculate_loss(f"cifs/{self.cif_name}.cif", 'mdl_config.yml')

        for iteration in range(1):
            self.best_losses = []
            self.best_loss = float('inf')
            self.avg_losses = []

            options = self.options  # cognitive, social, inertia
            particles = self.particles  # number of particles in system
            iters = self.iters
            steps = self.local_steps
            if not self.cell_perturb:
                dimensions = len(self.composition)*3
            else:
                dimensions = 9 + len(self.composition)*3  # first 9 are cell, rest are atom positions

            init_positions = np.empty((particles, dimensions))
            for i in range(particles):
                init_atoms = self.init_atoms()
                if self.cell_perturb:
                    flattened_cell = [i for l in init_atoms.get_cell().tolist() for i in l]
                    flattened_pos = [float(i) for l in init_atoms.positions for i in l]
                    init_pos = flattened_cell + flattened_pos
                else:
                    init_pos = [float(i) for l in init_atoms.positions for i in l]
                init_positions[i] = np.array(init_pos)
            filename = f"init_structure_psolocal_{iteration}" + ".cif"
            ase.io.write(filename, self.dimensions_to_atoms(init_positions[0]))

            self.optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options, init_pos=init_positions)
            #self.optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options, init_pos=init_positions, oh_strategy={'w':'exp_decay'})


            #cost, pos = self.optimizer.optimize(self.f, iters=10)
            for i in range(iters):
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

                print(f"Iteration {i + 1}: Ground Truth: {ground_truth_energy}, Best Cost = {self.optimizer.swarm.best_cost}")

                #local optimization
                positions = self.optimizer.swarm.position
                new_atoms = [self.dimensions_to_atoms(positions[i]) for i in range(len(positions))]

                def localopt(atoms, steps=1):
                    optimizer = BFGS(atoms, logfile=None)
                    optimizer.run(fmax=0.01, steps=steps)
                    return atoms

                optimized_atoms = [localopt(atoms, steps=steps) for atoms in new_atoms]
                self.optimizer.swarm.current_cost = np.array([atoms.get_potential_energy() for atoms in optimized_atoms])
                self.optimizer.swarm.position = np.array([self.atoms_to_dimensions(optimized_atoms[i]) for i in range(len(optimized_atoms))])


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

            optimized_structure = AseAtomsAdaptor.get_structure(self.dimensions_to_atoms(pos))
            original_cif = os.path.join("cifs", self.cif_name + ".cif")
            ground_truth = Structure.from_file(original_cif)
            matched = matcher.fit(ground_truth, optimized_structure)
            matches.append(matched)

            if matched:
                filename = f"matches/best_structure_{self.cif_name}_{iteration}" ".cif"
                ase.io.write(filename, self.dimensions_to_atoms(pos))
            else:
                filename = f"fails/best_structure_{self.cif_name}_{iteration}" ".cif"
                ase.io.write(filename, self.dimensions_to_atoms(pos))

        costs_filename = f"plots/{self.cif_name}_costs.txt"
        with open(costs_filename, "w") as f:
            f.write(f"Ground Truth: {ground_truth_energy}\n")
            for cost in costs:
                f.write(f"{cost}\n")

        return matches



if __name__ == "__main__":
    config = MC.MatterSimBackboneConfig(
        pretrained_model="MatterSim-v1.0.0-5M",
        graph_convertor=MC.MatterSimGraphConvertorConfig(
            twobody_cutoff=5.0,
            has_threebody=True,
            threebody_cutoff=4.0
        ),
        properties=[
            MC.EnergyPropertyConfig(
                loss=MC.MAELossConfig(),
                loss_coefficient=1.0
            ),
            MC.ForcesPropertyConfig(
                loss=MC.MAELossConfig(),
                loss_coefficient=10.0,
                conservative=True
            ),
            MC.StressesPropertyConfig(
                loss=MC.MAELossConfig(),
                loss_coefficient=1.0,
                conservative=True
            ),
        ],
        optimizer=MC.AdamWConfig(lr=1e-4),
        lr_scheduler=MC.CosineAnnealingLRConfig(
            T_max=100,
            eta_min=1e-6
        )
    )
    model = MatterSimM3GNetBackboneModule(config)

    all_matches = []

    for filename in os.listdir("cifs"):
        cif = os.path.join("cifs", filename)
        cif_name = os.path.splitext(filename)[0]

        composition = extract_composition(cif)
        #cell = extract_cell(cif)

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # cognitive, social, inertia
        particles = 10  # number of particles in system
        iters = 50
        local_steps = 100

        cell_perturb = True
        pso = PSO(cif_name, model, composition, None, options, particles, iters, local_steps, cell_perturb)
        matches = pso.run()
        all_matches.extend(matches)
        print(f"{cif_name} match: {matches}")

    num_true = sum(all_matches)
    total = len(all_matches)
    match_ratio = num_true / total if total > 0 else 0

    print(f"\nMatched {num_true}/{total} structures.")
    print(f"Match ratio: {match_ratio:.2f}")