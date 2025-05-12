from ase import Atoms
import ase
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from mattertune.backbones import MatterSimM3GNetBackboneModule, MatterSimBackboneConfig
from mattertune import configs as MC
from matdeeplearn.common.ase_utils import MDLCalculator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("../.."))))
from msp.utils.objectives import Energy
from msp.forcefield import MDL_FF
import json
import logging
from pyxtal import pyxtal
import periodictable
from pathlib import Path



logging.getLogger("mattertune").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("pandas").setLevel(logging.CRITICAL)

el_symbols = np.array([periodictable.elements[i].symbol for i in range(95)])
lj_rmins = np.genfromtxt(str(Path(__file__).parent / "lj_rmins.csv"),
  delimiter=",") * 0.85

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

seed=0
rng = np.random.default_rng(seed)
zs = ["Ti", "O"]
zcounts = [2, 4]
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

def get_z(site):
    return np.argmax(el_symbols == site.species.elements[0].symbol)


def lj_reject(structure):
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            if structure.sites[i].distance(structure.sites[j]) < lj_rmins[get_z(
                    structure.sites[i]) - 1][get_z(structure.sites[j]) - 1]:
                return True
    return False

class PSO():
    def __init__(self):
        self.cell_perturb = False
        self.composition = [22, 22, 8, 8, 8, 8]
        self.cell = [5.56641437, 5.56641437, 5.56641437, 140.05412510, 140.05412510, 57.77112661]
        self.best_losses = []
        self.best_loss = float('inf')
        self.avg_losses = []

        my_dataset = json.load(open("../data/data_subset_msp.json", "r"))
        train_config = 'mdl_config.yml'
        self.forcefield = MDL_FF(train_config, my_dataset)
        self.energy = Energy(normalize=True, ljr_ratio=1)

        self.optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=54, options={'c1': 0.5, 'c2': 0.3, 'w':0.9})

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

        self.calculator = model.ase_calculator()
        # self.calculator = MDLCalculator(config=train_config)

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
        rejected = True
        while rejected:
            try:
                xtal = pyxtal()
                xtal.from_random(3, np.random.choice(possible_sgs,
                                                     p=sg_probs), ['Ti', 'O'], [2, 4], random_state=rng)
                new_structure = xtal.to_pymatgen()
                rejected = lj_reject(new_structure)
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

    def obj_func(self):
        positions = self.optimizer.swarm.position
        new_atoms = [self.dimensions_to_atoms(positions[i]) for i in range(len(positions))]
        new_atoms, obj_loss, energy_loss, novel_loss, soft_sphere_loss = self.forcefield.optimize(new_atoms, steps=100,
                                                                                             objective_func=self.energy,
                                                                                             log_per=0,
                                                                                             learning_rate=.05,
                                                                                             batch_size=4,
                                                                                             cell_relax=True,
                                                                                             optim="Adam")
        raw_loss = []
        for j in range(len(new_atoms)):
            raw_loss.append(self.energy.norm_to_raw_loss(energy_loss[j][0], new_atoms[j].get_atomic_numbers()))

        for i in range(len(new_atoms)):
            with open('output.txt', 'a') as f:
                f.write('Structure ' + str(i) + '\n')
                f.write("\t\tObjective loss: " + str(obj_loss[i][0]) + '\n')
                f.write("\t\tEnergy loss: " + str(energy_loss[i][0]) + '\n')
                f.write("\t\tUnnormalized energy loss: " + str(raw_loss[i]) + '\n')
                f.write("\t\tNovel loss: " + str(novel_loss[i][0]) + '\n')
                f.write("\t\tSoft sphere loss: " + str(soft_sphere_loss[i][0]) + '\n\n\n')

        self.optimizer.swarm.position = np.array([self.atoms_to_dimensions(new_atoms[i]) for i in range(len(new_atoms))])

        return raw_loss

    def f(self, x):
        j = self.obj_func()

        for i in range(len(j)):
            if j[i] < self.best_loss:
                self.best_loss = j[i]
        self.best_losses.append(self.best_loss)
        self.avg_losses.append(np.mean(j))

        return np.array(j)

    def run(self):
        self.best_losses = []
        self.best_loss = float('inf')
        self.avg_losses = []

        costs = []
        for iteration in range(2):
            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # cognitive, social, inertia
            particles = 10  # number of particles in system
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
            filename = f"init_structure_psobasin_{iteration}" + ".cif"
            ase.io.write(filename, self.dimensions_to_atoms(init_positions[0]))

            self.optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options, init_pos=init_positions)
            #self.optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options, init_pos=init_positions, oh_strategy={'w':'exp_decay'})


            with open('output.txt', 'w') as f:
                pass

            cost, pos = self.optimizer.optimize(self.f, iters=500)
            costs.append(cost)

            filename = f"best_structure_psobasin_{iteration}" ".cif"
            ase.io.write(filename, self.dimensions_to_atoms(pos))

            plt.plot(self.best_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Best Loss')
            plt.title('Best Losses')
            plt.savefig(f'best_losses_psobasin_{iteration}.png')
            plt.close()

            plt.plot(self.avg_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title('Average Losses')
            plt.savefig(f'avg_losses__psobasin_{iteration}.png')
            plt.close()

        return costs

if __name__ == "__main__":
    pso = PSO()
    costs = pso.run()
    print(costs)