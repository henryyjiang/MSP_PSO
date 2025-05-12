import ase.io
#from matdeeplearn.common.ase_utils import MDLCalculator
from mattertune.backbones import MatterSimM3GNetBackboneModule, MatterSimBackboneConfig
from mattersim.forcefield import MatterSimCalculator
from mattertune import configs as MC

def calculate_loss(cif_file, train_config):
    """
    Calculate the loss (potential energy) for an atomic structure in a .cif file.

    Parameters:
        cif_file (str): Path to the .cif file containing the atomic structure.
        train_config (dict): Configuration for the MDLCalculator.

    Returns:
        float: The calculated potential energy (loss).
    """
    try:
        # Load the atomic structure from the .cif file
        atoms = ase.io.read(cif_file)

        # Set up the calculator with the training configuration
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

        atoms.set_calculator(model.ase_calculator())

        #calculator = MDLCalculator(config=train_config)
        #atoms.set_calculator(calculator)

        # Calculate the potential energy
        loss = atoms.get_potential_energy()
        return loss

    except Exception as e:
        print(f"Error occurred: {e}")
        return None


# Example usage
if __name__ == "__main__":
    cif_file_path = "cifs/AlCuO2.cif"

    train_config = 'mdl_config.yml'

    loss_value = calculate_loss(cif_file_path, train_config)
    if loss_value is not None:
        print(f"Calculated Loss (Potential Energy): {loss_value} eV")
