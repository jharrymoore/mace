import torch
from ase.calculators.calculator import Calculator, all_changes

from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from typing import List

class MACECalculator(Calculator):
    """MACE ASE Calculator"""

    implemented_properties = ["energy", "forces", "std_e", "std_f"]

    def __init__(
        self,
        model_paths: List[str],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float32",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.models = [torch.load(f=model_path, map_location=device) for model_path in model_paths]
        self.r_max = self.model.r_max
        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )

        torch_tools.set_default_dtype(default_dtype)

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)

        # predict + extract data
        outs = [model(batch) for model in self.models]
        std_f, mu_f = torch.std_mean([out["forces"].detach().cpu().numpy() for out in outs])
        std_e, mu_e = torch.std_mean([out["energy"].detach().cpu().item()for out in outs])

        # store results
        self.results = {
            "energy": mu_e * self.energy_units_to_eV,
            # force has units eng / len:
            "forces": mu_f * (self.energy_units_to_eV / self.length_units_to_A),
            "std_e": std_e * self.energy_units_to_eV, 
            "std_f": std_f * (self.energy_units_to_eV / self.length_units_to_A)
        }
