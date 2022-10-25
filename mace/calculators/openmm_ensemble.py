from typing import List
from e3nn.util import jit
import torch
import mace
from mace.calculators.neighbour_list_torch import primitive_neighbor_list_torch
from mace import data
from mace.tools import torch_geometric, utils
import ase



# Load a series of models, that are in some state of being pretrained, evaluate them all on the input structure provided to forward, return avg energies, forces + stdev


torch.set_default_dtype(torch.float32)


def compile_model(model_path):
    model = torch.load(model_path)
    res = {}
    res["model"] = jit.compile(model)
    res["z_table"] = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    res["r_max"] = model.r_max
    return res


class MACE_openmm(torch.nn.Module):
    def __init__(self, model_paths: List[str], atoms_obj: ase.Atoms):
        super().__init__()
        dats = [compile_model(model_path) for model_path in model_paths]
        config = data.config_from_atoms(atoms_obj)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=dat["z_table"], cutoff=dat["r_max"]
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch_dict = next(iter(data_loader)).to_dict()
        batch_dict.pop("edge_index")
        batch_dict.pop("energy", None)
        batch_dict.pop("forces", None)
        batch_dict.pop("positions")
        # batch_dict.pop("shifts")
        batch_dict.pop("weight")
        self.inp_dict = batch_dict
        self.models = [dat["model"] for dat in dats]
        self.r_max = dats[0]["r_max"]

    def forward(self, positions):
        sender, receiver, unit_shifts = primitive_neighbor_list_torch(
            quantities="ijS",
            pbc=(False, False, False),
            cell=self.inp_dict["cell"],
            positions=positions,
            cutoff=self.r_max,
            self_interaction=True,  # we want edges from atom to itself in different periodic images
            use_scaled_positions=False,  # positions are not scaled positions
            device="cpu",
        )
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= torch.all(unit_shifts == 0, dim=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]
        # Build output
        edge_index = torch.stack((sender, receiver))  # [2, n_edges]

        # From the docs: With the shift vector S, the distances D between atoms can be computed from
        # D = positions[j]-positions[i]+S.dot(cell)
        # shifts = torch.dot(unit_shifts, self.inp_dict["cell"])  # [n_edges, 3]
        inp_dict_this_config = self.inp_dict.copy()
        inp_dict_this_config["positions"] = positions
        inp_dict_this_config["edge_index"] = edge_index
        # inp_dict_this_config["shifts"] = shifts
        # inp_dict_this_config[""] =
        
        # TODO: inefficient serial evaluation for now, we can do some fun vmap trickery here with functorch
        results = [model(inp_dict_this_config) for model in self.models]

        # compute stdev
        std_e, mu_e  = torch.std_mean([res["energy"] for res in results])
        std_f, mu_f = torch.std_mean([res["forces"] for res in results])
        
        return (std_e, std_f, mu_e, mu_f)
