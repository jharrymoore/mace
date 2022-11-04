from e3nn.util import jit
import torch
from torch_nl import compute_neighborlist, compute_neighborlist_n2
import mace
from mace.calculators.neighbour_list_torch import primitive_neighbor_list_torch
from mace import data
from mace.tools import torch_geometric, utils
from typing import Optional, Iterable, List
import openmm
from openmmtorch import TorchForce
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
from ase import Atoms
from openmm.app import Topology


# torch.set_default_dtype(torch.float64)


def compile_model(model_path):
    model = torch.load(model_path)
    res = {}
    res["model"] = jit.compile(model)
    res["z_table"] = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    res["r_max"] = model.r_max
    return res


class MACE_openmm_old(torch.nn.Module):
    def __init__(self, model_path, atoms_obj):
        super().__init__()
        dat = compile_model(model_path)
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
        self.model = dat["model"]
        self.r_max = dat["r_max"]

    def forward(self, positions):
        sender, receiver, unit_shifts = primitive_neighbor_list_torch(
            quantities="ijS",
            pbc=(True, True, True),
            cell=self.inp_dict["cell"],
            positions=positions,
            cutoff=self.r_max,
            self_interaction=True,  # we want edges from atom to itself in different periodic images
            use_scaled_positions=False,  # positions are not scaled positions
            device="cpu",
        )
        # openMM passes positions in nanometers
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
        res = self.model(inp_dict_this_config)
        return (res["energy"], res["forces"])


class MACE_openmm(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        atom_indices: Optional[Iterable] = None,
        nl: str = "torch_nl_n2",
        atoms_obj: Optional[Atoms] = None,
        topology: Optional[Topology] = None,
        device: str = "cuda",
    ):
        super().__init__()
        if nl == "torch_nl":
            self.nl = compute_neighborlist
        elif nl == "torch_nl_n2":
            self.nl = compute_neighborlist_n2
        else:
            raise NotImplementedError
        self.device = torch.device(device)
        self.atom_indices = atom_indices
        dat = compile_model(model_path)
        # TODO: if only the topology was passed, create the config from this
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
        batch = next(iter(data_loader)).to(self.device)
        batch_dict = batch.to_dict()
        batch_dict.pop("edge_index")
        batch_dict.pop("energy", None)
        batch_dict.pop("forces", None)
        batch_dict.pop("positions")
        # batch_dict.pop("shifts")
        batch_dict.pop("weight")
        self.inp_dict = batch_dict
        self.model = dat["model"]
        self.r_max = dat["r_max"]

    def forward(self, positions, boxVectors):
        # openMM hands over the entire topology to the forward model, we need to select the subset involved in the ML computation
        positions = (
            positions[self.atom_indices] if self.atom_indices is not None else positions
        )
        positions = positions * 10
        boxVectors = boxVectors * 10
        boxVectors = boxVectors.type(torch.float32).to(self.device)
        bbatch = torch.zeros(positions.shape[0], dtype=torch.long, device=self.device)
        mapping, batch_mapping, shifts_idx = self.nl(
            cutoff=self.r_max,
            pos=positions.to(self.device),
            cell=boxVectors,
            pbc=torch.tensor([True, True, True], device=self.device),
            batch=bbatch,
        )

        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = mapping[0] == mapping[1]
        true_self_edge &= torch.all(shifts_idx == 0, dim=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = mapping[0][keep_edge]
        receiver = mapping[1][keep_edge]
        shifts_idx = shifts_idx[keep_edge]

        edge_index = torch.stack((sender, receiver))

        # From the docs: With the shift vector S, the distances D between atoms can be computed from
        # D = positions[j]-positions[i]+S.dot(cell)
        # shifts = torch.dot(unit_shifts, self.inp_dict["cell"])  # [n_edges, 3]
        inp_dict_this_config = self.inp_dict.copy()
        inp_dict_this_config["positions"] = positions.to(self.device)
        inp_dict_this_config["edge_index"] = edge_index
        inp_dict_this_config["shifts"] = shifts_idx
        # inp_dict_this_config["shifts"] = shifts
        # inp_dict_this_config[""] =
        res = self.model(inp_dict_this_config)
        # return (res["energy"], res["forces"])
        # return res["scaled_interaction_energy"]
        return res["energy"]


class MacePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return MacePotentialImpl(name)


class MacePotentialImpl(MLPotentialImpl):
    """
    Implements the mace potential in openMM, using TorchForce to add it to an openMM system
    """

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        # FIXME: this should not be a hardcoded filepath
        filename: str = "/home/jhm72/rds/hpc-work/mace-openmm/tests/test_openmm/MACE_SPICE.model",
        implementation: str = "nnpops",
        **args,
    ):
        model = torch.load(filename)
        model = jit.compile(model)
        print("MACE model compiled")
        # TODO: this should take a topology
        # A bit hacky to add the atoms object like this
        openmm_calc = MACE_openmm(filename, atom_indices=atoms, **args)
        jit.script(openmm_calc).save("md_test_mace.pt")
        force = TorchForce("md_test_mace.pt")
        # force.setOutputsForces(True)
        force.setUsesPeriodicBoundaryConditions(True)
        # modify the system in place to add the force
        system.addForce(force)
