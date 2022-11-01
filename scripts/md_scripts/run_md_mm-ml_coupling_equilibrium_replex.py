#
import sys
from ase.io import read
import torch

import sys
from openmm import Platform, LangevinMiddleIntegrator
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from openmm.app import (
    Simulation,
    StateDataReporter,
    ForceField,
    PDBReporter,
    PDBFile,
    HBonds,
    Modeller,
    PME,
)
from openmm.unit import nanometer, nanometers, molar
from openmm.unit import kelvin, picosecond, femtosecond, kilojoule_per_mole
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from mace.calculators.openmm import MacePotentialImplFactory
from openmmml import MLPotential

from openmmtools.openmm_torch.repex import RepexConstructor, MixedSystemConstructor

MLPotential.registerImplFactory("mace", MacePotentialImplFactory())
torch.set_default_dtype(torch.float32)

COMPLEX = "tests/test_openmm/tnks_complex.pdb"
LIGAND = "tests/test_openmm/5n.sdf"
platform = Platform.getPlatformByName("CUDA")
platform.setPropertyDefaultValue("DeterministicForces", "true")

# standalone proof of concept script for running a neq switch from the MM description of the system to the MM/ML mixed system


def main():

    molecule = Molecule.from_file(LIGAND)
    cplx = PDBFile(COMPLEX)
    modeller = Modeller(cplx.topology, cplx.positions)
    ligand_xyz = LIGAND.split(".")[0] + ".xyz"

    forcefield = ForceField(
        "amber/protein.ff14SB.xml",
        "amber/tip3p_standard.xml",
        # "amber/tip3p_HFE_multivalent.xml",
    )
    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
    forcefield.registerTemplateGenerator(smirnoff.generator)
    modeller.addSolvent(
        forcefield, padding=1.2 * nanometers, neutralize=True, ionicStrength=0.1 * molar
    )
    mm_system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1 * nanometer,
        constraints=HBonds,
    )

    # We can replace this with the openmmtools constructur fr the nxed system, is just a little bit nicer

    mixedSystem = MixedSystemConstructor(
        system=mm_system,
        topology=modeller.topology,
        nnpify_resname="UNK",
        nnp_potential="mace",
    ).mixed_system

    sampler = RepexConstructor(
        mixed_system=mixedSystem,
        positions=modeller.getPositions(),
        repex_storage_file="./out_complex.nc",
        temperature=300 * kelvin,
        n_states=5,
    ).sampler

    sampler.minimize()

    sampler.run()


if __name__ == "__main__":
    main(*sys.argv[1:])
