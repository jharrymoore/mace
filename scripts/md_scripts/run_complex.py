import sys
from ase.io import read
import torch
import numpy as np

# from mace.calculators import MACE_openmm2

import sys
from openmm import LangevinMiddleIntegrator, Platform, Vec3
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

# we would like to parametrise a full protein ligand system


# register the impl factory so we can call it later
MLPotential.registerImplFactory("mace", MacePotentialImplFactory())
torch.set_default_dtype(torch.float32)


def main(cplx: str, ligand: str):
    # load a starting configuration into an openmm system object
    platform = Platform.getPlatformByName("CUDA")
    molecule = Molecule.from_file(ligand)
    cplx = PDBFile(cplx)
    modeller = Modeller(cplx.topology, cplx.positions)
    ligand_xyz = ligand.split(".")[0] + ".xyz"
    atoms = read(ligand_xyz)

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
    chains = list(modeller.topology.chains())
    print(chains)
    ml_atoms = [atom.index for atom in chains[3].atoms()]
    print(f"selected {len(ml_atoms)} atoms for evaluation by the ML potential")
    assert len(ml_atoms) == len(atoms)
    potential = MLPotential("mace")
    system = potential.createMixedSystem(
        modeller.topology, mm_system, ml_atoms, atoms_obj=atoms
    )

    print("Preparing OpenMM Simulation...")

    temperature = 298.15 * kelvin
    frictionCoeff = 1 / picosecond
    timeStep = 1 * femtosecond
    integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)

    simulation = Simulation(
        modeller.topology,
        system,
        integrator,
        platform=platform,
        platformProperties={"Precision": "Single"},
    )
    simulation.context.setPositions(modeller.getPositions())
    state = simulation.context.getState(
        getEnergy=True,
        getVelocities=True,
        getParameterDerivatives=True,
        getForces=True,
        getPositions=True,
    )
    with open("init_pos.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.getPositions(), f)

    print("Minimising energy")
    simulation.minimizeEnergy()

    reporter = StateDataReporter(
        file=sys.stdout,
        reportInterval=1000,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        speed=True,
    )
    simulation.reporters.append(reporter)
    simulation.reporters.append(
        PDBReporter("output_complex.pdb", 100, enforcePeriodicBox=True)
    )

    simulation.step(100000)
    state = simulation.context.getState(getEnergy=True)
    energy_2 = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(energy_2)


if __name__ == "__main__":
    main(*sys.argv[1:])
