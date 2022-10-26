import sys
from ase.io import read
from openmmtorch import TorchForce
import torch
from e3nn.util import jit
from mace.calculators import MACE_openmm2

import sys
from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import Simulation, StateDataReporter, ForceField, PDBReporter
from openmm.unit import kelvin, picosecond, femtosecond, kilojoule_per_mole
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmml import MLPotential
from mace.calculators.openmm import MacePotentialImplFactory

# we would like to parametrise a full protein ligand system


MLPotential.registerImplFactory("mace", MacePotentialImplFactory())
torch.set_default_dtype(torch.float32)


def main(filename: str, model_path: str):
    # load a starting configuration into an openmm system object
    platform = Platform.getPlatformByName("CUDA")
    molecule = Molecule.from_file(filename)

    off_topology = molecule.to_topology()
    omm_top = off_topology.to_openmm()
    atoms = read(filename)

    mace_potential = MLPotential("mace")

    system = mace_potential.createSystem(omm_top, atoms_obj=atoms)

    print(system)
    print("Preparing OpenMM Simulation...")

    temperature = 298.15 * kelvin
    frictionCoeff = 1 / picosecond
    timeStep = 1 * femtosecond
    integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)

    simulation = Simulation(omm_top, system, integrator, platform=platform)
    simulation.context.setPositions(atoms.positions)

    reporter = StateDataReporter(
        file=sys.stdout,
        reportInterval=1,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        speed=True,
    )
    simulation.reporters.append(reporter)
    simulation.reporters.append(PDBReporter("output.pdb", 1))

    simulation.step(10000)
    state = simulation.context.getState(getEnergy=True)
    energy_2 = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(energy_2)


if __name__ == "__main__":
    main(*sys.argv[1:])
