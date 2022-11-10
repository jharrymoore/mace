import sys
from ase.io import read
import torch

import sys
from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import (
    Simulation,
    StateDataReporter,
    ForceField,
    PDBReporter,
    Modeller,
    PME,
    HBonds,
)
from openmm.unit import (
    kelvin,
    picosecond,
    femtosecond,
    kilojoule_per_mole,
    nanometer,
    nanometers,
    angstrom,
    molar,
)
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmml import MLPotential
from mace.calculators.openmm import MacePotentialImplFactory

MLPotential.registerImplFactory("mace", MacePotentialImplFactory())
torch.set_default_dtype(torch.float32)
platform = Platform.getPlatformByName("CUDA")


def main(filename: str):
    # load a starting configuration into an openmm system object
    molecule = Molecule.from_file(filename)
    off_topology = molecule.to_topology()
    omm_top = off_topology.to_openmm()
    atoms_xyz = filename.split(".")[0] + ".xyz"
    atoms = read(atoms_xyz)
    modeller = Modeller(omm_top, atoms.positions / 10)
    forcefield = ForceField(
        "amber/tip3p_standard.xml",
        # "amber/tip3p_HFE_multivalent.xml",
    )
    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
    forcefield.registerTemplateGenerator(smirnoff.generator)
    modeller.addSolvent(
        forcefield, padding=1.5 * nanometers, neutralize=True, ionicStrength=0.1 * molar
    )
    mm_system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1 * nanometer,
        constraints=HBonds,
    )
    omm_box_vecs = modeller.topology.getPeriodicBoxVectors()
    print("Box size", omm_box_vecs[0][0].value_in_unit(angstrom))

    atoms.set_cell(
        [
            omm_box_vecs[0][0].value_in_unit(angstrom),
            omm_box_vecs[1][1].value_in_unit(angstrom),
            omm_box_vecs[2][2].value_in_unit(angstrom),
        ]
    )
    mace_potential = MLPotential("mace")
    chains = list(modeller.topology.chains())
    print(chains)
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    assert len(ml_atoms) == len(atoms)
    system = mace_potential.createMixedSystem(
        modeller.topology, mm_system, ml_atoms, atoms_obj=atoms
    )
    print("Preparing OpenMM Simulation...")

    temperature = 298.15 * kelvin
    frictionCoeff = 1 / picosecond
    timeStep = 1 * femtosecond
    integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)
    # integrator.setRandomNumberSeed(42)

    simulation = Simulation(
        modeller.topology,
        system,
        integrator,
        platform=platform,
        # platformProperties={"Precision": "Single"},
    )
    simulation.context.setPositions(modeller.getPositions())
    print("Minimising energy")
    simulation.minimizeEnergy()

    reporter = StateDataReporter(
        file=sys.stdout,
        reportInterval=100,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        speed=True,
    )
    simulation.reporters.append(PDBReporter("output_solvated_test_mol_large.pdb", 1000))
    simulation.reporters.append(reporter)

    simulation.step(1000000)
    state = simulation.context.getState(getEnergy=True)
    energy_2 = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(energy_2)


if __name__ == "__main__":

    main(*sys.argv[1:])
