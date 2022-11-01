#
import sys
from ase.io import read
import torch

import sys
from openmm import Platform
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
    ml_atoms = [atom.index for atom in chains[3].atoms()]
    print(f"selected {len(ml_atoms)} atoms for evaluation by the ML potential")
    assert len(ml_atoms) == len(atoms)
    potential = MLPotential("mace")
    # set interpolate flag, defines lambda_interpolate to control the switching.  Call setParameter() on the context
    system = potential.createMixedSystem(
        modeller.topology, mm_system, ml_atoms, atoms_obj=atoms, interpolate=True
    )

    print("Preparing OpenMM Simulation...")

    temperature = 298.15 * kelvin
    frictionCoeff = 1 / picosecond
    timeStep = 1 * femtosecond
    # in the paper they do a 10 ps switching time, leaving at 1ps for testing
    n_step_neq = 1000

    # TODO: maybe easier to manually step the integrator through nsteps_neq instead of creating the simulation environment
    alchemical_functions = {"lambda_interpolate": "lambda"}
    integrator = AlchemicalNonequilibriumLangevinIntegrator(
        alchemical_functions=alchemical_functions,
        nsteps_neq=n_step_neq,
        temperature=temperature,
        collision_rate=frictionCoeff,
        timestep=timeStep,
    )

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
        totalSteps=n_step_neq,
        remainingTime=True,
    )
    simulation.reporters.append(reporter)
    # Append the snapshots to the pdb file
    simulation.reporters.append(
        PDBReporter("output_frames.pdb", n_step_neq / 80, enforcePeriodicBox=True)
    )
    # We need to take the final state
    simulation.step(n_step_neq)
    print(
        "work done during switch from mm to ml",
        integrator.get_protocol_work(dimensionless=True),
    )
    state = simulation.context.getState(getEnergy=True)
    energy_2 = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(energy_2)


if __name__ == "__main__":
    main(*sys.argv[1:])
