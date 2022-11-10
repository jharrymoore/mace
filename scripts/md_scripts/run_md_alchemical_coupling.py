#
import sys
from ase.io import read
import torch
import mdtraj as md

# from mace.calculators import MACE_openmm2

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
from openmm.unit import nanometer, nanometers, molar, amu
from openmm.unit import kelvin, picosecond, femtosecond, kilojoule_per_mole
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from mace.calculators.openmm import MacePotentialImplFactory
from openmmml import MLPotential
from openmmtools import alchemy

from perses.annihilation.relative import HybridTopologyFactory
from perses.rjmc.atom_mapping import AtomMapper

MLPotential.registerImplFactory("mace", MacePotentialImplFactory())
torch.set_default_dtype(torch.float32)

LIGAND_1 = "tests/test_openmm/5n.sdf"
LIGAND_2 = "tests/test_openmm/5i.sdf"
COMPLEX = "test/test_openmm/tnks_complex.pdb"

# note this is future work - it is a fair bit of work to generate a hybrid topology that works with the mace potential


def main():
    # load a starting configuration into an openmm system object
    platform = Platform.getPlatformByName("CUDA")
    platform.setPropertyDefaultValue("DeterministicForces", "true")

    molecule1 = Molecule.from_file(LIGAND_1)
    molecule2 = Molecule.from_file(LIGAND_2)

    atomMapper = AtomMapper()
    atomMapper.get_best_mapping(molecule1, molecule2)

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
        rigidWater=True,
        hydrogenMass=4 * amu,
    )
    chains = list(modeller.topology.chains())
    print(chains)
    ml_atoms = [atom.index for atom in chains[3].atoms()]
    print(f"selected {len(ml_atoms)} atoms for evaluation by the ML potential")
    assert len(ml_atoms) == len(atoms)
    potential = MLPotential("mace")
    # set interpolate flag, defines lambda_interpolate to control the switching.  Call setParameter() on the context
    system = potential.createMixedSystem(
        modeller.topology, mm_system, ml_atoms, atoms_obj=atoms, interpolate=True
    )
    # Prepare the atom mapping between two ligands

    # add alchemical decoupling to the ML atoms
    # ligand_atoms = md.Topology.from_openmm(modeller.topology).select("resname UNK")
    # print(ligand_atoms)
    # alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=ligand_atoms)

    # factory = alchemy.AbsoluteAlchemicalFactory()
    # alchemical_system = factory.create_alchemical_system(system, alchemical_region)

    print("Preparing OpenMM Simulation...")

    temperature = 298.15 * kelvin
    frictionCoeff = 1 / picosecond
    timeStep = 1 * femtosecond
    n_step_neq = 10000
    # Define the coupling between the two end states
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
        alchemical_system,
        integrator,
        platform=platform,
        platformProperties={"Precision": "Single"},
    )
    simulation.context.setPositions(modeller.getPositions())
    simulation.context.setVelocitiesToTemperature(temperature)
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
        PDBReporter("output_comple_decoupling.pdb", 100, enforcePeriodicBox=True)
    )

    simulation.step(n_step_neq)
    print("work done during switch from mm to ml", integrator.total_work)
    state = simulation.context.getState(getEnergy=True)
    energy_2 = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(energy_2)


if __name__ == "__main__":
    main()
