import sys
from ase.io import read
import torch
from argparse import ArgumentParser
import numpy as np
import sys
from rdkit import Chem
from openmm.openmm import System
from openmm import Platform, LangevinMiddleIntegrator
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from openmm.app import (
    Simulation,
    Topology,
    StateDataReporter,
    ForceField,
    PDBReporter,
    PDBFile,
    HBonds,
    Modeller,
    PME,
)
from typing import List, Optional, Tuple
from openmm.unit import nanometer, nanometers, molar
from openmm.unit import kelvin, picosecond, femtosecond, kilojoule_per_mole
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from mace.calculators.openmm import MacePotentialImplFactory
from openmmml import MLPotential

from openmmtools.openmm_torch.repex import (
    MixedSystemConstructor,
    RepexConstructor,
    get_atoms_from_resname,
)
from tempfile import mkstemp
import os


# COMPLEX = "tests/test_openmm/tnks_complex.pdb"
# LIGAND = "tests/test_openmm/5n.sdf"
# # platform = Platform.getPlatformByName("CUDA")
# # platform.setPropertyDefaultValue("DeterministicForces", "true")
# RESNAME = "UNK"

# standalone proof of concept script for running a neq switch from the MM description of the system to the MM/ML mixed system


def get_xyz_from_mol(mol):

    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return xyz


MLPotential.registerImplFactory("mace", MacePotentialImplFactory())
torch.set_default_dtype(torch.float32)


class MixedSystem:
    def __init__(
        self,
        file: str,
        smiles: str,
        model_path: str,
        forcefields: List[str],
        resname: str,
        padding: float,
        ionicStrength: float,
        nonbondedCutoff: float,
        potential: str,
        temperature: float,
        friction_coeff: float = 1.0,
        timestep: float = 1,
    ) -> None:

        self.forcefields = forcefields
        self.padding = padding
        self.ionicStrength = ionicStrength
        self.nonbondedCutoff = nonbondedCutoff
        self.resname = resname
        self.potential = potential
        self.temperature = temperature
        self.friction_coeff = friction_coeff / picosecond
        self.timestep = timestep * femtosecond

        self.mixed_system, self.modeller = self.create_mixed_system(
            file=file, smiles= smiles, model_path=model_path
        )

    def initialize_mm_forcefield(self, molecule: Optional[Molecule] = None):
        forcefield = ForceField(*self.forcefields)
        if molecule is not None:
            print("registering smirnoff template generator")
            smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
            forcefield.registerTemplateGenerator(smirnoff.generator)
        return forcefield


    def create_mixed_system(
        self, file: str, model_path: str, smiles: str = None, 
    ) -> Tuple[System, Modeller]:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str smiles: smiles of the small molecule, only required when passed as part of the complex
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        
        molecule = Molecule.from_smiles(smiles)
        _, tmpfile = mkstemp(suffix="xyz")
        molecule._to_xyz_file(tmpfile)
        atoms = read(tmpfile)
        os.remove(tmpfile)
        # Handle a complex, passed as a pdb file
        if file.endswith(".pdb"):
            input_file = PDBFile(file)
            modeller = Modeller(input_file.topology, input_file.positions)
        # Handle a ligand, passed as an sdf
        elif file.endswith(".sdf"):
            input_file = Molecule.from_file(file)
            modeller = Modeller(
                input_file.to_topology().to_openmm(),
                get_xyz_from_mol(input_file.to_rdkit()),
            )
        forcefield = self.initialize_mm_forcefield(molecule)
        modeller.addSolvent(
            forcefield, padding=self.padding, ionicStrength=self.ionicStrength * molar
        )

        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=self.nonbondedCutoff * nanometer,
            constraints=HBonds,
            rigidWater=True,
            removeCMMotion=False

        )

        mixed_system = MixedSystemConstructor(
            system=mm_system,
            topology=modeller.topology,
            nnpify_resname=self.resname,
            nnp_potential=self.potential,
            atoms_obj=atoms,
            filename=model_path,
        ).mixed_system

        return mixed_system, modeller

    def run_mixed_md(self, steps: int, interval: int):
        """Runs plain MD on the mixed system, writes a pdb trajectory

        :param int steps: number of steps to run the simulation for
        :param int interval: reportInterval attached to reporters
        """
        integrator = LangevinMiddleIntegrator(
            self.temperature, self.friction_coeff, self.timestep
        )

        simulation = Simulation(
            self.modeller.topology,
            self.mixed_system,
            integrator,
            platformProperties={"Precision": "Mixed"},
        )
        simulation.context.setPositions(self.modeller.getPositions())
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
            reportInterval=interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
        simulation.reporters.append(reporter)
        simulation.reporters.append(
            PDBReporter(
                file="output_complex.pdb",
                reportInterval=interval,
                enforcePeriodicBox=True,
            )
        )

        simulation.step(steps)
        state = simulation.context.getState(getEnergy=True)
        energy_2 = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        print(energy_2)

    def run_replex_equilibrium_fep(self, replicas: int) -> None:

        sampler = RepexConstructor(
            mixed_system=self.mixed_system,
            initial_positions=self.modeller.getPositions(),
            repex_storage_file="./out_complex.nc",
            temperature=self.temperature * kelvin,
            n_states=replicas,
            storage_kwargs={
                "storage": "/home/jhm72/rds/hpc-work/mace-openmm/repex.nc",
                "checkpoint_interval": 100,
                "analysis_particle_indices": get_atoms_from_resname(
                    topology=self.modeller.topology, resname=self.resname
                ),
            },
        ).sampler

        sampler.minimize()

        print("Minimised system")

        sampler.run()

    def run_neq_switching(self, steps: int, interval: int) -> float:
        """Compute the protocol work performed by switching from the MM description to the MM/ML through lambda_interpolate

        :param int steps: number of steps in non-equilibrium switching simulation
        :param int interval: reporterInterval
        :return float: protocol work from the integrator
        """
        alchemical_functions = {"lambda_interpolate": "lambda"}
        integrator = AlchemicalNonequilibriumLangevinIntegrator(
            alchemical_functions=alchemical_functions,
            nsteps_neq=steps,
            temperature=self.temperature,
            collision_rate=self.friction_coeff,
            timestep=self.timestep,
        )

        simulation = Simulation(
            self.modeller.topology,
            self.mixed_system,
            integrator,
            platformProperties={"Precision": "Mixed"},
        )
        simulation.context.setPositions(self.modeller.getPositions())

        print("Minimising energy")
        simulation.minimizeEnergy()

        reporter = StateDataReporter(
            file=sys.stdout,
            reportInterval=interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            totalSteps=steps,
            remainingTime=True,
        )
        simulation.reporters.append(reporter)
        # Append the snapshots to the pdb file
        simulation.reporters.append(
            PDBReporter("output_frames.pdb", steps / 80, enforcePeriodicBox=True)
        )
        # We need to take the final state
        simulation.step(steps)
        print(
            "work done during switch from mm to ml",
            integrator.get_protocol_work(dimensionless=True),
        )
