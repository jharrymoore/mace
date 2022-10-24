import sys
from ase.io import read
from openmmtorch import TorchForce
import torch
from e3nn.util import jit
from mace import data
from mace.tools import torch_geometric, utils
from mace.calculators import MACE_openmm

import sys
import simtk
from openmm import LangevinMiddleIntegrator, System, Platform
from openmm.app import Simulation, StateDataReporter, PDBFile, ForceField, HBonds, PME 
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, kilojoule_per_mole
from openmm import unit, app
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator





# TODO: do we need double precision?
torch.set_default_dtype(torch.float64)


def main(filename: str, model_path: str):
    # load a starting configuration into an openmm system object
    platform = Platform.getPlatformByName('CPU')
    molecule = Molecule.from_file(filename)


    # for now, simultaneously load the sdf into an ase.atoms object
    atoms = read("tests/test_openmm/test_one_mol.xyz")

    # I think we have to parametrise the thing first, even if we are going to immediately replace the thing
    forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')
    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
    forcefield.registerTemplateGenerator(smirnoff.generator)
    off_topology = molecule.to_topology()
    omm_top = off_topology.to_openmm()
    print(omm_top)
    system = forcefield.createSystem(omm_top)
    print(system)
    while system.getNumForces() > 0:
        system.removeForce(0)

    # now turn off the parameters for the small molecule
    # atoms = read(filename)
    model = torch.load(model_path)
    model = jit.compile(model)
    print('MACE model compiled')

    openmm_calc = MACE_openmm(model_path, atoms)
    jit.script(openmm_calc).save("md_test_model.pt")
    force = TorchForce("md_test_model.pt")
    force.setOutputsForces(True)
    
    system.addForce(force)
    print("Preparing OpenMM Simulation...")

    temperature = 298.15 * kelvin
    frictionCoeff = 1 / picosecond
    timeStep = 1 * femtosecond
    integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)

    simulation = Simulation(omm_top, system, integrator, platform=platform)
    simulation.context.setPositions(atoms.positions)

    reporter = StateDataReporter(file=sys.stdout, reportInterval=1, step=True, time=True, potentialEnergy=True, temperature=True)
    simulation.reporters.append(reporter)

    simulation.step(1000)
    state = simulation.context.getState(getEnergy=True)
    energy_2 = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(energy_2)



if __name__ == "__main__":
    main(*sys.argv[1:])