# Script to read in the equilibrium simulations from gromacs (4x) and prepare the OpenMM systems for each, then apply the perturbative correction by switching to the mixed system

import os
from glob import glob
import sys
from parmed import load_file

from openmm.app import PME, HBonds
from openmm import Platform
from openmm.unit import nanometer
from tqdm import tqdm
from mace.tools.compute_neq_dG import NEQCorrection

platform = Platform.getPlatformByName("CUDA")


from parmed import gromacs


gromacs.GROMACS_TOPDIR = "/home/jhm72/rds/hpc-work/pmx/src/pmx/data/mutff"
# gromacs.GROMACS_TOPDIR = "/usr/local/software/spack/spack-views/rocky8-icelake-20220710/gromacs-2021.5/intel-2021.6.0/intel-oneapi-mkl-2022.1.0/intel-oneapi-mpi-2021.6.0/i7ewgrsxn36bpv47pxfto7cmgjjfdbvy/share/gromacs/top"


def compute_ml_corrections(edge_path: str):
    # We want to work over the standard structure of a standard PMX output, but only run upto the equilibrium trajectories stage, actually we can leverage trjconv to generate the snapshots directly, instead of having to work out how to do this from openMM

    # we take an edge path containing all the simulation output, and hijack the sampled frames from trjconv to perform the corrections

    # will need to loop over replicas, eventually, and probably dispatch the jobs to independent nodes once we have them prepared

    for branch in ["unbound", "bound"]:
        for state in ["stateA", "stateB"]:
            top_file = glob(os.path.join(edge_path, branch, "*.top"))[0]
            top = load_file(top_file)
            gro = [
                load_file(f)
                for f in tqdm(
                    glob(
                        os.path.join(
                            edge_path,
                            branch,
                            state,
                            "run1",
                            "transitions",
                            "frame*.gro",
                        )
                    )[:5]
                )
            ]
            gro_file = gro[0]
            top.box = gro_file.box[:]

            omm_system = top.createSystem(
                nonbondedMethod=PME,
                nonbondedCutoff=1 * nanometer,
                constraints=HBonds,
            )
            # Create the system independently of the positions
            snapshots = [f.positions for f in gro]

            ml_correction = NEQCorrection(
                omm_system, top, platform=platform, snapshots=snapshots
            )

            ml_correction.apply_ml_correction()

            print(
                f"ml correction for {branch}, {state}:",
                ml_correction.dG,
                "±",
                ml_correction.dG_err,
            )


if __name__ == "__main__":
    compute_ml_corrections(sys.argv[1])
