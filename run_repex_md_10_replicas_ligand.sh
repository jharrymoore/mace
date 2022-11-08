#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks 12
#SBATCH -t 12:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:3
#SBATCH -A csanyi-SL3-GPU

source /home/jhm72/miniconda3/bin/activate /home/jhm72/miniconda3/envs/mlmm


# srun -n 8  --gpu-bind=map_gpu:0,0,0,0,1,1,1,1 python scripts/run_md.py -f tests/test_openmm/tnks_complex.pdb --smiles 'c1(ccc(cc1)C(C)(C)OCCO)c1[nH]c(=O)c2c(n1)ccc(c2)F'  --steps 100 --run_type repex --replicas 8 --log_level 10 --storage_path repex_new.nc


# srun -n 8  --gpu-bind=map_gpu:0,0,0,0,1,1,1,1 python scripts/run_md.py -f tests/test_openmm/5n.sdf --smiles 'c1(ccc(cc1)C(C)(C)OCCO)c1[nH]c(=O)c2c(n1)ccc(c2)F'  --run_type repex --replicas 8 --log_level 20 --storage_path repex_5n_solvent.nc --resname 5n


srun -n 12 --gpu-bind=map_gpu:0,0,0,0,1,1,1,1,2,2,2,2 python scripts/run_md.py -f tests/test_openmm/5n.sdf --smiles 'c1(ccc(cc1)C(C)(C)OCCO)c1[nH]c(=O)c2c(n1)ccc(c2)F'  --run_type repex  --resname 5n  --padding 1.0 --replicas 12 --storage_path $PWD/repex_test_ligand.nc --log_level 10