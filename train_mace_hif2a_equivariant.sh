#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -A csanyi-SL3-GPU


source /home/jhm72/miniconda3/bin/activate /home/jhm72/miniconda3/envs/mace_openmm
which python
python scripts/run_train.py     --name="MACE_model_equivariant_single_precision"     --train_file=../ace-training/hif2a/hif2a_md_configs_xtb.xyz     --valid_fraction=0.1     --test_file=../BOTNet-datasets/dataset_3BPA/test_300K.xyz     --config_type_weights='{"Default":1.0}'       --model="MACE"     --hidden_irreps='128x0e + 128x1o'     --r_max=5.0     --batch_size=128     --max_num_epochs=1500     --swa     --start_swa=1200     --ema     --ema_decay=0.99     --amsgrad          --device=cuda --restart_latest --default_dtype=float32
