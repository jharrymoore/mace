from argparse import ArgumentParser
from mace.tools.mixed_system import MixedSystem
from mace import tools
import logging

def main():
    parser = ArgumentParser()

    parser.add_argument("--file", "-f", type=str)
    parser.add_argument("--smiles", type=str, help="smiles for the small molecule", default=None)
    parser.add_argument("--run_type", choices=["md", "repex", "neq"], type=str, default="md")
    parser.add_argument("--steps", "-s", type=int, default=10000)
    parser.add_argument("--padding", "-p", default=1.2, type=float)
    parser.add_argument("--nonbondedCutoff", "-c", default=1.0, type=float)
    parser.add_argument("--ionicStrength", "-i", default=0.15, type=float)
    parser.add_argument("--potential", default="mace", type=str)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument("--output_file", "-o", type=str, default="output.pdb", help="output file for the pdb reporter")
    parser.add_argument("--log_level", default=logging.INFO, type=int)
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument(
        "--forcefields",
        type=list,
        default=["amber/protein.ff14SB.xml", "amber/tip3p_standard.xml"],
    )
    parser.add_argument(
        "--interval", help="steps between saved frames", type=int, default=100
    )
    parser.add_argument(
        "--resname",
        "-r",
        help="name of the ligand residue in pdb file",
        default="UNK",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        "-m",
        help="path to the mace model",
        default="tests/test_openmm/MACE_SPICE.model",
    )
    args = parser.parse_args()



    tools.setup_logger(level=args.log_level, directory=args.log_dir)

    mixed_system = MixedSystem(
        file=args.file,
        smiles=args.smiles,
        model_path=args.model_path,
        forcefields=args.forcefields,
        resname=args.resname,
        ionicStrength=args.ionicStrength,
        nonbondedCutoff=args.nonbondedCutoff,
        potential=args.potential,
        padding=args.padding,
        temperature=args.temperature,
    )
    if args.run_type == "md":
        mixed_system.run_mixed_md(args.steps, args.interval, args.output_file)
    elif args.run_type == "repex":
        mixed_system.run_replex_equilibrium_fep(args.replicas)
    elif args.run_type == "neq":
        mixed_system.run_neq_switching(args.steps, args.interval)
    else:
        raise ValueError(f"run_type {args.run_type} was not recognised")


if __name__ == "__main__":
    main()
