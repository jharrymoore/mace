import pytest
from mace.tools.mixed_system import MixedSystem


data = [
    (
        "tnks_complex.pdb",
        "c1(ccc(cc1)C(C)(C)OCCO)c1[nH]c(=O)c2c(n1)ccc(c2)F",
        "UNK", -410543.3272970009
    ),
    ("5n.sdf", "c1(ccc(cc1)C(C)(C)OCCO)c1[nH]c(=O)c2c(n1)ccc(c2)F", "5n", -42584.749481789506),
]


mixed_system_kwargs = {"forcefields":["amber/protein.ff14SB.xml", "amber/tip3p_standard.xml"],
"temperature": 298.15,
"model_path":"MACE_SPICE.model",
"padding": 1.5,
"ionicStrength": 0.15,
"nonbondedCutoff": 1.0,
"potential": "mace",
"repex_storage_path": "test.nc"
}

STEPS=10

@pytest.mark.parametrize("file, smiles, resname, output", data)
def test_run_plain_md(file, smiles, resname, output):
    mixedSystem = MixedSystem(file=file, smiles=smiles, resname=resname, **mixed_system_kwargs)
    potential_energy = mixedSystem.run_mixed_md(steps=STEPS, interval=1, output_file="test_output.pdb")
    print("Potential ENERGY:", potential_energy)
    # TODO: fix the seed to make simulations 
    assert abs(potential_energy - (output)) < 1000

@pytest.mark.parametrize("file, smiles, resname, output",data)
def test_run_neq_switching(file, smiles, resname, output):
    mixedSystem = MixedSystem(file=file, smiles=smiles, resname=resname, **mixed_system_kwargs)
    protocol_work = mixedSystem.run_neq_switching(steps=STEPS, interval=1)
    assert protocol_work == -41575
    
# @pytest.mark.parametrize("file, smiles, resname", files)
# def test_run_repex(file, smiles, resname):
#     mixedSystem = MixedSystem(file=file, smiles=smiles, resname=resname, **mixed_system_kwargs)
#     mixedSystem.run_replex_equilibrium_fep(replicas=2)