from openff.units import unit
import numpy as np
from modelforge.curate.properties import (
    AtomicNumbers,
    PartialCharges,
    Positions,
    DipoleMomentPerSystem,
    DipoleMomentScalarPerSystem,
)


from modelforge.potential.potential import (
    load_inference_model_from_checkpoint,
    NeuralNetworkPotentialFactory,
)
from modelforge.curate.datasets.tmqm_openff_curation import tmQMOpenFFCuration
import torch


# load trained model from wandb
import wandb
import pandas as pd
from loguru import logger as log

# define a simple command line arg parser so we can set the name of the group and project name

from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument(
    "--group",
    type=str,
    default=None,
    help="Wandb group name that we will process, name needs: `n_configs_<integer>_` in it so we can grab # configs.",
)
arg_parser.add_argument(
    "--project", type=str, default="modelforge_nnp_training", help="Wandb project name."
)
arg_parser.add_argument(
    "--mode",
    type=str,
    default="GPU",
    help="Run on GPU or CPU, default GPU.",
    choices=["GPU", "CPU"],
)
arg_parser.add_argument(
    "--hdf5_file",
    type=str,
    default="/home/cri/mf_datasets/hdf5_files/tmqm_openff_tests/tmqm_openff_fixed_test_subset_sm_1_overlap_v1.3.hdf5",
    help="HDF5 file to test on.",
)
args = arg_parser.parse_args()

if args.group is not None:
    GROUP = args.group
if args.project is not None:
    PROJECT = args.project
if args.mode == "CPU":
    log.info("Forcing CPU mode.")
    torch.cuda.is_available = lambda: False
    device = "cpu"
else:
    if torch.cuda.is_available():
        log.info("CUDA is available, will use GPU.")
        device = "cuda"
    else:
        log.info("CUDA is not available.")
        raise ValueError("CUDA is not available, cannot run in GPU mode.")

if args.hdf5_file is not None:
    input_hdf5_file = args.hdf5_file
else:
    raise ValueError("No HDF5 file provided.")

# PROJECT = "modelforge_nnp_training"
# GROUP = "tmqm_openff_aimnet2_E_all"

log.info(f"Loading models from wandb project: {PROJECT}, group: {GROUP}")
# initialize the wandb api with the appropriate project and group
api = wandb.Api()
filters = {"group": GROUP}
runs = api.runs(path=PROJECT, filters=filters)

# loop over all the relevant runs and grab info about them
model_files = []

for run in runs:
    temp_dict = {}

    # assemble the run path used to grab the model artifact
    run_path = f"modelforge_nnps/{PROJECT}/model-{run.id}"

    # get the best artifact for the run
    artifact_path = f"{run_path}:best"

    # get the integer following 'n_configs' in the run.name
    # current format is "tmqm_openff_sm_1_n_configs_2_random_all_v1.3_ps125_ds44"
    # this should probably end up being a tag in the future
    n_configs = int(run.name.split("n_configs_")[1].split("_")[0])

    # get the artifact so we can get the specific version
    # We will populate a bunch of information into a dictionary for each model
    artifact = api.artifact(artifact_path)
    temp_dict["n_configs"] = n_configs
    temp_dict["run_path"] = run_path
    temp_dict["version"] = artifact.version
    temp_dict["created_by"] = run.name
    temp_dict["url"] = run.url
    model_files.append(temp_dict)

log.info("Found the following models:")
for mf in model_files:
    log.info(
        f"Model trained with {mf['n_configs']} configurations: {mf['created_by']} ({mf['url']})"
    )

# Store all the initialized potentials in a list
# It is faster to load all the potentials first and then run the tests than reloading the dataset for each potential
potentials = []

# rather than having to use enumerate, just initialize a counter and store it with each potential
# note, doing this because we will have multiple runs for eachc n_configs value that we want to average over later
# so we can't just setup a dict with n_configs as the key
counter = 0

# check if cuda is available and log that info to the screen


for model_file in model_files:
    # note when using VDW we cannot use jit compilation

    potential = NeuralNetworkPotentialFactory.load_from_wandb(
        run_path=model_file["run_path"],
        version=model_file["version"],
        jit=False,
    )

    potential.to(device=device)
    # potential.to(device="cuda" if torch.cuda.is_available() else "cpu")
    potentials.append((model_file["n_configs"], counter, potential))
    counter += 1


import h5py

# this is the "fixed" tmqm openff test set

from tqdm import tqdm
from dataclasses import dataclass
from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT
from modelforge.utils.prop import NNPInput


# load dataset and test the nnp
energy_ref = []
dipole_moment_ref = []
partial_charge_ref = []


molecule_names = []

# get a set of the n_configs values for later use
unique_n_configs = set()

# loop over the models and set up dictionaries with lists for the results
energy_diff = {}
energy_pred = {}
dipole_moment_diff = {}
dipole_moment_pred = {}
partial_charge_diff = {}
partial_charge_pred = {}
partial_charge_diff = {}
partial_charge_pred = {}

for pot in potentials:

    n_configs, counter, potential = pot
    name = f"{n_configs}_{counter}"

    energy_diff[name] = []
    energy_pred[name] = []
    dipole_moment_diff[name] = []
    dipole_moment_pred[name] = []
    partial_charge_diff[name] = []
    partial_charge_pred[name] = []

    unique_n_configs.add(n_configs)

with h5py.File(input_hdf5_file, "r") as f:
    keys = list(f.keys())

    for key in tqdm(keys):

        # grab all the data from the hdf5 file

        atomic_numbers = f[key]["atomic_numbers"][()]
        positions = f[key]["positions"][()]
        energy_key = "dft_total_energy_corrected"

        energy = f[key][energy_key][()]
        total_charge = f[key]["total_charge"][()]
        number_of_atoms = atomic_numbers.shape[0]
        n_configs = f[key]["n_configs"][()]
        spin_multiplicity = f[key]["per_system_spin_multiplicity"][()]
        dipole_moment = f[key]["scf_dipole"][()]
        partial_charge = f[key]["lowdin_partial_charges"][()]

        # find the appropriate potential based on n_configs
        for potential_n_configs, counter, potential in potentials:
            name = f"{potential_n_configs}_{counter}"
            for n_config in range(n_configs):
                # print(f"Processing config {n_config} of {n_configs}")

                # could create a helper function to convert a record to NNPInput based on the properties association dict in
                # the toml files.
                nnp_input = NNPInput(
                    atomic_numbers=torch.tensor(
                        atomic_numbers.squeeze(), dtype=torch.int32
                    ),
                    positions=torch.tensor(
                        positions[n_config].reshape(-1, 3), dtype=torch.float32
                    ),
                    per_system_total_charge=torch.tensor(
                        total_charge[n_config].reshape(-1, 1), dtype=torch.float32
                    ),
                    atomic_subsystem_indices=torch.zeros(
                        number_of_atoms, dtype=torch.int32
                    ),
                    per_system_spin_state=torch.tensor(
                        spin_multiplicity[n_config].reshape(-1, 1), dtype=torch.float32
                    ),
                )
                nnp_input.to_device(
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                molecule_names.append(key)
                output = potential(nnp_input)

                # test energy results
                energy_temp = (
                    output["per_system_energy"].cpu().detach().numpy().reshape(-1)[0]
                )

                energy_diff[name].append(
                    float((energy_temp - energy[n_config]).reshape(-1)[0])
                )
                energy_pred[name].append(float(energy_temp.reshape(-1)[0]))
                energy_ref.append(float(energy[n_config].reshape(-1)[0]))

                # test partial charge results
                partial_charge_temps = (
                    output["per_atom_charge"].cpu().detach().numpy().reshape(-1)
                )
                partial_charge_diff[name].append(
                    (partial_charge_temps - partial_charge[n_config]).reshape(-1)
                )
                partial_charge_pred[name].append(partial_charge_temps.reshape(-1))
                partial_charge_ref.append(partial_charge[n_config].reshape(-1))

                # test dipole moment results
                dc = tmQMOpenFFCuration("tmqm_openff")
                dipole_moment_temp = dc.compute_dipole_moment(
                    atomic_numbers=AtomicNumbers(
                        value=nnp_input.atomic_numbers.cpu()
                        .detach()
                        .numpy()
                        .reshape(-1, 1)
                    ),
                    partial_charges=PartialCharges(
                        value=partial_charge_temps.reshape(1, -1, 1),
                        units=unit.Unit(f[key]["lowdin_partial_charges"].attrs["u"]),
                    ),
                    positions=Positions(
                        value=nnp_input.positions.cpu()
                        .detach()
                        .numpy()
                        .reshape(1, -1, 3),
                        units=unit.Unit(f[key]["positions"].attrs["u"]),
                    ),
                ).value

                dipole_moment_diff[name].append(
                    list(dipole_moment_temp - dipole_moment[n_config])
                )
                dipole_moment_pred[name].append(list(dipole_moment_temp))
                dipole_moment_ref.append(list(dipole_moment[n_config]))

energy_mae_all = {}
partial_charge_mae_all = {}
dipole_moment_mae_all = {}
energy_rmse_all = {}
partial_charge_rmse_all = {}
dipole_moment_rmse_all = {}

for i in unique_n_configs:
    energy_mae_all[i] = []
    partial_charge_mae_all[i] = []
    dipole_moment_mae_all[i] = []
    energy_rmse_all[i] = []
    partial_charge_rmse_all[i] = []
    dipole_moment_rmse_all[i] = []

results_dict = {}
for name in energy_diff.keys():
    n_configs = int(name.split("_")[0])
    # calculate MAE for the system and save
    energy_mae = np.sum(abs(np.array(energy_diff[name]))) / len(energy_diff[name])
    energy_rmse = np.sqrt(
        np.sum((np.array(energy_diff[name])) ** 2) / len(energy_diff[name])
    )

    partial_charge_diff_temp = np.concatenate(partial_charge_diff[name])
    partial_charge_mae = np.sum(abs(np.array(partial_charge_diff_temp))) / len(
        partial_charge_diff_temp
    )
    partial_charge_rmse = np.sqrt(
        np.sum((np.array(partial_charge_diff_temp)) ** 2)
        / len(partial_charge_diff_temp)
    )
    dipole_moment_diff_temp = np.array(dipole_moment_diff[name]).reshape(-1)

    dipole_moment_mae = np.sum(abs(np.array(dipole_moment_diff_temp))) / len(
        dipole_moment_diff_temp
    )

    dipole_moment_rmse = np.sqrt(
        np.sum((np.array(dipole_moment_diff_temp)) ** 2) / len(dipole_moment_diff_temp)
    )

    energy_mae_all[n_configs].append(energy_mae)
    energy_rmse_all[n_configs].append(energy_rmse)
    partial_charge_mae_all[n_configs].append(partial_charge_mae)
    partial_charge_rmse_all[n_configs].append(partial_charge_rmse)
    dipole_moment_mae_all[n_configs].append(dipole_moment_mae)
    dipole_moment_rmse_all[n_configs].append(dipole_moment_rmse)

    print("=========================================")
    print(f"Results for model trained with {name.split('_')[0]} configurations:")

    print(f"Energy MAE: {energy_mae} kJ/mol")
    print(f"Energy RMSE: {energy_rmse} kJ/mol")
    print(f"Partial Charge MAE: {partial_charge_mae} e")
    print(f"Partial Charge RMSE: {partial_charge_rmse} e")
    print(f"Dipole Moment MAE: {dipole_moment_mae} e*nm")
    print(f"Dipole Moment RMSE: {dipole_moment_rmse} e*nm")

    results_dict[name] = {
        "n_configs": int(name.split("_")[0]),
        "energy": {
            "mae": energy_mae,
            "rmse": energy_rmse,
            "units": "kilojoule_per_mole",
        },
        "partial_charge": {
            "mae": partial_charge_mae,
            "rmse": partial_charge_rmse,
            "units": "e",
        },
        "dipole_moment": {
            "mae": dipole_moment_mae,
            "rmse": dipole_moment_rmse,
            "units": "e*nm",
        },
    }

# get mean and stddev for each n_configs value
mean_std_results = {}
for n_configs in unique_n_configs:
    energy_mae_mean = np.mean(energy_mae_all[n_configs])
    energy_mae_std = np.std(energy_mae_all[n_configs])
    energy_rmse_mean = np.mean(energy_rmse_all[n_configs])
    energy_rmse_std = np.std(energy_rmse_all[n_configs])
    partial_charge_mae_mean = np.mean(partial_charge_mae_all[n_configs])
    partial_charge_mae_std = np.std(partial_charge_mae_all[n_configs])
    partial_charge_rmse_mean = np.mean(partial_charge_rmse_all[n_configs])
    partial_charge_rmse_std = np.std(partial_charge_rmse_all[n_configs])
    dipole_moment_mae_mean = np.mean(dipole_moment_mae_all[n_configs])
    dipole_moment_mae_std = np.std(dipole_moment_mae_all[n_configs])
    dipole_moment_rmse_mean = np.mean(dipole_moment_rmse_all[n_configs])
    dipole_moment_rmse_std = np.std(dipole_moment_rmse_all[n_configs])

    print("=========================================")
    print(f"Average Results for models trained with {n_configs} configurations:")
    print(f"Energy MAE: {energy_mae_mean} ± {energy_mae_std} kJ/mol")
    print(f"Energy RMSE: {energy_rmse_mean} ± {energy_rmse_std} kJ/mol")
    print(f"Partial Charge MAE: {partial_charge_mae_mean} ± {partial_charge_mae_std} e")
    print(
        f"Partial Charge RMSE: {partial_charge_rmse_mean} ± {partial_charge_rmse_std} e"
    )
    print(f"Dipole Moment MAE: {dipole_moment_mae_mean} ± {dipole_moment_mae_std} e*nm")
    print(
        f"Dipole Moment RMSE: {dipole_moment_rmse_mean} ± {dipole_moment_rmse_std} e*nm"
    )

mean_std_results[n_configs] = {
    "energy": {
        "mae": {
            "mean": energy_mae_mean,
            "stddev": energy_mae_std,
            "units": "kilojoule_per_mole",
        },
        "rmse": {
            "mean": energy_rmse_mean,
            "stddev": energy_rmse_std,
            "units": "kilojoule_per_mole",
        },
    },
    "partial_charge": {
        "mae": {
            "mean": partial_charge_mae_mean,
            "stddev": partial_charge_mae_std,
            "units": "e",
        },
        "rmse": {
            "mean": partial_charge_rmse_mean,
            "stddev": partial_charge_rmse_std,
            "units": "e",
        },
    },
    "dipole_moment": {
        "mae": {
            "mean": dipole_moment_mae_mean,
            "stddev": dipole_moment_mae_std,
            "units": "e*nm",
        },
        "rmse": {
            "mean": dipole_moment_rmse_mean,
            "stddev": dipole_moment_rmse_std,
            "units": "e*nm",
        },
    },
}

import yaml

with open(f"results_{PROJECT}_{GROUP}.yaml", "w") as f_out:
    info = {
        "project": PROJECT,
        "group": GROUP,
        "input_hdf5_file": input_hdf5_file,
    }
    yaml.dump(info, f_out)
    for model in model_files:
        yaml.dump(model, f_out)
    yaml.dump(results_dict, f_out)
    yaml.dump(mean_std_results, f_out)
