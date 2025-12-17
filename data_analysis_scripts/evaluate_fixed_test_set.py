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

model_files = [
    {
        "n_configs": 1,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-ij4a3yis",
        "version": "v0",
    },
    {
        "n_configs": 1,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-e28ywlou",
        "version": "v1",
    },
    {
        "n_configs": 1,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-xaqylfqm",
        "version": "v0",
    },
    {
        "n_configs": 2,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-lhaeu65h",
        "version": "v0",
    },
    {
        "n_configs": 2,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-r4q0pplj",
        "version": "v1",
    },
    {
        "n_configs": 2,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-8egb5ye9",
        "version": "v1",
    },
    {
        "n_configs": 5,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-ass55gbd",
        "version": "v0",
    },
    {
        "n_configs": 5,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-443yq68r",
        "version": "v1",
    },
    {
        "n_configs": 5,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-v8luywqb",
        "version": "v0",
    },
    {
        "n_configs": 10,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-3y9q68ac",
        "version": "v1",
    },
    {
        "n_configs": 10,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-5syof3vo",
        "version": "v1",
    },
    {
        "n_configs": 10,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-ccpmdd94",
        "version": "v0",
    },
    {
        "n_configs": 20,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-d3majsan",
        "version": "v1",
    },
    {
        "n_configs": 20,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-j0vptf83",
        "version": "v0",
    },
    {
        "n_configs": 20,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-nf9q1nkf",
        "version": "v0",
    },
    {
        "n_configs": 30,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-00yytjb4",
        "version": "v1",
    },
    {
        "n_configs": 30,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-rso8k9ob",
        "version": "v1",
    },
    {
        "n_configs": 30,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-k8ogvygd",
        "version": "v1",
    },
    {
        "n_configs": 40,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-5xub46ma",
        "version": "v1",
    },
    {
        "n_configs": 40,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-t4wj6634",
        "version": "v1",
    },
    {
        "n_configs": 40,
        "run_path": "modelforge_nnps/modelforge_nnp_training/model-5cywz2ck",
        "version": "v1",
    },
]

potentials = []

counter = 0
for model_file in model_files:
    potential = NeuralNetworkPotentialFactory.load_from_wandb(
        run_path=model_file["run_path"],
        version=model_file["version"],
        jit=False,
    )
    potential.to(device="cuda" if torch.cuda.is_available() else "cpu")
    potentials.append((model_file["n_configs"], counter, potential))
    counter += 1
# potential = NeuralNetworkPotentialFactory.load_from_wandb(
#     run_path="modelforge_nnps/modelforge_nnp_training/model-ij4a3yis",
#     version="v0",
#     jit=False,
# )


# potential.to(device="cuda" if torch.cuda.is_available() else "cpu")


import h5py

# this is the "fixed" tmqm openff test set
input_filenames = {
    # 1: "/home/cri/mf_datasets/hdf5_files/qm9_tmqm_openff_tests/tmqm_openff_dataset_v1.2.hdf5",
    1: "/home/cri/mf_datasets/hdf5_files/tmqm_openff_tests/tmqm_openff_fixed_test_subset_sm_1_overlap_v1.3.hdf5",
    # 3: "/home/cri/mf_datasets/hdf5_files/qm9_tmqm_openff_tests/tmqm_openff_fixed_test_subset_sm_3_v1.2.hdf5",
    # 5: "/home/cri/mf_datasets/hdf5_files/qm9_tmqm_openff_tests/tmqm_openff_fixed_test_subset_sm_5_v1.2.hdf5",
}

from tqdm import tqdm
from dataclasses import dataclass
from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT
from modelforge.utils.prop import NNPInput


# load dataset and test the nnp
energy_ref = []
dipole_moment_ref = []
partial_charge_ref = []


molecule_names = []


with h5py.File(input_filenames[1], "r") as f:
    keys = list(f.keys())

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

    for key in tqdm(keys):

        # grab all the data from the hdf5 file
        # we could certainly write some helper functions to do this better
        # e.g., loading this into a SourceDataset class instance and accessing
        # each record by key
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

    for name in energy_diff.keys():
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
            np.sum((np.array(dipole_moment_diff_temp)) ** 2)
            / len(dipole_moment_diff_temp)
        )

        print("=========================================")
        print(f"Results for model trained with {name.split('_')[0]} configurations:")

        print(f"Energy MAE: {energy_mae} kJ/mol")
        print(f"Energy RMSE: {energy_rmse} kJ/mol")
        print(f"Partial Charge MAE: {partial_charge_mae} e")
        print(f"Partial Charge RMSE: {partial_charge_rmse} e")
        print(f"Dipole Moment MAE: {dipole_moment_mae} Debye")
        print(f"Dipole Moment RMSE: {dipole_moment_rmse} Debye")

        # save this to a yaml file
        import yaml

        results_dict = {
            name: {
                "n_configs": int(name.split("_")[0]),
                "energy_mae_kj_per_mol": energy_mae,
                "energy_rmse_kj_per_mol": energy_rmse,
                "partial_charge_mae_e": partial_charge_mae,
                "partial_charge_rmse_e": partial_charge_rmse,
                "dipole_moment_mae_debye": dipole_moment_mae,
                "dipole_moment_rmse_debye": dipole_moment_rmse,
            }
        }
        with open(f"fixed_test_set_results_random_overlap_{name}.yaml", "w") as f_out:
            yaml.dump(results_dict, f_out)
#
