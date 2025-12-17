import wandb
import pandas as pd


PROJECT = "modelforge_nnp_training"
GROUP_NAME = "tmqm_openff_aimnet2_E_all"

api = wandb.Api()

filters = {"group": GROUP_NAME}

runs = api.runs(path=PROJECT, filters=filters)

for run in runs:

    run_path = f"modelforge_nnps/{PROJECT}/model-{run.id}"

    print(run.url)
    artifact_path = f"{run_path}:best"
    print(artifact_path)
    print(run.name)
    # get the integer following 'n_configs' in the run.name
    n_configs = int(run.name.split("n_configs_")[1].split("_")[0])

    artifact = api.artifact(artifact_path)
    print(artifact.name, artifact.type, artifact.version)
    print("n_configs:", n_configs)
