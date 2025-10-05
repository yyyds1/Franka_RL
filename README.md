# RL for Dexhand & Arm Manipulation

## Overview

This repository provides RL framework for Dexhand and robotics arm manipulation tasks. It allows you to train your agent to complete diverse Human-Object Interection(HOI) tasks by imitating demo trajectories.

## Installation

- Clone or copy this repository:
    ```bash
    git clone https://github.com/yyyds1/Franka_RL.git
    ```

- It's recommended to create a new python environment for the repository(e.g. Conda environment)
:
    ```bash
    conda create -n frankarl python=3.11
    conda activate frankarl
    ```

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Using a python interpreter that has Isaac Lab installed, install the dependences and the library in editable mode using:

    ```bash
    pip install --no-index --no-cache-dir pytorch3d==0.7.3 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html
    pip install -r requirements.txt
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/Franka_RL
    ```

- Verify that the extension is correctly installed by listing the available tasks (Note: If the task name changes, it may be necessary to update the search pattern `"Template-"` in the `scripts/list_envs.py` file so that it can be listed.):
    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/list_envs.py
    ```

## Datasets

## Run a Task

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Franka_RL"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```