# ETH Zürich Computational Intelligence Lab - Semester Project
Team Meowtrix Purrdiction

## Installation
- Clone the repository to your home directory.
- For Euler setup:
    - Run `cd $HOME/meowtrix-purrdiction && bash setup.sh` which will handle pretty much everything (virtual env creation and installation of necessary packages). Setup script creates the virtual environment in the `$SCRATCH` folder in Euler not to occupy extra memory of your quota. Make sure that you run the setup script again every two weeks. 
- For Local setup:
    - Anaconda:
    ```bash
    bash setup.sh -lc
    ```
    - Virtualenv
    ```bash
    bash setup.sh -l
    ```
- Pass `-w` flag to setup wandb.
- By default, the setup script will create a database file in the main directory, which is used by Optuna.

## Wandb
Installation script [](#installation) handles wandb login and setup when `-w` flag is passed. If you haven't exported your wandb credentials to your environment, you are advised to reply 'yes' to wandb login prompt. 

By default, wandb logging is disabled. To enable, pass `--use-wandb=True` in the training command.

## Report
Overleaf link for the report [here](https://www.overleaf.com/project/62431cee5ebeafd813c104c3).

# Optuna
To use Optuna with storage you need to create a database file in the project folder. This is also handled by the setup script, if for some reason setup script fails, run `echo >> cil.db`. 

## Baselines
Following baselines are taken from  

## Training
- Euler:
    - Modify `train.sh` accordingly and run `bsub < scripts/train.sh` on Euler to submit a SLURM job.


# Reproducibility
