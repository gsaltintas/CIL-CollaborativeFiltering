# Bayesian Optimization on Kernel-Trained Auto-Encoders for Collaborative Filtering
<center>
<h2>ETH Zürich Computational Intelligence Lab </h2>
<h3><em>Team Meowtrix Purrdiction </em> &#128049; </h3>
</center>

## Paper &#128394;
This is the implementation accompanying *Bayesian Optimization on Kernel-Trained Auto-Encoders for Collaborative Filtering*. The Overleaf link for our paper can be found [here](https://www.overleaf.com/project/62431cee5ebeafd813c104c3).


## Installation &#128295;
- We provide a setup script that can set up the project with Anaconda or Virtual Environment on Euler or your local machine, but interested users can set up on their own. We provide a `requirements.txt` file.
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
The setup script will create a database file in the main directory by default, which Optuna uses.

<span style="background-color:#ff9999; color:black"> <b>Macbook users</b></span>: Please be aware that matplotlib installation may be problematic for M1 Chip, you may comment it out in requirements.txt.

## Wandb &#128239;&#128214;
- We offer [Wandb](https://docs.wandb.ai/) logging for experiment tracking. If not used, GLocal-K experiments are logged to local Tensorboard, and baselines experiments are logged as written output.
- [Installation script](#installation) handles wandb login and setup when `-w` flag is passed. If you haven't exported your wandb credentials to your environment, you are advised to reply 'yes' to wandb login prompt. 
- **Wandb during experiments**: By default, wandb logging is disabled. To enable, pass `--use-wandb=True` to the training command. Note that wandb logging is possible for BO on baselines, GLocal-K training and BO on GLocal-K.

## Bayesian Optimization using Optuna Hyperparameter Search Framework &#128201;
- We use [Optuna](https://optuna.org/) as our Bayesian Hyperparameter Search Framework. 
- **Pruning**: We don't use any pruning algorithms for baselines but to enable median pruning for GLocal-K optuna implementation pass `--enable-pruning=True` to the training script.
- **Storage**: We run our experiments with Optuna storage to be able to run sequential and parallel trials. However, for third-party users' convenience we disable storage option, pass `--use-storage=True` to training script to enable it. To use Optuna with storage you need to create a database file in the project folder. This is also handled by the setup script, if for some reason setup script fails, run `echo >> cil.db`. 
- **Sampler**: We use the default TPESampler.
- **Reproducibility**: Please note that we refrain from using a deterministic sampler for Optuna experiments as the documentation suggests (see [here](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results)).
- BO is implemented for SVD, SVD++, and GLocal-K only.


## Training & Reproducibility &#128274;&#128640;
We provide a single entry point for coherence and ease of use `src/train.py`. This script accepts command line arguments (please see `src/utils/config.py` for the complete list of parameters or examine corresponding sections for relevant arguments).

- **Euler**:
    - Modify `train.sh` accordingly and run `bsub < scripts/train.sh` on Euler to submit an LSF job.
- **Local Machine**:
    - Sample training commands for each implemented algorithm can be found in their corresponding section.


## Baselines &#129352;
Following baselines are taken from the [Surprise library](surprise.readthedocs.io): SVD, SVD++, NMF, KNN baseline, Co-clustering and SlopeOne.
- **Training**: Each baseline have its own set of parameters. We list the training commands for each algorithm with our best-performing submissions:
    - SVD 
    ```bash
    python -m train  --experiment-type=train --use-wandb=False --algo=svd --n-factors=66 --biased=True --lr-all=0.014034216851890047 --reg-all=0.0904990847308604 --init-mean=0.9995295507254714 --init-std-dev=0.3654283107617773 --seed=42
    ```
    - SVD++
    ```bash
    python -m train  --experiment-type=train --use-wandb=False --algo=svdpp --n-factors=200 --n-epochs=60 --lr-all=0.005 --reg-all=0.1 --seed=42
    ```
    - NMF 
    ```bash
    python -m train  --experiment-type=train --use-wandb=False --algo=nmf --n-factors=20 --biased=False --reg-pu=0.1 --reg-qi=0.1 --seed=42
    ```
    - KNN baseline
    ```bash
    python -m train  --experiment-type=train --use-wandb=False --algo=knn --k=40     --sim-options-name=pearson_baseline --sim-options-shrinkage=1   --bsl-options-name=als
    ```
    - Co-clustering
    ```bash
    python -m train  --experiment-type=train --use-wandb=False --algo=coclustering  --n-cltr-u=4 --n-cltr-i=10 --n-epochs=10 --seed=config.seed
    ```
    - SlopeOne
    ```bash
    python -m train  --experiment-type=train --use-wandb=False --algo=s1 
    ```

    | Algorithm | Public Leaderboard Score | Optimization Method |
    | ---  |  ---------------------   | ----- |
    | SVD  | 0.99760                    | BO |
    | SVD++| 0.98955                  | GridSearch |
    | NMF  | 1.00307                   | GridSearch |
    | KNN Baseline | 1.00296            | GridSearch |

- **Hyperparameter Search**: This script can run GridSearchCV and Bayesian Optimization on the baselines. Please note that for simplicity and ease of use of the script, we don't support passing the hyperparameter search space from the command line. Hyperparameter search spaces for baselines are listed in the [below table](#table-summarizing-hyper-parameter-space-for-baselines). 
    - **Grid Search**: Grid search with 5-fold cross-validation is taken from the Surprise Library. Relevant arguments that can be passed to the training script are as follows:
    ```bash
    --refit [True, False] --algo [svd, svdpp, coclustering, nmf, knn] --n-jobs (int)
    ```
    - **Bayesian Optimization with Optuna**: As outlined in [BO section](#bayesian-optimization-using-optuna-hyperparameter-search-framework), we use Optuna to further optimize the two best-performing baselines (SVD, SVD++). We wrote wrapper classes on the corresponding Suprise baselines for compatibility. Relevant arguments that can be passed to the training script are as follows:
    ```bash
    --refit [True, False] --algo [svd, svdpp] 
    --use-wandb  [True, False] --use-storage [True, False] --study-name "" 
    --n-trials (int) --timeout (int) --n_jobs (int)
    --seed (int) --train-size (float: fraction of training data in the whole dataset) --verbose [0, 1, 2]
    ``` 
    Then, we performed Bayesian Optimization on the best-performing two baselines (SVD, SVD++) with Optuna. 

### Table summarizing hyper-parameter space for baselines

\*: default value used, +: Grid search values, $ : BO search range

Model | n\_factors | biased | lr | regularizer | n\_epochs | $\mu_0$ | $\sigma_0$ |
| -- | -- | -- | --| -- | -- | -- | --|
SVD + |  50, 100, 200 | True, False | 0.005, 0.05 | 0.02, 0.1 | 20 \* | 0 \* | 0.1 \* 
| SVD $ | [50, 200] | True, False | [1e-5, 0.1] | [0.001, 0.1] | 20 \* | [0, 3] | [0.1, 1]  
SVD++ + | 150, 200 |x| 0.005, 0.001 | 0.1, 0.2 | 30, 60 | 0 \* | 0.1 \*  
| SVD++ $ | [40, 200] | x | [1e-5, 0.1] | [0.001, 0.1] | [40, 70] | [0, 3] | [0.1, 1]
|NMF + |    15, 20 | True, False | 0.005 \* | $p_u$: 0.06, 0.01, $q_i$: 0.06, 0.01 | 50 \* | x | x 

| Model | k | sim | shrinkage | bsl|
| -| -| -| -| -|
| KNN + | (10, 25, 40) |pearson_baseline|(0,1)| (als, sgd) |

<!-- 
SVD: n_factors:(50, 100, 200), biased: (True, False), lr_all:(0.005, 0.05), reg_all:(0.02, 0.1)
SVD++ 
    n_factors:(150, 200), n_epochs:(30, 60), lr_all:(0.005, 0.001), reg_all:(0.1, 0.2)
NMF:
    n_factors:(15, 20), biased: (True, False), reg_pu: (0.06, 0.1), reg_qi: (0.06, 0.1),
KNN 
    k:(10, 25, 40), sim_options: {name: (pearson_baseline), shrinkage: (0,1)}, bsl_options: {name: (als, sgd)} -->

## GLocal-K &#129351;
We implemented GLocal-K^ in [Pytorch Lightning](https://www.pytorchlightning.ai/) and [Pytorch](https://pytorch.org/) with RBF Kernel drawing inspiration from the official [implementation](https://github.com/usydnlp/Glocal_K).

Please note that depending on the hyperparameter choice, GLocal-K may have 6 to 15 million trainable parameters. We provide a training script under `scripts/train_glocal.sh`.

> ^ S. C. Han, T. Lim, S. Long, B. Burgstaller, and J. Poon, “GLocal-K: Global and Local Kernels for Recommender Systems,” in Proceedings of the 30th ACM International Conference on Information & Knowledge Management, Oct. 2021, pp. 3063–3067. doi: 10.1145/3459637.3482112.

### Optuna

### Training
```bash
python -m train --use-wandb False --experiment-dir experiments \
     --experiment-type train --algo glocal_k \
    --NUM-WORKERS 1  --n-hid 1000  --n-dim 10  --n-layers 2  --gk-size 5 \
    --lambda-2 20  --lambda-s 0.006  --iter-p 5  --iter-f 5 \
     --epoch-p 30 --epoch-f 60 --dot-scale 1 --seed 42 \
    --lr-fine 1 --lr-pre 0.1
```
