# Setup script for Euler, but can work in unix based systems as well
course="cil"

module load gcc/6.3.0 python/3.8.5 eth_proxy

if [ ! -d "${SCRATCH}/${course}_env" ]; then
    echo "Creating virtual env at ${SCRATCH}/${course}_env"
    python -m venv "${SCRATCH}/${course}_env"
    source "${SCRATCH}/${course}_env/bin/activate"
    python -m pip install --upgrade pip
fi

source "${SCRATCH}/${course}_env/bin/activate"
pip install -r requirements.txt


cd "${HOME}/${course}"

echo "Now, let's complete wandb login."
if [[ -z "${WANDB_ENTITY}" || -z "${WANDB_API_KEY}" ]]; then
    read -p "Wandb entries seem to be not properly setup. Do you want me to setup wandb for you? [yes/no] " wandb_requested
    if [ "$wandb_requested" = "yes" ]; then
        echo "Wandb installation requested"
        echo "Wandb entity name not defined"
        read -p "Enter your Wandb username: " wandb_username
        echo "export WANDB_ENTITY=\"$wandb_username\"" >> ~/.bashrc
        read -s -p "Go to https://wandb.ai/authorize and copy paste your APIKEY: " wandb_api_key
        echo "export WANDB_API_KEY=\"$wandb_api_key\"" >> ~/.bashrc
        source ~/.bashrc
        echo "Wandb installation complete"
    elif [ "$wandb_requested" = "no" ]; then
        echo "Wandb setup not requested. For convenience make sure your api key and username are in ~/.bashrc"
    else
        echo "Invalid command"
    fi
else
    echo "Wandb is already setup"
fi

echo "Attempting wandb login"
wandb login

echo "Attempting bash kernel install"
python -m bash_kernel.install
