# Setup script for Euler, but can work in unix based systems as well
course="meowtrix-purrdiction"


Help()
{
   # Display Help
   echo "This script will setup your workspace in Euler for the ${course} Project."
   echo
   echo "Syntax: scriptTemplate [-w|h|l|c]"
   echo "options:"
   echo "l     Setup in local machine, default: disabled"
   echo "w     Complete wandb setup, default: disabled."
   echo "c     Use anaconda, default: disabled. Only available in local machine."
   echo "h     Print this Help."
   echo
}

# Global parameters
SETUP_WANDB="false"
LOCAL_MACHINE="false"
USE_CONDA="false"

while getopts ":hwlc" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      w) # Use wandb
         SETUP_WANDB="true";;
      l) # setup in local machine
         LOCAL_MACHINE="true";;
      c) # anaconda
         USE_CONDA="true";;
      \?) # Invalid option
          echo "Error: Invalid option"
          exit;;
   esac
done

wandb_login() {
    echo "Now, let's complete wandb login."
    if [[ "$SETUP_WANDB" == "true" ]]; then
        pip install wandb
        echo "Attempting wandb setup"
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
    else 
        echo "Wandb setup disabled, run with flag -w to enable wandb setup."
    fi
}


if [[ "$LOCAL_MACHINE" == "false"  &&  "$USE_CONDA" == "true" ]]; then
    echo; echo "Error: Use conda available only in local machine, must pass arguments as -lc"
    exit
fi

if [[ "$LOCAL_MACHINE" == "false" ]]; then
    echo; echo "Setting up in euler"
    module load gcc/8.2.0 cuda/11.3.1 python_gpu/3.8.5 eth_proxy
else
    echo; echo "Setting up in local machine"

    if [[ "$USE_CONDA" == "true" ]]; then
        conda create -n $course -y python=3.8;
        conda activate $course
        python setup.py install
    else
        python -m venv $course
        python -m pip install --upgrade pip
        source $course/bin/activate
        python setup.py install
    fi
    if [[ ! -f cil.db ]]; then
        echo "Creating database file"
        echo > cil.db
    fi
    wandb_login
    echo "Setup completed"
    exit
fi

# continue qith Euler setup
if [ ! -d "${SCRATCH}/${course}_env" ]; then
    echo "Creating virtual env at ${SCRATCH}/${course}_env"
    python -m venv "${SCRATCH}/${course}_env"
fi
source "${SCRATCH}/${course}_env/bin/activate"
python -m pip install --upgrade pip

if [ ! -d "${HOME}/${course}" ]; then
    echo; echo "Repository not available, please check configuration."
    exit
fi 


if [[ ! -f cil.db ]]; then
    echo "Creating database file"
    echo > cil.db
fi
cd "${HOME}/${course}"
python setup.py install

wandb_login
echo "Setup completed"