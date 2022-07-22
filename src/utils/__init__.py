from pathlib import Path

from .arg_parser import script_init_common
from .config import Config

DATA_PATH = Path(__file__).parents[2].joinpath("data").absolute().as_posix() + "/"
DB_PATH = Path(__file__).parents[2].joinpath("cil.db").absolute().as_posix() + "/"
