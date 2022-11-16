from pathlib import Path
from yaml.loader import SafeLoader

import os
import yaml

data_path = Path(os.path.dirname(os.path.realpath(__file__))).parent
config_path = str(data_path.joinpath("config.yaml"))

with open(config_path) as f:
	config = yaml.load(f, Loader=SafeLoader)
