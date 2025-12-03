#!/usr/bin/env python3

import sys
import os

codebase_dir = "/home/smanzoor/welch/get_multimodel"

# Add project root to Python path so Hydra can find the model targets
PROJECT_ROOT = codebase_dir
if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        os.chdir(PROJECT_ROOT)

from get_model.config.config import export_config, load_config, load_config_from_yaml

