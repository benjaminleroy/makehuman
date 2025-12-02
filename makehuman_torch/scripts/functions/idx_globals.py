import os
import sys
from io_funcs import load_vocabulary

# paths
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'index')
mod_path = os.path.join(data_path, 'modifiers.txt')
dep_path = os.path.join(data_path, 'dependencies.txt')
target_path = os.path.join(data_path, 'targets.txt')


# global dictionaries
MOD_NAMES, MOD_NAME_TO_ID = load_vocabulary(mod_path)
DEP_NAMES, DEP_NAME_TO_ID = load_vocabulary(dep_path)
TARGET_NAMES, TARGET_NAME_TO_ID = load_vocabulary(target_path)

