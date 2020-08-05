import importlib
from glob import glob
module_names = glob('./neuralphys/models/*rpin*.py')
for module_name in module_names:
    name = module_name.split('/')[-1].split('.')[0]
    if name == '__init__':
        continue
    importlib.import_module('neuralphys.models.' + name)
