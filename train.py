import sys
import yaml
from solver import Solver


yml_path = sys.argv[1]
with open(yml_path) as f:
    config = yaml.load(f)

solver = Solver(**(config['model_params']))
solver.fit(**(config['fit_params']))
