import os
import subprocess
import tempfile
import torch
from concurrent.futures import ProcessPoolExecutor

def call_raxml(args):
    newick_str, alignment_file, model = args
    with tempfile.TemporaryDirectory() as tmpdir:
        tree_file = os.path.join(tmpdir, 'tree.nwk')
        prefix    = os.path.join(tmpdir, 'eval')
        with open(tree_file, 'w') as f:
            f.write(newick_str)
        
        cmd = ['/raid/home/hatang/miniconda3/envs/bmeprl/bin/raxml-ng', '--evaluate',
               '--msa',     alignment_file,
               '--model',   model,
               '--tree',    tree_file,
               '--prefix',  prefix,
               '--threads', '1',
               '--redo']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f'raxml-ng failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}')
        log_file = prefix + '.raxml.log'
        with open(log_file) as f:
            for line in f:
                if 'Final LogLikelihood:' in line:
                    return float(line.split()[-1])
        raise RuntimeError(f'Cannot parse raxml-ng output:\n{result.stdout}')
