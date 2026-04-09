import os
import glob
import common
from typing import Dict, Tuple, List


def load_data(data_dir: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    tree_start = {}
    tree_gt = {}

    for path in sorted(glob.glob(os.path.join(data_dir, f'*{common.POSTFIX_START}'))):
        tree_id = os.path.basename(path).replace(common.POSTFIX_START, '')
        gt_path = os.path.join(data_dir, f'{tree_id}{common.POSTFIX_GT}')
        with open(path) as f:
            tree_start[tree_id] = f.read().strip()
        with open(gt_path) as f:
            tree_gt[tree_id] = f.read().strip()

    return tree_start, tree_gt


def load_data_indices(data_dir: str, indices: List[int]) -> Tuple[Dict[int, str], Dict[int, str]]:
    all_start, all_gt = load_data(data_dir)
    tree_start = {idx: all_start[str(idx)] for idx in indices}
    tree_gt = {idx: all_gt[str(idx)] for idx in indices}
    return tree_start, tree_gt


def get_tree_indices(data_dir: str) -> List[int]:
    indices = []
    for path in sorted(glob.glob(os.path.join(data_dir, f'*{common.POSTFIX_START}'))):
        tree_id = os.path.basename(path).replace(common.POSTFIX_START, '')
        try:
            indices.append(int(tree_id))
        except ValueError:
            pass
    return sorted(indices)