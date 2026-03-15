RL framework's possible options:
- Loss: Q learning, REINFORCE
- Objective functions/rewards: RF distance, BMEPRL, Likelihood

A network

A tree:
- Attributes:
    + adjacent matrix/distance matrix
- Methods: 
    + apply_SPR move
    + calculate the action (to feed the network)

A agent: 
- Attributes:
    + network
    + rule (policy - a str)
- Methods: 
    + choose actions: input - output

A environment:
- Attributes:
    + obj_type -> obj_fn -> reward_fn
    + Tree: the current tree
    + The current metric being tracked/optimized
    + Remaining steps in current episode
- Methods:
    + reset: choose a new tree
    + step: applying -> return new_state, reward
    + compute: a treelikelihood

Replaybuffer: 
- attributes:
    + memory: 
- methods: 
    + add_memory














Here is the concise map of those components in this repo.

**Network**
- `FFNet` at ff_net.py
- Inputs: `x` (state-action feature tensor), typically `[batch, n_actions, n_features]`
- Outputs: per-action score/logit (`[batch, n_actions, out_features]`, usually `out_features=1`)
- Purpose: score each candidate SPR action for the agent

**Tree**
- Core class: `Tree` at tree.py
- Wrapper used by env: `TrainingTree` at tree.py (and batched version at batched_bmep_tree.py)

Attributes (your focus):
- Distance matrix: `Tree.d`
- Adjacency matrix: `Tree.adj`

Methods (your focus):
1. `Tree.apply_SPR_move(pruned, regrafted)` in tree.py
- Input: prune edge/node tuple + regraft edge/node tuple
- Output: none (in-place topology update)
- Purpose: apply one SPR move to current tree

2. Action candidates to feed network:
- `Tree.compute_features_cpp()`
- `TrainingTree._compute_features()` stores `_actions`, `_features_tensor`
- `TrainingTree.get_features(normalize=False)`
- Input: current tree state
- Output: valid actions + action-feature tensor
- Purpose: produce candidate action features for agent/network

3. Objective computation (BMEP): (BO)
- `Tree.compute_obj()`, `Tree.compute_obj_val_from_adj_mat(...)`, `TrainingTree.get_length()`
- Input: tree (or adjacency + distance matrix)
- Output: scalar tree objective (BMEP length)
- Purpose: evaluate tree quality

Likelihood note:
- In this RL pipeline, the tracked objective is BMEP/tree length, not phylogenetic likelihood (no active tree-likelihood objective wired in `BMEPEnvironment`).

**Agent**
- `SPRPolicy` / `SPRValue` at agent.py

Attributes:
- Network: `policy_network` or `value_network`
- Rule/policy selector: `train_selection_method`, `test_selection_method` (greedy/random/eps-greedy/sample/etc.)

Method:
- `forward(...)`
- Input: state-action features (and optional `act_in`)
- Output: selected action index (plus optional logits/probs/values depending on flags)
- Purpose: choose action from candidate moves

**Environment**
- `BMEPEnvironment` at bmep_env.py

Attributes:
- Objective pipeline: `_obj_type -> _obj_fn` (currently `bmep_tree_l`)
- Reward pipeline: `_reward_type -> _reward_fn` (currently `improve_over_best_score`)
- Tree state is in `BMEPState` (`init_tree`, `current_tree`, `best_tree`) from bmep_utils.py
- Metric tracked/optimized: best tree length improvement (BMEP)
- Remaining steps: tracked by trainer loop (pg_trainer.py), not as an env internal counter

Methods:
1. `reset(problem_data)`
- Input: problem data (`BMEPData`)
- Output: new `BMEPState` with initialized trees
- Purpose: start episode

2. `step(problem_data, current_state, action, bspr_baseline_steps=0)`
- Input: current state + action
- Output: `(new_state, reward)` or `(new_state, reward, bspr_info)`
- Purpose: apply move, update state, compute reward

3. Objective compute path used by env:
- `_obj_fn = bmep_tree_l` in bmep_utils.py
- `bmep_tree_l(problem_data, tree) -> tree.get_length()`