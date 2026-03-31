RL framework's possible options:
- Loss: Q learning, REINFORCE
- Objective functions/rewards: RF distance, BMEPRL, Likelihood
- Train and inference strategies: greedy, epsilon-greedy, softmax
- Train and inference parameters:

Vong lap training:
- Moi epoch chon 1 cay
- Thuc hien mot chuoi cac buoc SPR tren cay do
- Luu lai: state, action, reward, .... vao buffer[tree][epoch]
- Train: mang no ron network
    + Mang REINFORCE
    => Input: state + action
    => Output: reward
    => Loss: REINFORCE
    + Mang deep Q-learning
    => Input: state_1 + action_1 vao mang 1; state_2 + action_2 vao mang 2
    => Output: reward theo bieu thuc Bellman equation
    => Loss: MSE
- Test: 
    + Tu 1 state, xet toan bo SPR move
    + Encode state + action => cho qua mang no ron
    + Tinh diem va greedy'
    

A network

A tree:
- Attributes:
    + adjacent matrix/distance matrix
- Methods: 
    + apply_SPR move
    => Input: pruned + regraft
    => Output: None (modify the adjacent matrix)

    + calculate the action (to feed the network)
    => Input: None
    => Output: all possible SPR move + after state from the current tree
    (convert the adjacent matrix from tree --> C++. Xet toan bo move co the --> convert roi ve python)

A agent: 
- Attributes:
    + network
    + rule (policy - a str)
- Methods: forward
    => Input: all possible state-action features
    => Output: action idx

A environment: only consider one tree at a time.
- Attributes:
    + obj_type -> obj_fn -> reward_fn
    + Tree: the current tree
    + The current metric being tracked/optimized
    + Remaining steps in current episode
- Methods:
    + reset: 
        => Input: tree from newick file -> convert to a tree
        => Output: state of that tree
    + step: 
        => Input: current_state, action
        => Output: new_state, reward
    + compute: 
        => Input: state of a tree
        => Output: likelihood/length of current tree

Replaybuffer: 
- attributes:
    + memory[tree][epoch]
- methods: 
    + add_memory



Ba biến này là cấu hình để tạo các bài toán phylogenetic cho quá trình huấn luyện agent:

'instance_size': 20 — Kích thước của mỗi bài toán (số taxa/loài)

Mỗi "instance" là một ma trận khoảng cách (distance matrix) có kích thước 20×20
Đây là chiều dành của vấn đề cần giải quyết
'n_train_instances': 200 — Số lượng bài toán khác nhau để huấn luyện

Agent sẽ được huấn luyện trên 200 bài toán độc lập
Mỗi bài toán được tạo bằng cách lấy ngẫu nhiên 20 taxa từ dataset gốc
'n_test_instances': 50 — Số lượng bài toán khác nhau để kiểm tra

Agent được đánh giá trên 50 bài toán độc lập (khác với training set)
Dùng để đo hiệu suất và tránh overfitting
Như vậy: Agent sẽ được huấn luyện trên 200 bài toán (mỗi bài có 20 loài), rồi được kiểm tra trên 50 bài toán khác nhạc để đánh giá khả năng tổng quát hóa.














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