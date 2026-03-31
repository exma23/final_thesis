from feat_cpp.fastcpp import FastCpp

_cpp = FastCpp(lib_path='feat_cpp/bridge.so')

def create_batch(newick: str, action_chosen, gt_newick: str):
    return _cpp.get_state_action(newick, action_chosen, gt_newick)