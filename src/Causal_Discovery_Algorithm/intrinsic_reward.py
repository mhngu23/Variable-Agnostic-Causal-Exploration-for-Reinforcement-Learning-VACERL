
def create_intrinsic_reward(configuration, causal_graph):
    """
    This functions will calculate the intrinsic rewards based on the hierarchy of a (state, action) pair in the causal graph.
    """
    hyper_const = configuration["reward"]
    goal_reward = 1
    disminish_reward = goal_reward/len(causal_graph)

    keys_list = list(causal_graph.keys())
    keys_list.reverse()


    # init the reward dictionary
    intrinsic_reward = {keys_list[0]:goal_reward}

    for i in range(len(keys_list)):
        key = keys_list[i]
        if key in intrinsic_reward.keys() and i == 0:
            # get current item at current height
            current_height = causal_graph[key]
            for item in current_height:
                # example for cases with 5 items in causal graph (1 - 1/5) * 0.01 = 0.008
                intrinsic_reward[item] = (intrinsic_reward[key] - disminish_reward) * hyper_const
                if intrinsic_reward[item] < 0:
                    intrinsic_reward[item] = 0
        elif key in intrinsic_reward.keys() and i > 0:
            current_height = causal_graph[key]
            for item in current_height:
                # example for cases with 5 items in causal graph 0.008 - 1/5 * 0.01 = 0.006
                intrinsic_reward[item] = intrinsic_reward[key] - disminish_reward * hyper_const
                if intrinsic_reward[item] < 0:
                    intrinsic_reward[item] = 0
        else:
            continue
    return intrinsic_reward

def create_intrinsic_reward_attention_only(configuration, dict_attention):
    """
    This functions will calculate the intrinsic rewards based on the hierarchy of a (state, action) pair in the causal graph.
    """
    hyper_const = configuration["reward"]

    keys_list = list(dict_attention.keys())

    # init the reward dictionary
    intrinsic_reward = {}

    for i in range(len(keys_list)):
        key = keys_list[i]
        [value, _, _] = dict_attention.get(tuple(key))
        intrinsic_reward[key] = hyper_const * value
    return intrinsic_reward