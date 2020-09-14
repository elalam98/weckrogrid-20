"""
Created on Fri Apr 10 18:45:19 2020
Scenario simulation
@author: Stamatis
"""

""" select number of random walks """
num_random_walks = 1

""" set initial seed """
if True:
    initial_seed = 13
    np.random.seed(initial_seed)
    random.seed(initial_seed)

""" random walk """
overall_rewards = []
for rw in range(num_random_walks):
    reset_environment()
    optimal_policy, rewards_path, states_path, prv_state = [], [], [], start_state
    for i in range(decision_periods):
        if rw == 0:
            if i == 2 or i == 3:
                pass
                #prv_state[0][25] = 200
                #prv_state[0][33] = 50
                #prv_state[0][41] = 400
        optimal_action, _ = MGrid.q_values(prv_state)
        if optimal_action != 0:
            if actions_map[optimal_action][0] in power_plants.keys() and actions_map[optimal_action][1] != 0:
                #print(i+1, power_plants[actions_map[optimal_action][0]].life_list[power_plants[actions_map[optimal_action][0]].life_state], power_plants[actions_map[optimal_action][0]].eff_list[power_plants[actions_map[optimal_action][0]].eff_state])
                pass
            if actions_map[optimal_action][0] in storage_units.keys() and actions_map[optimal_action][1] != 0:
                #print(i+1, storage_units[actions_map[optimal_action][0]].life_list[storage_units[actions_map[optimal_action][0]].life_state], storage_units[actions_map[optimal_action][0]].eff_list[storage_units[actions_map[optimal_action][0]].eff_state], storage_units[actions_map[optimal_action][0]].dod_list[storage_units[actions_map[optimal_action][0]].dod_state])
                pass
        #print(prv_state)
        nxt_state, reward_period = MGrid.compute_nextstate_reward(prv_state, optimal_action, 10)
        optimal_policy.append(actions_map[optimal_action])
        rewards_path.append(reward_period)
        states_path.append(prv_state)
        prv_state = nxt_state
    overall_rewards.append(np.mean(rewards_path))
