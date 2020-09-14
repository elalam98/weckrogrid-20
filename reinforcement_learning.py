"""
Created on Fri Jan 10 14:38:03 2020
Reinforcement learning
@author: Stamatis
"""

""" model id and choose save/load """
rl_model_id = 16
rl_model_save = False
rl_model_load = False

""" set initial seed """
initial_seed = 13
np.random.seed(initial_seed)
random.seed(initial_seed)

""" rl hyperparameters """
rl_hparams = dict()
rl_hparams['algorithm'] = 'ddqn'
rl_hparams['num_episodes'] = 1_000
rl_hparams['print_every'] = rl_hparams['num_episodes'] // 100 if rl_hparams['num_episodes'] >= 100 else rl_hparams['num_episodes']
rl_hparams['save_every'] = rl_hparams['num_episodes'] // 10
rl_hparams['update_target'] = rl_hparams['num_episodes'] // 1_000 if rl_hparams['num_episodes'] >= 1_000 else 1
rl_hparams['decision_periods'] = decision_periods
rl_hparams['buffer_size'] = 20_000
rl_hparams['gamma'] = 0.9
rl_hparams['epsilon'] = 1
rl_hparams['epsilon_min'] = 0.01
rl_hparams['lrate'] = 0.01
rl_hparams['tau'] = 1
rl_hparams['nn_units'] = 128
rl_hparams['nn_layers'] = 2
rl_hparams['batch_size'] = 32
rl_hparams['dropout'] = None
rl_hparams['verbose_loop'] = False
rl_hparams['normalize_states'] = True

class StopExecution(Exception):
    def _render_traceback_(self):
        pass

""" create a class for the MGrid Agent """
class MGridAgent:
    
    """ initialize the agent """
    def __init__(self):
        self.buffer = deque(maxlen=rl_hparams['buffer_size'])
        self.epsilon, self.epsilon_step = rl_hparams['epsilon'], (rl_hparams['epsilon'] - rl_hparams['epsilon_min']) / rl_hparams['num_episodes']
        self.experiences_counter = 0
        self.main_model = self.create_model()
        self.target_model = self.create_model()
    
    """ helper function for creating the deep neural networks """    
    def create_model(self):
        model = Sequential()
        model.add(Dense(rl_hparams['nn_units'], kernel_initializer=initializers.glorot_uniform(seed=initial_seed), input_dim=states_dim, activation='relu'))
        for _ in range(rl_hparams['nn_layers']):
            model.add(Dense(rl_hparams['nn_units'], kernel_initializer=initializers.glorot_uniform(seed=initial_seed), activation='relu'))
            if rl_hparams['dropout']:
                model.add(Dropout(rl_hparams['dropout']))
        model.add(Dense(actions_dim))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=rl_hparams['lrate']))
        return model
    
    """ function for agent action """
    def take_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)   # explore
        if rl_hparams['normalize_states']:
            return np.argmax(self.main_model.predict(norm_state(state))[0]) # exploit
        else:
            return np.argmax(self.main_model.predict(state)[0]) # exploit
        
    """ compute next state and reward based on state transition rules and reward function """
    def compute_nextstate_reward(self, state, action, num_runs):
        """ define next state """
        action_tuple = actions_map[action]
        state = state[0]
        period = int(state[0])
        next_state = [period+1] # periods go up by 1
        """ change the capacity based on the selected action and compute any retirement costs """
        retirement_cost = 0
        if action_tuple != 0:
            if action_tuple[0] in power_plants.keys(): # decision to install/replace a power plant
                retirement_cost = power_plants[action_tuple[0]].decom_cost * power_plants[action_tuple[0]].cap # decommissioning costs
                retirement_cost += power_plants[action_tuple[0]].rem_loan # remaining loan payment 
                power_plants[action_tuple[0]].install_new(action_tuple[1])
            else: # decision to install/replace a storage unit
                retirement_cost = storage_units[action_tuple[0]].decom_cost * storage_units[action_tuple[0]].cap # decommissioning costs
                retirement_cost += storage_units[action_tuple[0]].rem_loan # remaining loan payment
                storage_units[action_tuple[0]].install_new(action_tuple[1])

        """ compute operation and outage cost """
        operation_cost, outage_cost, _, _ = outage_simulation_helper(period, num_runs)

        """ make other necessary changes to states and compute loan payments and om cost """
        investment_cost, om_cost = 0, 0
        for power_plant in power_plants: # decrease remaining life of power plants
            if power_plants[power_plant].cap > 0:
                investment_cost += min(power_plants[power_plant].rem_loan, power_plants[power_plant].loan_payment * years_in_period)
                power_plants[power_plant].rem_loan = max(power_plants[power_plant].rem_loan - power_plants[power_plant].loan_payment * years_in_period, 0)
                om_cost += power_plants[power_plant].om_cost
                if power_plants[power_plant].rem_life <= years_in_period:
                    retirement_cost += power_plants[power_plant].decom_cost * power_plants[power_plant].cap
            power_plants[power_plant].decrease_life()
            for external_feature in external_features_power_plants:
                power_plants[power_plant].state_transition(external_feature)    # markov chain transitions for external features
                next_state.append(getattr(power_plants[power_plant], external_feature + '_list')[getattr(power_plants[power_plant], external_feature + '_state')])
            for internal_feature in internal_features_power_plants:
                next_state.append(getattr(power_plants[power_plant], internal_feature))
        for storage_unit in storage_units: # decrease remaining life of storage units
            if storage_units[storage_unit].cap > 0:
                investment_cost += min(storage_units[storage_unit].rem_loan, storage_units[storage_unit].loan_payment * years_in_period)
                storage_units[storage_unit].rem_loan = max(storage_units[storage_unit].rem_loan - storage_units[storage_unit].loan_payment * years_in_period, 0)
                om_cost += storage_units[storage_unit].om_cost
                if storage_units[storage_unit].rem_life <= years_in_period:
                    retirement_cost += storage_units[storage_unit].decom_cost * storage_units[storage_unit].cap  
            storage_units[storage_unit].decrease_life()    
            for external_feature in external_features_storage_units:
                storage_units[storage_unit].state_transition(external_feature)  # markov chain transitions for external features
                next_state.append(getattr(storage_units[storage_unit], external_feature + '_list')[getattr(storage_units[storage_unit], external_feature + '_state')])
            for internal_feature in internal_features_storage_units:
                next_state.append(getattr(storage_units[storage_unit], internal_feature))            

        next_state = np.array(next_state)
        reward = - investment_cost - operation_cost - outage_cost - om_cost - retirement_cost
        return next_state.reshape(1,states_dim), reward
        
    """ update buffer """
    def buffer_update(self, state, action, reward, new_state, done):
        if rl_hparams['normalize_states']:
            state, new_state = norm_state(state).reshape(1,states_dim), norm_state(new_state).reshape(1,states_dim)
        else:
            state, new_state = state.reshape(1,states_dim), new_state.reshape(1,states_dim)
        self.buffer.append([state, action, reward, new_state, done])
        self.experiences_counter += 1
        
    """ experience replay for dqn algorithm """
    def experience_replay_dqn(self):
        if len(self.buffer) < rl_hparams['batch_size']:
            return
        x_batch, y_batch = [], []
        batch = random.sample(self.buffer, rl_hparams['batch_size'])
        for state, action, reward, new_state, done in batch:
            target = self.main_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_future = max(self.target_model.predict(new_state)[0]) # select the best action and evaluate it on the target network
                target[0][action] = reward + q_future * rl_hparams['gamma']
            x_batch.append(state[0])
            y_batch.append(target[0])
        self.main_model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        
    """ experience replay for ddqn algorithm """
    def experience_replay_ddqn(self):
        if len(self.buffer) < rl_hparams['batch_size']:
            return
        x_batch, y_batch = [], []
        batch = random.sample(self.buffer, rl_hparams['batch_size'])
        for state, action, reward, new_state, done in batch:
            target = self.main_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                best_action = np.argmax(self.main_model.predict(new_state)[0])  # select the best action on the main network
                q_future = self.target_model.predict(new_state)[0][best_action] # evaluate this action on the target network
                target[0][action] = reward + q_future * rl_hparams['gamma']
            x_batch.append(state[0])
            y_batch.append(target[0])
        self.main_model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        
    """ experience replay for greedy algorithm """
    def experience_replay_greedy(self):
        if len(self.buffer) < rl_hparams['batch_size']:
            return
        x_batch, y_batch = [], []
        batch = random.sample(self.buffer, rl_hparams['batch_size'])
        for state, action, reward, new_state, done in batch:
            target = self.main_model.predict(state)
            target[0][action] = reward
            x_batch.append(state[0])
            y_batch.append(target[0])
        self.main_model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        
    """ update the target network """
    def target_update(self):
        main_weights, target_weights = self.main_model.get_weights(), self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = main_weights[i] * rl_hparams['tau'] + target_weights[i] * (1 - rl_hparams['tau'])
        self.target_model.set_weights(target_weights)

    """ use the trained neural network to get q-values """
    def q_values(self, state, print_all=False):
        if rl_hparams['normalize_states']:
            q_val = self.main_model.predict(norm_state(state).reshape(1,states_dim))[0]
        else:
            q_val = self.main_model.predict(state.reshape(1,states_dim))[0]
        if print_all:
            print('\nOptimal action: ' + str(np.argmax(q_val)))
            print('\nQ-value for optimal action: ' + str(q_val[np.argmax(q_val)]))
            for i,v in enumerate(q_val):
                print('\nQ-value for action ' + str(i) + ': ' + str(v))
        return np.argmax(q_val), q_val
        
    """ save model """
    def save_model(self, snap):
        self.main_model.save(snap)
        
    def print_statistics(self, episode, treward):
        time.sleep(0)
        print('\n\nAverage cost for the percentile ' + str(episode//rl_hparams['print_every']) + ': ' + str(-treward//rl_hparams['print_every']))
        time_per_episode = (time.time() - start_time) / episode
        time_to_finish = time_per_episode * (rl_hparams['num_episodes'] - episode)
        time_to_finish_hours, time_to_finish = str(int(time_to_finish // SEC_IN_HOUR)), time_to_finish % SEC_IN_HOUR
        time_to_finish_minutes, time_to_finish = str(int(time_to_finish // SEC_IN_MIN)), time_to_finish % SEC_IN_MIN
        time_to_finish_seconds = str(int(time_to_finish // 1))
        if len(time_to_finish_hours) == 1:
            time_to_finish_hours = '0' + time_to_finish_hours
        if len(time_to_finish_minutes) == 1:
            time_to_finish_minutes = '0' + time_to_finish_minutes
        if len(time_to_finish_seconds) == 1:
            time_to_finish_seconds = '0' + time_to_finish_seconds
        print('\nApproximate time to finish: ' + str(time_to_finish_hours) + ':' + str(time_to_finish_minutes) + ':' + str(time_to_finish_seconds))
        time.sleep(0)
        return
    
    def print_final_statistics(self):
        return
    
    def print_figures(self,saved_rew):
        x_axis = [x for x in range(1,(rl_hparams['num_episodes']//rl_hparams['print_every'])+1)]
        plt.plot(x_axis,saved_rew)
        
""" agent initialization """
MGrid = MGridAgent()

""" load a pre-trained model """
if rl_model_load:
    MGrid.main_model = load_model(file_path + '/Saved Versions/Version ' + str(rl_model_id) + '/MGrid_main_model_v' + str(rl_model_id) + '.h5')
    raise StopExecution

""" main function """
total_reward, saved_total_costs = 0, []
experience_replay = {'dqn': MGrid.experience_replay_dqn, 'ddqn': MGrid.experience_replay_ddqn, 'greedy': MGrid.experience_replay_greedy}
episode_iterator = [range(rl_hparams['num_episodes']), tqdm(range(rl_hparams['num_episodes']))]
start_time = time.time()
for episode in episode_iterator[rl_hparams['verbose_loop']]:
    reset_environment()
    MGrid.epsilon -= MGrid.epsilon_step
    cur_state = start_state
    print(episode)
    print(cur_state)
    """ save model instance """
    if rl_model_save and episode != 0 and episode % rl_hparams['save_every'] == 0:
        MGrid.main_model.save(file_path + '/MGrid_main_model_v' + str(rl_model_id) + '_' + str(episode // rl_hparams['save_every']) + '.h5')
    """ print statistics """
    if episode != 0 and episode % rl_hparams['print_every'] == 0:
        MGrid.print_statistics(episode,total_reward)
        saved_total_costs.append(-total_reward//rl_hparams['print_every'])
        total_reward = 0
    """ update target network """
    if episode != 0 and episode % rl_hparams['update_target'] == 0:
        MGrid.target_update()
    for decision_period in range(rl_hparams['decision_periods']):
        action = MGrid.take_action(cur_state)
        new_state, reward = MGrid.compute_nextstate_reward(cur_state, action, settings['num_outage_simulation_runs'])
        total_reward += reward
        done = (decision_period + 1 == rl_hparams['decision_periods'])
        MGrid.buffer_update(cur_state, action, reward, new_state, done)
        if MGrid.experiences_counter % rl_hparams['batch_size'] == 0:
            experience_replay[rl_hparams['algorithm']]()
        cur_state = new_state
MGrid.print_statistics(episode+1,total_reward)
saved_total_costs.append(-total_reward//rl_hparams['print_every'])
    
""" print final statistics """
MGrid.print_final_statistics()
MGrid.print_figures(saved_total_costs)

""" save results """
if rl_model_save:
    MGrid.main_model.save(file_path + '/MGrid_main_model_v' + str(rl_model_id) + '.h5') # save model
    parameters = dict()
    parameters['decision_periods'], parameters['years_in_period'], parameters['loan_horizon'], parameters['settings'], parameters['nn_hparams'], parameters['rl_hparams'] = decision_periods, years_in_period, loan_horizon, settings, nn_hparams, rl_hparams
    with open(file_path + '/MGrid_parameters_v' + str(rl_model_id) + '.txt', 'w') as outfile:
        json.dump(parameters, outfile)