from deep_rl.agents.DQN_agents.DQN import DQN
from deep_rl.agents.HER_Base import HER_Base

class DQN_HER_alt(HER_Base, DQN):
    """DQN algorithm with hindsight experience replay, modified so that 
       the experiences and loss are done as Hairuo wanted: 
       - goal sampling is done with immediate next state as goal
       - hence reward is always reward_for_achieving_goal
       - loss is calculated with the addition of cross entropy. """

    agent_name = "DQN_HER_alt"
    def __init__(self, config):
        DQN.__init__(self, config)
        HER_Base.__init__(self, self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
                          self.hyperparameters["HER_sample_proportion"])

    def step(self):
        """Runs one episode of the game, including learning steps if required"""
        while not self.done:
            # pick a move
            self.action = self.pick_action()
            # put move into environment, get output from state_dict and store it
            self.conduct_action_in_changeable_goal_envs(self.action)
            # updates every n steps via config param (1 in Bit Flipping) and as
            # long as there is enough experience in the buffers
            if self.time_for_q_network_to_learn():
                # default learning iters = 1 for Bit_Flipping
                for _ in range(self.hyperparameters["learning_iterations"]):
                    self.learn(experiences=self.sample_from_HER_and_Ordinary_Buffer())
            # stores actions, reawrds, etc. from last step
            self.track_changeable_goal_episodes_data()
            # saves normal experience replay with newly formatted data from ^
            self.save_experience()
            # do the HER replay only at the end. Base implementation is just
            # final
            if self.done: self.save_alternative_experience()
            self.state_dict = self.next_state_dict  # this is to set the state for the next iteration
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1

    def enough_experiences_to_learn_from(self):
        """Returns booleans indicating whether there are enough experiences in the two replay buffers to learn from"""
        return len(self.memory) > self.ordinary_buffer_batch_size and len(self.HER_memory) > self.HER_buffer_batch_size


    def save_alternative_experience(self):
        """Saves the experiences as if the next state visited in the episode was the goal state"""
        # need to be careful to store goals as g(state) and not just assume
        # g(state) = state. used achieved_goal
        new_states = [self.create_state_from_observation_and_desired_goal(
                ep_obs, ep_achieved_goal)
                for ep_obs, ep_achieved_goal in 
                # the goal achieved by the current action is next_achieved_goal
                zip(self.episode_observations, self.episode_next_achieved_goals)]
        new_next_states = [self.create_state_from_observation_and_desired_goal(
                obs, next_achieved_goal)
                for obs, next_achieved_goal in 
                zip(self.episode_next_observations, self.episode_next_achieved_goals)]
        new_rewards = [self.environment.reward_for_achieving_goal]*len(
                self.episode_next_achieved_goals)
        # subtract from the reward 9
        losses = []
        for state in new_states:
            probs = self.compute_q_values_for_next_states(state)
          
        torch_states = torch.from_numpy(np.vstack(new_states)).float().to(
                self.device)
        probs = self.compute_q_valeus_for_next_states(torch_states)



#        criterion = nn.CrossEntropyLoss()
#        probs = 
        # given current state/goal, probability of choosing different actions.
        # Current state goals are new_states list.

        # given states, gives q values for all the actions
#        self.compute_q_values_for_next_states(next_states)
#        loss = criterion(probs, torch.Tensor([action]).long()).reshape([1])
#        policy_loss.append(-1*loss)
        print('new_rewards: ' + str(new_rewards))

        if self.hyperparameters["clip_rewards"]:
            new_rewards = [max(min(reward, 1.0), -1.0) for reward in new_rewards]

        self.HER_memory.add_experience(new_states, self.episode_actions, new_rewards, new_next_states, self.episode_dones)
