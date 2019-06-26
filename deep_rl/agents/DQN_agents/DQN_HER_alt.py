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
        print('observations:\n' + '\n'.join(map(str, self.episode_observations)))
        print(str(self.episode_desired_goals[0]) + ' < goal')
        print('actions: ' + str(self.episode_actions))

        """Saves the experiences as if the next state visited in the episode was the goal state"""
        # need to be careful to store goals as g(state) and not just assume
        # g(state) = state. used achieved_goal
        new_states = [self.create_state_from_observation_and_desired_goal(
                ep_obs, ep_achieved_goal)
                for ep_obs, ep_achieved_goal in 
                # the goal achieved by the current action is next_achieved_goal
                zip(self.episode_observations, self.episode_next_achieved_goals)]
        print('new_states: \n' + '\n'.join(map(str,new_states)))
#        new_states = [self.create_state_from_observation_and_desired_goal(observation, new_goal) for observation in self.episode_observations]
        new_next_states = [self.create_state_from_observation_and_desired_goal(
                obs, next_achieved_goal)
                for obs, next_achieved_goal in 
                zip(self.episode_next_observations, self.episode_next_achieved_goals)]
        print('new_next_states: \n' + '\n'.join(map(str,new_next_states)))
#        new_next_states = [self.create_state_from_observation_and_desired_goal(observation, new_goal) for observation in
#                      self.episode_next_observations]
#        new_rewards = [self.environment.compute_reward(next_achieved_goal,
#            next_achieved_goal, None)
        new_rewards = [self.environment.reward_for_achieving_goal]*len(
                self.episode_next_achieved_goals)
#            for next_achieved_goal in
#            self.episode_next_achieved_goals]
        print('new_rewards: ' + str(new_rewards))
#        new_rewards = [self.environment.compute_reward(next_achieved_goal, new_goal, None) for next_achieved_goal in  self.episode_next_achieved_goals]

        if self.hyperparameters["clip_rewards"]:
            new_rewards = [max(min(reward, 1.0), -1.0) for reward in new_rewards]

        self.HER_memory.add_experience(new_states, self.episode_actions, new_rewards, new_next_states, self.episode_dones)
