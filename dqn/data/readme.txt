To keep track of the parameters used for different runs:


ReverseEnv_5_2_1_0.png
- using Bit Flipping reward style, limit eps to len(str) steps,
ReverseEnv_6_3_1_0.png
- Same as above
ReverseEnv_DQN_HER_10_3_1_0.png
- Using 1/0 Hairuo reward, limit to len(str) steps, no std. dev. in
  generate_strings from reverse_env.py so that all strings are 5 steps away
ReverseEnv_DQN_HER_7_3_1_0.png
Reverse_DQN_HER_10_3_1_0.png
- Using 1/0 Hairuo reward I think, limit to len(str), but with std. dev. in
  generate_strings from reverse_env.py to (5, 0.5, 2, 0) I think.
Reverse_Env_random_agent_5_2_1_0.png
- Random agent.
