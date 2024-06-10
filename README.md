# Variable-Agnostic-Causal-Exploration-for-Reinforcement-Learning-VACERL

This is the codebase for the paper Variable-Agnostic-Causal-Exploration-for-Reinforcement-Learning published at ECML PKDD 2024

To run VACERL with a specific set of parameters, use the following command:
python main.py --env MG_1 --render-mode rgb_array --algorithm  HER --seed_type multi --reward 0.0001 --buffer-size-causal-discovery 50 --number_attention 70 --state_diff_attention 0.9 --state_diff_policy 0.85 --alteration-index 10  --training_f 600 --training_s 600 --total_timestep 10000000 --head_timestep 500000 --loop 20    

