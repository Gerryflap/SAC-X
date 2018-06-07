import gym

gym.envs.register(
    id='MountainCarLong-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 600},
    reward_threshold=10000.00,
)