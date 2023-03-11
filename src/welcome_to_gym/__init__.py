from gym.envs.registration import register

register(
    id='WelcomeTo-v0',
    entry_point='welcome_to_gym.envs:WelcomeToEnv',
)
