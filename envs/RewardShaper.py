class RewardShaper():
    def __init__(self, env_name):
        self.env_name = env_name

    def shape_reward(self, reward, next_state, done):
        # Reward shaping
        if self.env_name == "MountainCarContinuous-v0":
            if next_state[0] > 0.0 and next_state[1] > 0:
                reward = 0.01
            elif next_state[0] > 0.1:
                reward = 5.0
            else:
                reward = -0.1
        elif self.env_name == "InvertedPendulum-v2" or \
                self.env_name == "InvertedDoublePendulum-v2" or \
                self.env_name == "CartPole-v0":
            reward = -1.0 if done else 0.1
        elif self.env_name == "Acrobot-v1":
            if reward == -1.0:
                reward = -0.1
        elif self.env_name == "FetchReach-v1" or \
                self.env_name == "HandReach-v0" or \
                self.env_name == "HandManipulateBlock-v0" or \
                self.env_name == "HandManipulateEgg-v0" or \
                self.env_name == "FetchPush-v1":
            reward = 1.0 if abs(reward) < 1e-3 else -1.0
        elif self.env_name == "HumanoidStandup-v2":
            reward = (reward - 30.0) / 30.0

        return reward
