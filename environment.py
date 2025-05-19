import numpy as np

class AdversarialPush:
    def __init__(self):
        """환경을 초기화합니다."""
        self.state_dim = 10
        self.action_dim = 2

    def reset(self):
        """환경을 초기화하고 초기 상태를 반환합니다."""
        state_self = np.random.rand(self.state_dim)
        state_opponent = np.random.rand(self.state_dim)
        return state_self, state_opponent

    def step(self, action_self, action_opponent):
        """한 스텝을 진행하고 다음 상태, 보상, 종료 여부를 반환합니다."""
        next_state_self = np.random.rand(self.state_dim)
        next_state_opponent = np.random.rand(self.state_dim)
        reward_self = np.random.rand()
        reward_opponent = np.random.rand()
        done = np.random.choice([True, False])
        return next_state_self, next_state_opponent, reward_self, reward_opponent, done
