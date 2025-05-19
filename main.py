import json
from agent import Agent
from replay_buffer import ReplayBuffer
from environment import AdversarialPush
from utils import load_config, save_model

def main():
    """SPAC 학습을 실행합니다."""
    config = load_config('config.json')
    env = AdversarialPush()
    agent = Agent(config)
    replay_buffer = ReplayBuffer(config['training']['buffer_size'])

    for episode in range(config['training']['num_episodes']):
        state_self, state_opponent = env.reset()
        done = False
        while not done:
            action_self = agent.get_action(state_self)
            action_opponent = agent.get_action(state_opponent)
            next_state_self, next_state_opponent, reward_self, reward_opponent, done = env.step(action_self, action_opponent)
            experience = (state_self, action_self, state_opponent, action_opponent, reward_self, next_state_self, next_state_opponent)
            replay_buffer.add(experience)
            state_self, state_opponent = next_state_self, next_state_opponent

            agent.train(replay_buffer, config['training']['batch_size'])

        if episode % 100 == 0:
            save_model(agent.actor, f'actor_{episode}.h5')
            save_model(agent.critic, f'critic_{episode}.h5')
            print(f"에피소드 {episode} 완료")

if __name__ == '__main__':
    main()
