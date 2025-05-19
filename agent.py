import tensorflow as tf
from actor import Actor
from critic import ComprehensiveCritic

class Agent:
    def __init__(self, config):
        """에이전트를 초기화합니다."""
        self.actor = Actor(config['environment']['state_dim'], config['environment']['action_dim'])
        self.critic = ComprehensiveCritic(config['environment']['state_dim'], config['environment']['action_dim'])
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=config['agent']['actor_lr'])
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=config['agent']['critic_lr'])
        self.gamma = config['agent']['gamma']
        self.tau = config['agent']['tau']

    def get_action(self, state):
        """상태를 입력받아 행동을 반환합니다."""
        mean, log_std = self.actor(state)
        std = tf.exp(log_std)
        action = mean + tf.random.normal(shape=mean.shape) * std
        return action

    def train(self, replay_buffer, batch_size):
        """리플레이 버퍼에서 데이터를 샘플링하여 학습합니다."""
        if len(replay_buffer) < batch_size:
            return
        experiences = replay_buffer.sample(batch_size)
        states_self, actions_self, states_opponent, actions_opponent, rewards, next_states_self, next_states_opponent = zip(*experiences)

        # 텐서로 변환
        states_self = tf.convert_to_tensor(states_self)
        actions_self = tf.convert_to_tensor(actions_self)
        states_opponent = tf.convert_to_tensor(states_opponent)
        actions_opponent = tf.convert_to_tensor(actions_opponent)
        rewards = tf.convert_to_tensor(rewards)
        next_states_self = tf.convert_to_tensor(next_states_self)
        next_states_opponent = tf.convert_to_tensor(next_states_opponent)

        # 크리틱 업데이트
        with tf.GradientTape() as tape:
            next_mean, next_log_std = self.actor(next_states_self)
            next_action_self = next_mean + tf.random.normal(shape=next_mean.shape) * tf.exp(next_log_std)
            next_action_opponent = self.get_action(next_states_opponent)
            target_q = rewards + self.gamma * self.critic(next_states_self, next_action_self, next_states_opponent, next_action_opponent)
            current_q = self.critic(states_self, actions_self, states_opponent, actions_opponent)
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # 액터 업데이트
        with tf.GradientTape() as tape:
            mean, log_std = self.actor(states_self)
            action = mean + tf.random.normal(shape=mean.shape) * tf.exp(log_std)
            actor_loss = -tf.reduce_mean(self.critic(states_self, action, states_opponent, actions_opponent))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
