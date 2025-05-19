import tensorflow as tf

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        """액터 네트워크를 초기화합니다."""
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.mean = tf.keras.layers.Dense(action_dim)
        self.log_std = tf.keras.layers.Dense(action_dim)

    def call(self, state):
        """상태를 입력받아 행동의 평균과 로그 표준편차를 반환합니다."""
        x = self.dense1(state)
        x = self.dense2(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std
