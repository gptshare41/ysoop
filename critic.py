import tensorflow as tf

class ComprehensiveCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        """종합적인 크리틱 네트워크를 초기화합니다."""
        super(ComprehensiveCritic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.q_value = tf.keras.layers.Dense(1)

    def call(self, state_self, action_self, state_opponent, action_opponent):
        """종합적인 정보를 입력받아 Q 값을 반환합니다."""
        x = tf.concat([state_self, action_self, state_opponent, action_opponent], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.q_value(x)
