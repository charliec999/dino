import pyglet
import numpy as np
import random
import tensorflow as tf
from pyglet.window import key

window_width = 800
window_height = 400
ground_y = 130
dino_color = (0, 128, 0)
cactus_color = (139, 0, 0)
ground_color = (139, 69, 19)
sky_color = (135 / 255, 206 / 255, 235 / 255, 1)

class Dino:
    def __init__(self):
        self.x = 100
        self.y = ground_y
        self.width = 20
        self.height = 40
        self.is_jumping = False
        self.jump_speed = 0
        self.gravity = 1

    def jump(self):
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_speed = 20

    def update(self):
        if self.is_jumping:
            self.y += self.jump_speed
            self.jump_speed -= self.gravity
            if self.y <= ground_y:
                self.y = ground_y
                self.is_jumping = False

class Cactus:
    def __init__(self):
        self.x = window_width
        self.y = ground_y
        self.width = 20
        self.height = 40

    def update(self, speed):
        self.x -= speed

class DQNAgent:
    def __init__(self):
        self.state_size = 4
        self.action_size = 2
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Game:
    def __init__(self):
        self.window = pyglet.window.Window(window_width, window_height)
        self.dino = Dino()
        self.cacti = []
        self.agent = DQNAgent()
        self.score = 0
        self.game_speed = 5
        self.batch_size = 32
        self.frames_since_action = 0
        self.max_cacti = 5  # Limit the number of cacti on screen
        pyglet.clock.schedule_interval(self.update, 1 / 120)

    def on_draw(self):
        self.window.clear()
        pyglet.gl.glClearColor(*sky_color)
        self.draw_ground()
        self.draw_dino()
        for cactus in self.cacti:
            self.draw_cactus(cactus)
        score_label = pyglet.text.Label(f'Score: {self.score}', font_size=20, x=10, y=window_height - 30)
        score_label.draw()

    def draw_ground(self):
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
            ('v2f', [0, ground_y, window_width, ground_y, window_width, 0, 0, 0]),
            ('c3B', ground_color * 4)
        )

    def draw_dino(self):
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
            ('v2f', [self.dino.x, self.dino.y, self.dino.x + self.dino.width, self.dino.y,
                      self.dino.x + self.dino.width, self.dino.y + self.dino.height, self.dino.x, self.dino.y + self.dino.height]),
            ('c3B', dino_color * 4)
        )

    def draw_cactus(self, cactus):
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
            ('v2f', [cactus.x, cactus.y, cactus.x + cactus.width, cactus.y,
                      cactus.x + cactus.width, cactus.y + cactus.height, cactus.x, cactus.y + cactus.height]),
            ('c3B', cactus_color * 4)
        )

    def update(self, dt):
        self.dino.update()
        
        # Only predict and act every 5 frames
        self.frames_since_action += 1
        if self.frames_since_action >= 5:
            self.learn()
            self.frames_since_action = 0
        
        # Limit cactus spawning frequency
        if len(self.cacti) < self.max_cacti and random.random() < 0.05:  # Adjust the probability
            self.cacti.append(Cactus())
        
        for cactus in self.cacti:
            cactus.update(self.game_speed)
            if cactus.x < -cactus.width:
                self.cacti.remove(cactus)
                self.score += 1
            if self.check_collision(cactus):
                self.reset_game()
                break

    def check_collision(self, cactus):
        return (self.dino.x < cactus.x + cactus.width and
                self.dino.x + self.dino.width > cactus.x and
                self.dino.y < cactus.y + cactus.height and
                self.dino.y + self.dino.height > cactus.y)

    def reset_game(self):
        self.cacti.clear()
        self.score = 0
        self.dino.y = ground_y
        self.dino.is_jumping = False

    def learn(self):
        try:
            state = np.array([self.dino.y, self.dino.x, len(self.cacti), self.cacti[0].x if self.cacti else window_width])
            state = np.reshape(state, [1, self.agent.state_size])
            action = self.agent.act(state)

            if action == 1:
                self.dino.jump()
            
            next_state = np.array([self.dino.y, self.dino.x, len(self.cacti), self.cacti[0].x if self.cacti else window_width])
            next_state = np.reshape(next_state, [1, self.agent.state_size])
            
            if self.cacti:
                reward = 1 if not self.check_collision(self.cacti[0]) else -10
            else:
                reward = 1
            
            done = False
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.replay(self.batch_size)
        except Exception as e:
            print(f"Error in learn method: {e}")

game = Game()
game.window.push_handlers(on_draw=game.on_draw)
pyglet.app.run()
