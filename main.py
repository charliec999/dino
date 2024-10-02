import pyglet
import random
import numpy as np
import tensorflow as tf
from collections import deque

window = pyglet.window.Window(width=800, height=400, caption='Chrome Dino Game RL')
batch = pyglet.graphics.Batch()

fullscreen = False
dino_color = (255, 0, 0)
cactus_color = (0, 255, 0)
ground_color = (160, 82, 45)
sky_color = (135, 206, 235)

class Dino:
    def __init__(self):
        self.x = 50
        self.y = 60
        self.y_velocity = 0
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.y_velocity = 15
            self.is_jumping = True

    def update(self, dt):
        self.y += self.y_velocity
        if self.y > 60:
            self.y_velocity -= 50 * dt
        else:
            self.y = 60
            self.y_velocity = 0
            self.is_jumping = False

class Cactus:
    def __init__(self, x):
        self.x = x
        self.y = 60

    def update(self, dt, speed):
        self.x -= speed * dt

class Ground:
    def __init__(self):
        self.x1 = 0
        self.x2 = 800

    def update(self, dt, speed):
        self.x1 -= speed * dt
        self.x2 -= speed * dt
        if self.x1 < -800:
            self.x1 += 800
        if self.x2 < -800:
            self.x2 += 800

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=3, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(2)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

dino = Dino()
ground = Ground()
cacti = []
score = 0
attempts = 0
game_over = False
agent = DQNAgent()
batch_size = 32
game_speed = 350
difficulty_increase_rate = 0.01

score_label = pyglet.text.Label('Score: 0', x=10, y=370, font_size=18, color=(255, 255, 255, 255), batch=batch)
attempt_label = pyglet.text.Label('Attempts: 0', x=700, y=370, font_size=18, color=(255, 255, 255, 255), batch=batch)
output_label = pyglet.text.Label('', x=400, y=350, font_size=18, color=(255, 255, 255, 255), batch=batch)

def reset_game():
    global dino, cacti, score, game_over
    dino = Dino()
    cacti = []
    score = 0
    game_over = False

def update(dt):
    global score, game_over, attempts, game_speed

    if game_over:
        attempts += 1
        reset_game()
        return

    dino.update(dt)
    ground.update(dt, game_speed)

    if random.randint(1, 100) == 1 and len(cacti) < 2:
        cacti.append(Cactus(800))

    for cactus in cacti:
        cactus.update(dt, game_speed)
        if cactus.x < -30:
            cacti.remove(cactus)
            score += 1
            score_label.text = f'Score: {score}'
            game_speed += 3

        if dino.x < cactus.x + 20 and dino.x + 20 > cactus.x and dino.y < cactus.y + 20 and dino.y + 20 > cactus.y:
            game_over = True
            output_label.text = 'Game Over! Resetting...'
            return

    if cacti:
        inputs = [
            (cacti[0].x - dino.x) / 800,
            (cacti[0].y - dino.y) / 60,
            dino.y_velocity / 15
        ]
        inputs = np.reshape(inputs, [1, 3])
        action = agent.act(inputs)
        if action == 1:
            dino.jump()
            output_label.text = "Bot thinks: Jump"
        else:
            output_label.text = "Bot thinks: Don't Jump"

    agent.remember(inputs, action, [0], inputs, game_over)
    agent.replay(batch_size)

    if attempts % 100 == 0:
        print(f'Attempt {attempts}: Score {score}, Epsilon {agent.epsilon:.2f}')

def draw_neural_network():
    layers = [3, 24, 24, 2]
    x_offset = (window.width - 200) / 2
    y_offset = 300

    for layer_index in range(len(layers)):
        for node_index in range(layers[layer_index]):
            x = x_offset + layer_index * 60 + 30
            y = y_offset - node_index * 20
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x - 10, y - 10, x + 10, y - 10, x + 10, y + 10, x - 10, y + 10]),
                                 ('c3B', (255, 255, 255) * 4))

@window.event
def on_draw():
    window.clear()
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [0, 0, window.width, 0, window.width, 200, 0, 200]),
                         ('c3B', sky_color * 4))
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [0, 0, window.width, 0, window.width, 60, 0, 60]),
                         ('c3B', ground_color * 4))
    draw_neural_network()
    if not game_over:
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [dino.x, dino.y, dino.x + 20, dino.y, dino.x + 20, dino.y + 20, dino.x, dino.y + 20]),
                             ('c3B', dino_color * 4))
        for cactus in cacti:
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [cactus.x, cactus.y, cactus.x + 20, cactus.y, cactus.x + 20, cactus.y + 20, cactus.x, cactus.y + 20]),
                                 ('c3B', cactus_color * 4))
    score_label.draw()
    attempt_label.text = f'Attempts: {attempts}'
    attempt_label.draw()
    output_label.draw()

@window.event
def on_key_press(symbol, modifiers):
    global fullscreen
    if symbol == pyglet.window.key.F:
        fullscreen = not fullscreen
        window.set_fullscreen(fullscreen)

pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()
