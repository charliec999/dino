import pyglet
import numpy as np
import tensorflow as tf
from pyglet.window import key
import random

window_width = 800
window_height = 400
ground_color = (139, 69, 19)
sky_color = (135, 206, 235)
dino_color = (0, 128, 0)
cactus_color = (139, 0, 0)

class Dino:
    def __init__(self):
        self.x = 100
        self.y = 150
        self.width = 20
        self.height = 40
        self.jump_height = 100
        self.is_jumping = False
        self.jump_speed = 0
        self.score = 0

    def jump(self):
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_speed = self.jump_height

    def update(self):
        if self.is_jumping:
            self.y += self.jump_speed
            self.jump_speed -= 5
            if self.jump_speed < -self.jump_height:
                self.is_jumping = False
                self.jump_speed = 0
                self.y = 150

class Cactus:
    def __init__(self):
        self.x = window_width
        self.y = 150
        self.width = 20
        self.height = 40

    def update(self, game_speed):
        self.x -= game_speed

class Ground:
    def __init__(self):
        self.y = 130
        self.color = ground_color

    def draw(self):
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
            ('v2f', [0, self.y, window_width, self.y, window_width, 0, 0, 0]),
            ('c3B', self.color * 4)
        )

class Agent:
    def __init__(self):
        self.model = self.build_model()
        self.memory = []
        self.gamma = 0.95

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=4, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def act(self, state):
        if np.random.rand() <= 0.1:
            return random.randrange(2)
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

window = pyglet.window.Window(window_width, window_height)
dino = Dino()
cacti = []
ground = Ground()
agent = Agent()
attempts = 0
score = 0

@window.event
def on_draw():
    window.clear()
    pyglet.gl.glClearColor(*sky_color)
    ground.draw()
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
        ('v2f', [dino.x, dino.y, dino.x + dino.width, dino.y, dino.x + dino.width, dino.y + dino.height, dino.x, dino.y + dino.height]),
        ('c3B', dino_color * 4)
    )
    for cactus in cacti:
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
            ('v2f', [cactus.x, cactus.y, cactus.x + cactus.width, cactus.y, cactus.x + cactus.width, cactus.y + cactus.height, cactus.x, cactus.y + cactus.height]),
            ('c3B', cactus_color * 4)
        )
    score_label = pyglet.text.Label(f'Score: {dino.score}', font_size=20, x=10, y=window_height - 30)
    score_label.draw()
    attempt_label = pyglet.text.Label(f'Attempts: {attempts}', font_size=20, x=10, y=window_height - 60)
    attempt_label.draw()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.SPACE:
        dino.jump()

def update(dt):
    global score, attempts
    game_speed = 5
    dino.update()
    if random.random() < 0.02:
        cacti.append(Cactus())
    for cactus in cacti:
        cactus.update(game_speed)
        if cactus.x < 0:
            cacti.remove(cactus)
            dino.score += 1
            attempts += 1
        if (dino.x < cactus.x + cactus.width and
            dino.x + dino.width > cactus.x and
            dino.y < cactus.y + cactus.height and
            dino.y + dino.height > cactus.y):
            attempts += 1
            dino.y = ground.y
            cacti.clear()
            dino.score = 0
    inputs = [
        (cacti[0].x - dino.x) / window_width if cacti else 0,
        (cacti[0].y - dino.y) / window_height if cacti else 0,
        dino.y / window_height,
        dino.y / ground.y
    ]
    action = agent.act(inputs)
    if action == 1:
        dino.jump()
    reward = 1 if dino.score > 0 else -1
    agent.remember(inputs, action, reward, inputs, False)
    agent.replay(32)

pyglet.clock.schedule_interval(update, 1 / 60)
pyglet.app.run()
