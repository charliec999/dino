import pyglet
import numpy as np
import random
from collections import deque
import tensorflow as tf

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
GROUND_Y = 100
COLORS = {
    'dino': (0, 128, 0),
    'cactus': (139, 0, 0),
    'ground': (139, 69, 19),
    'sky': (135, 206, 235),
    'text': (255, 255, 255),
    'node': (255, 0, 0)
}

class Dino:
    def __init__(self):
        self.x, self.y = 50, GROUND_Y
        self.width, self.height = 40, 60
        self.jump_velocity = 0
        self.gravity = -0.8
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.jump_velocity = 15
            self.is_jumping = True

    def update(self):
        if self.is_jumping:
            self.y += self.jump_velocity
            self.jump_velocity += self.gravity
            if self.y <= GROUND_Y:
                self.y = GROUND_Y
                self.is_jumping = False
                self.jump_velocity = 0

class Obstacle:
    def __init__(self, x):
        self.x = x
        self.y = GROUND_Y
        self.width = random.randint(20, 40)
        self.height = random.randint(40, 80)

    def update(self, speed):
        self.x -= speed

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Game(pyglet.window.Window):
    def __init__(self):
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "Dino AI", resizable=True)
        self.batch = pyglet.graphics.Batch()
        self.dino = Dino()
        self.obstacles = []
        self.speed = 5
        self.score = 0
        self.high_score = 0
        self.attempts = 0
        self.game_over = False
        self.frame_count = 0
        self.difficulty_increment = 0.001
        self.state_size = 4
        self.action_size = 2
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.batch_size = 32
        pyglet.clock.schedule_interval(self.update, 1/60.0)

    def on_draw(self):
        self.clear()
        pyglet.gl.glClearColor(*COLORS['sky'], 1)
        self.batch.draw()
        self.draw_ground()
        self.draw_dino()
        self.draw_obstacles()
        self.draw_ui()

    def draw_ground(self):
        pyglet.shapes.Rectangle(0, 0, self.width, GROUND_Y, color=COLORS['ground'], batch=self.batch).draw()

    def draw_dino(self):
        pyglet.shapes.Rectangle(self.dino.x, self.dino.y, self.dino.width, self.dino.height, color=COLORS['dino'], batch=self.batch).draw()

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            pyglet.shapes.Rectangle(obstacle.x, obstacle.y, obstacle.width, obstacle.height, color=COLORS['cactus'], batch=self.batch).draw()
        if self.obstacles:
            obstacle = self.obstacles[0]
            node_x = (self.dino.x + self.dino.width + obstacle.x) / 2
            node_y = (self.dino.y + obstacle.y + obstacle.height) / 2
            pyglet.shapes.Circle(node_x, node_y, 5, color=COLORS['node'], batch=self.batch).draw()

    def draw_ui(self):
        pyglet.text.Label(f"Score: {self.score}", x=10, y=self.height - 30, color=(255, 255, 255, 255), batch=self.batch).draw()
        pyglet.text.Label(f"High Score: {self.high_score}", x=10, y=self.height - 60, color=COLORS['text'], batch=self.batch).draw()
        pyglet.text.Label(f"Attempts: {self.attempts}", x=10, y=self.height - 90, color=COLORS['text'], batch=self.batch).draw()

    def update(self, dt):
        self.frame_count += 1
        if self.game_over:
            return
        self.dino.update()
        state = self.get_state()
        action = self.agent.act(state)
        if action == 1:
            self.dino.jump()
        self.speed += self.difficulty_increment
        self.update_obstacles()
        self.check_collisions(state, action)
        self.score += 1
        self.agent.remember(state, action, 0.1, self.get_state(), False)
        self.train_agent()

    def update_obstacles(self):
        for obstacle in self.obstacles:
            obstacle.update(self.speed)
        if len(self.obstacles) == 0 or self.obstacles[-1].x < self.width - 200:
            self.obstacles.append(Obstacle(self.width))
        self.obstacles = [obs for obs in self.obstacles if obs.x + obs.width > 0]

    def check_collisions(self, state, action):
        for obstacle in self.obstacles:
            if self.check_collision(self.dino, obstacle):
                self.game_over = True
                self.agent.remember(state, action, -10, self.get_state(), True)
                self.high_score = max(self.high_score, self.score)
                self.reset_game()
                return

    def get_state(self):
        if self.obstacles:
            obstacle = self.obstacles[0]
            return np.array([[
                self.dino.y / self.height,
                obstacle.x / self.width,
                obstacle.width / self.width,
                self.speed / 20
            ]])
        return np.array([[self.dino.y / self.height, 1, 0, self.speed / 20]])

    def check_collision(self, dino, obstacle):
        return (dino.x < obstacle.x + obstacle.width and
                dino.x + dino.width > obstacle.x and
                dino.y < obstacle.y + obstacle.height and
                dino.y + dino.height > obstacle.y)

    def reset_game(self):
        self.dino = Dino()
        self.obstacles = []
        self.speed = 5
        self.score = 0
        self.game_over = False
        self.attempts += 1

    def train_agent(self):
        if self.frame_count % 10 == 0 and len(self.agent.memory) >= self.batch_size:
            self.agent.replay(self.batch_size)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.F:
            self.toggle_fullscreen()

    def toggle_fullscreen(self):
        self.set_fullscreen(not self.fullscreen)

if __name__ == "__main__":
    game = Game()
    pyglet.app.run()
