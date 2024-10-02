import pyglet
import random
import numpy as np

window = pyglet.window.Window(width=800, height=400, caption='Chrome Dino Game ML')
batch = pyglet.graphics.Batch()

dino_color = (0, 150, 0)
cactus_color = (150, 100, 0)
ground_color = (139, 69, 19)  # Brown color for the ground
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

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(3)
        self.fitness = 0

    def predict(self, inputs):
        return np.dot(inputs, self.weights) > 0

    def mutate(self):
        self.weights += np.random.randn(3) * 0.1

dino = Dino()
ground = Ground()
cacti = []
score = 0
attempts = 0
generation = 1
population_size = 100
current_network = 0
networks = [NeuralNetwork() for _ in range(population_size)]
game_speed = 350
difficulty_increase_rate = 0.01

score_label = pyglet.text.Label('Score: 0', x=10, y=370, font_size=18, color=(255, 255, 255, 255), batch=batch)
attempt_label = pyglet.text.Label('Attempts: 0', x=10, y=350, font_size=18, color=(255, 255, 255, 255), batch=batch)
generation_label = pyglet.text.Label(f'Gen: {generation}', x=700, y=370, font_size=18, color=(255, 255, 255, 255), batch=batch)
thinking_label = pyglet.text.Label('', x=300, y=370, font_size=14, color=(255, 255, 255, 255), batch=batch)
bot_output_label = pyglet.text.Label('', x=10, y=330, font_size=14, color=(255, 255, 255, 255), batch=batch)

def reset_game():
    global dino, cacti, score, attempts, game_speed
    dino = Dino()
    cacti = []
    score = 0
    attempts += 1
    game_speed = 350
    score_label.text = f'Score: {score}'
    attempt_label.text = f'Attempts: {attempts}'
    bot_output_label.text = ''

def update(dt):
    global score, current_network, generation, game_speed

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

        if (dino.x < cactus.x + 20 and dino.x + 20 > cactus.x and
            dino.y < cactus.y + 20 and dino.y + 20 > cactus.y):
            reset_game()
            return

    if cacti:
        inputs = [
            (cacti[0].x - dino.x) / 800,
            (cacti[0].y - dino.y) / 60,
            dino.y_velocity / 15
        ]
        if networks[current_network].predict(inputs):
            dino.jump()
            thinking_label.text = "Bot: Jumping!"
            bot_output_label.text = f"Bot Output: Jumping! W0: {networks[current_network].weights[0]:.2f}, W1: {networks[current_network].weights[1]:.2f}, W2: {networks[current_network].weights[2]:.2f}"
        else:
            thinking_label.text = "Bot: Not Jumping!"
            bot_output_label.text = f"Bot Output: Not Jumping! W0: {networks[current_network].weights[0]:.2f}, W1: {networks[current_network].weights[1]:.2f}, W2: {networks[current_network].weights[2]:.2f}"

    game_speed += difficulty_increase_rate

def draw_neural_network():
    node_spacing = 100
    start_x = (window.width - 3 * node_spacing) / 2
    for i in range(3):
        x = start_x + i * node_spacing
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, 370, x + 40, 370, x + 40, 400, x, 400]),
                             ('c3B', (255, 255, 255) * 4))
        pyglet.text.Label(f'W{i}: {networks[current_network].weights[i]:.2f}', x=x + 5, y=375,
                          font_size=12, color=(0, 0, 0, 255), batch=batch).draw()

@window.event
def on_draw():
    window.clear()
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [0, 200, window.width, 200, window.width, window.height, 0, window.height]),
                         ('c3B', sky_color * 4))
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [0, 0, window.width, 0, window.width, 200, 0, 200]),
                         ('c3B', ground_color * 4))
    draw_neural_network()
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [dino.x, dino.y, dino.x + 20, dino.y, dino.x + 20, dino.y + 40, dino.x, dino.y + 40]),
                         ('c3B', dino_color * 4))
    for cactus in cacti:
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [cactus.x, cactus.y, cactus.x + 20, cactus.y, cactus.x + 20, cactus.y + 40, cactus.x, cactus.y + 40]),
                             ('c3B', cactus_color * 4))
    score_label.draw()
    attempt_label.draw()
    generation_label.draw()
    thinking_label.draw()
    bot_output_label.draw()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.F:
        window.set_fullscreen(not window.fullscreen)

@window.event
def on_resize(width, height):
    window.set_size(width, height)

pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()
