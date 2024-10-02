import pyglet
import random
import numpy as np

window = pyglet.window.Window(width=800, height=400, caption='Chrome Dino Game ML')
batch = pyglet.graphics.Batch()

fullscreen = False
dino_color = (255, 0, 0)
cactus_color = (0, 255, 0)
ground_color = (255, 255, 0)

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
        self.x2 = 400

    def update(self, dt, speed):
        self.x1 -= speed * dt
        self.x2 -= speed * dt
        if self.x1 < -400:
            self.x1 += 800
        if self.x2 < -400:
            self.x2 += 800

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(3)
        self.fitness = 0

    def predict(self, inputs):
        return np.dot(inputs, self.weights) > 0

    def mutate(self):
        self.weights += np.random.randn(3) * 0.05

def check_collision(dino, cactus):
    return (dino.x < cactus.x + 20 and dino.x + 20 > cactus.x and dino.y < cactus.y + 20 and dino.y + 20 > cactus.y)

dino = Dino()
ground = Ground()
cacti = []
score = 0
game_over = False
generation = 1
population_size = 100
current_network = 0
networks = [NeuralNetwork() for _ in range(population_size)]
game_speed = 350
difficulty_increase_rate = 0.01

score_label = pyglet.text.Label('0', x=10, y=370, font_size=18, color=(255, 255, 255, 255), batch=batch)
generation_label = pyglet.text.Label(f'Gen: {generation}', x=700, y=370, font_size=18, color=(255, 255, 255, 255), batch=batch)

def reset_game():
    global dino, cacti, score, game_over, game_speed
    dino = Dino()
    cacti = []
    score = 0
    game_over = False
    game_speed = 350

def update(dt):
    global score, game_over, current_network, generation, game_speed, difficulty_increase_rate

    if game_over:
        current_network += 1
        if current_network >= population_size:
            networks.sort(key=lambda x: x.fitness, reverse=True)
            new_networks = networks[:10]
            for _ in range(population_size - 10):
                network = random.choice(new_networks)
                new_network = NeuralNetwork()
                new_network.weights = network.weights.copy()
                new_network.mutate()
                new_networks.append(new_network)
            networks[:] = new_networks
            current_network = 0
            generation += 1
            generation_label.text = f'Gen: {generation}'
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
            score_label.text = f'{score}'
            game_speed += 3

        if check_collision(dino, cactus):
            game_over = True
            networks[current_network].fitness = score
            return

    if cacti:
        inputs = [
            (cacti[0].x - dino.x) / 800,
            (cacti[0].y - dino.y) / 60,
            dino.y_velocity / 15
        ]
        if networks[current_network].predict(inputs):
            dino.jump()

    game_speed += difficulty_increase_rate

def draw_neural_network():
    for i in range(3):
        x = (window.width - 120) / 2 + i * 60
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, 370, x + 40, 370, x + 40, 400, x, 400]),
                             ('c3B', (255, 255, 255) * 4))
        pyglet.text.Label(f'W{i}: {networks[current_network].weights[i]:.2f}', x=x + 5, y=375,
                          font_size=12, color=(0, 0, 0, 255), batch=batch).draw()

@window.event
def on_draw():
    window.clear()
    pyglet.graphics.draw(2, pyglet.gl.GL_QUADS, ('v2f', [0, 0, window.width, 0, window.width, 200, 0, 200]), 
                         ('c3B', (100, 100, 100) * 4))
    draw_neural_network()
    if not game_over:
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [dino.x, dino.y, dino.x + 20, dino.y, dino.x + 20, dino.y + 20, dino.x, dino.y + 20]), 
                             ('c3B', dino_color * 4))
        for cactus in cacti:
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [cactus.x, cactus.y, cactus.x + 20, cactus.y, cactus.x + 20, cactus.y + 20, cactus.x, cactus.y + 20]),
                                 ('c3B', cactus_color * 4))

@window.event
def on_key_press(symbol, modifiers):
    global fullscreen
    if symbol == pyglet.window.key.F:
        fullscreen = not fullscreen
        window.set_fullscreen(fullscreen)

pyglet.clock.schedule_interval(update, 1/120)
pyglet.app.run()
