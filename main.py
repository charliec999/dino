import pyglet
import random
import numpy as np

window = pyglet.window.Window(width=800, height=400, caption='Chrome Dino Game ML')
batch = pyglet.graphics.Batch()

fullscreen = False
dino_color = (255, 255, 255)
cactus_color = (200, 50, 50)
ground_color = (150, 150, 150)

class Dino:
    def __init__(self):
        self.width = 40
        self.height = 60
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
        self.width = 20
        self.height = random.randint(30, 60)
        self.x = x
        self.y = 60

    def update(self, dt, speed):
        self.x -= speed * dt

class Ground:
    def __init__(self):
        self.y = 0

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(3)
        self.fitness = 0

    def predict(self, inputs):
        return np.dot(inputs, self.weights) > 0

    def mutate(self):
        self.weights += np.random.randn(3) * 0.05

def check_collision(dino, cactus):
    return (dino.x < cactus.x + cactus.width and 
            dino.x + dino.width > cactus.x and 
            dino.y < cactus.y + cactus.height and 
            dino.y + dino.height > cactus.y)

dino = Dino()
ground = Ground()
cacti = []
score = 0
game_over = False
generation = 1
population_size = 100
current_network = 0
networks = [NeuralNetwork() for _ in range(population_size)]
game_speed = 300
difficulty_increase_rate = 0.01

score_label = pyglet.text.Label('Score: 0', x=10, y=370, font_size=18, color=(255, 255, 255, 255))
generation_label = pyglet.text.Label(f'Gen: {generation}', x=700, y=370, font_size=18, color=(255, 255, 255, 255))

def reset_game():
    global dino, cacti, score, game_over, game_speed
    dino = Dino()
    cacti = []
    score = 0
    game_over = False
    game_speed = 300

def draw_dino():
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [dino.x, dino.y, dino.x + dino.width, dino.y,
                                                        dino.x + dino.width, dino.y + dino.height, dino.x, dino.y + dino.height]),
                         ('c3B', dino_color * 4))

def draw_cacti():
    for cactus in cacti:
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [cactus.x, cactus.y, cactus.x + cactus.width, cactus.y,
                                                            cactus.x + cactus.width, cactus.y + cactus.height, cactus.x, cactus.y + cactus.height]),
                             ('c3B', cactus_color * 4))

def draw_ground():
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [0, ground.y, 800, ground.y,
                                                        800, ground.y + 60, 0, ground.y + 60]),
                         ('c3B', ground_color * 4))

def draw_neural_network():
    weights = networks[current_network].weights
    for i, weight in enumerate(weights):
        # Normalize weight to be in range [0, 1] for color
        weight_color = max(0, min(1, weight))
        color = (int(255 * weight_color), int(255 * (1 - weight_color)), 0)
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [50 + i * 60, 370, 50 + i * 60 + 40, 370,
                                                            50 + i * 60 + 40, 370 + 20, 50 + i * 60, 370 + 20]),
                             ('c3B', color * 4))

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

    if random.randint(1, 100) == 1 and len(cacti) < 2:
        cacti.append(Cactus(800))

    for cactus in cacti:
        cactus.update(dt, game_speed)
        if cactus.x < -30:
            cacti.remove(cactus)
            score += 1
            score_label.text = f'Score: {score}'
            game_speed += 3

        if check_collision(dino, cactus):
            game_over = True
            networks[current_network].fitness = score
            return

    if cacti:
        inputs = [
            (cacti[0].x - dino.x) / 800,
            (cacti[0].height - dino.height) / 60,
            dino.y_velocity / 15
        ]
        if networks[current_network].predict(inputs):
            dino.jump()

    game_speed += difficulty_increase_rate

@window.event
def on_draw():
    window.clear()
    draw_ground()
    draw_dino()
    draw_cacti()
    draw_neural_network()
    score_label.draw()
    generation_label.draw()

@window.event
def on_key_press(symbol, modifiers):
    global fullscreen
    if symbol == pyglet.window.key.F:
        fullscreen = not fullscreen
        window.set_fullscreen(fullscreen)

pyglet.clock.schedule_interval(update, 1/120)
pyglet.app.run()
