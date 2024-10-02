import pyglet
import random
import numpy as np

window = pyglet.window.Window(width=800, height=400, caption='Chrome Dino Game ML')
batch = pyglet.graphics.Batch()

class Dino:
    def __init__(self):
        self.sprite = pyglet.shapes.Rectangle(50, 60, 40, 60, color=(100, 100, 100), batch=batch)
        self.y_velocity = 0
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.y_velocity = 20
            self.is_jumping = True

    def update(self, dt):
        self.sprite.y += self.y_velocity
        if self.sprite.y > 60:
            self.y_velocity -= 60 * dt
        else:
            self.sprite.y = 60
            self.y_velocity = 0
            self.is_jumping = False

class Cactus:
    def __init__(self, x):
        height = random.randint(30, 60)
        self.sprite = pyglet.shapes.Rectangle(x, 60, 20, height, color=(0, 100, 0), batch=batch)

    def update(self, dt, speed):
        self.sprite.x -= speed * dt

class Ground:
    def __init__(self):
        self.sprites = [
            pyglet.shapes.Rectangle(0, 0, 800, 60, color=(200, 200, 200), batch=batch),
            pyglet.shapes.Line(0, 60, 800, 60, width=2, color=(100, 100, 100), batch=batch)
        ]

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(3)
        self.fitness = 0

    def predict(self, inputs):
        return np.dot(inputs, self.weights) > 0

    def mutate(self):
        self.weights += np.random.randn(3) * 0.1

def check_collision(a, b):
    return (a.x < b.x + b.width and
            a.x + a.width > b.x and
            a.y < b.y + b.height and
            a.y + a.height > b.y)

dino = Dino()
ground = Ground()
cacti = []
score = 0
game_over = False
generation = 1
population_size = 50
current_network = 0
networks = [NeuralNetwork() for _ in range(population_size)]
game_speed = 300

score_label = pyglet.text.Label('Score: 0', x=10, y=370, batch=batch)
generation_label = pyglet.text.Label(f'Generation: {generation}', x=10, y=350, batch=batch)
game_over_label = pyglet.text.Label('Game Over', x=400, y=200, anchor_x='center', batch=batch, color=(255, 0, 0, 0))

def reset_game():
    global dino, cacti, score, game_over, game_speed
    dino = Dino()
    cacti = []
    score = 0
    game_over = False
    game_speed = 300
    game_over_label.color = (255, 0, 0, 0)

def update(dt):
    global score, game_over, current_network, generation, game_speed

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
            generation_label.text = f'Generation: {generation}'
        reset_game()
        return

    dino.update(dt)

    if random.randint(1, 100) == 1 and len(cacti) < 3:
        cacti.append(Cactus(800))

    for cactus in cacti:
        cactus.update(dt, game_speed)
        if cactus.sprite.x < -30:
            cacti.remove(cactus)
            score += 1
            score_label.text = f'Score: {score}'
            game_speed += 2

        if check_collision(dino.sprite, cactus.sprite):
            game_over = True
            networks[current_network].fitness = score
            game_over_label.color = (255, 0, 0, 255)
            return

    if cacti:
        inputs = [
            (cacti[0].sprite.x - dino.sprite.x) / 800,
            (cacti[0].sprite.height - dino.sprite.height) / 60,
            dino.y_velocity / 20
        ]
        if networks[current_network].predict(inputs):
            dino.jump()

@window.event
def on_draw():
    window.clear()
    batch.draw()

pyglet.clock.schedule_interval(update, 1/60)
pyglet.app.run()