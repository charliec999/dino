import pyglet
import random
import numpy as np

window = pyglet.window.Window(width=800, height=400, caption='Chrome Dino Game ML')
batch = pyglet.graphics.Batch()

fullscreen = False
dino_image = pyglet.image.load('assets/dino.png')
cactus_image = pyglet.image.load('assets/cactus.png')
ground_image = pyglet.image.load('assets/ground.png')

class Dino:
    def __init__(self):
        self.sprite = pyglet.sprite.Sprite(dino_image, x=50, y=60, batch=batch)
        self.y_velocity = 0
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.y_velocity = 15
            self.is_jumping = True

    def update(self, dt):
        self.sprite.y += self.y_velocity
        if self.sprite.y > 60:
            self.y_velocity -= 50 * dt
        else:
            self.sprite.y = 60
            self.y_velocity = 0
            self.is_jumping = False

class Cactus:
    def __init__(self, x):
        self.sprite = pyglet.sprite.Sprite(cactus_image, x=x, y=60, batch=batch)

    def update(self, dt, speed):
        self.sprite.x -= speed * dt

class Ground:
    def __init__(self):
        self.sprites = [
            pyglet.sprite.Sprite(ground_image, x=0, y=0, batch=batch),
            pyglet.sprite.Sprite(ground_image, x=400, y=0, batch=batch)
        ]

    def update(self, dt, speed):
        for sprite in self.sprites:
            sprite.x -= speed * dt
            if sprite.x < -400:
                sprite.x += 800

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(3)
        self.fitness = 0

    def predict(self, inputs):
        return np.dot(inputs, self.weights) > 0

    def mutate(self):
        self.weights += np.random.randn(3) * 0.05

def check_collision(a, b):
    return (a.x < b.x + b.width and a.x + a.width > b.x and a.y < b.y + b.height and a.y + a.height > b.y)

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
        if cactus.sprite.x < -30:
            cacti.remove(cactus)
            score += 1
            score_label.text = f'{score}'
            game_speed += 3

        if check_collision(dino.sprite, cactus.sprite):
            game_over = True
            networks[current_network].fitness = score
            return

    if cacti:
        inputs = [
            (cacti[0].sprite.x - dino.sprite.x) / 800,
            (cacti[0].sprite.height - dino.sprite.height) / 60,
            dino.y_velocity / 15
        ]
        if networks[current_network].predict(inputs):
            dino.jump()

    game_speed += difficulty_increase_rate

@window.event
def on_draw():
    window.clear()
    batch.draw()

@window.event
def on_key_press(symbol, modifiers):
    global fullscreen
    if symbol == pyglet.window.key.F:
        fullscreen = not fullscreen
        window.set_fullscreen(fullscreen, width=window.width, height=window.height)

pyglet.clock.schedule_interval(update, 1/120)
pyglet.app.run()
