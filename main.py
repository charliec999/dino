import pyglet
import numpy as np
from PIL import Image

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
    
    def forward(self, X):
        self.hidden = np.maximum(0, np.dot(X, self.weights1))
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2))
        return self.output
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400
GROUND_HEIGHT = 50
DINO_X = 50

class Dino:
    def __init__(self):
        self.y = GROUND_HEIGHT
        self.velocity = 0
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.velocity = 15
            self.is_jumping = True

    def update(self):
        self.y += self.velocity
        self.velocity -= 1
        if self.y <= GROUND_HEIGHT:
            self.y = GROUND_HEIGHT
            self.is_jumping = False
            self.velocity = 0

class Obstacle:
    def __init__(self, x):
        self.x = x
        self.width = 20
        self.height = 40

    def update(self):
        self.x -= 5

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

dino = Dino()
obstacles = [Obstacle(WINDOW_WIDTH)]
score = 0

nn = NeuralNetwork(3, 4, 1)

@window.event
def on_draw():
    window.clear()
    pyglet.shapes.Rectangle(0, 0, WINDOW_WIDTH, GROUND_HEIGHT, color=(100, 100, 100)).draw()
    pyglet.shapes.Rectangle(DINO_X, dino.y, 40, 60, color=(200, 200, 200)).draw()
    for obstacle in obstacles:
        pyglet.shapes.Rectangle(obstacle.x, GROUND_HEIGHT, obstacle.width, obstacle.height, color=(150, 75, 0)).draw()
    score_label = pyglet.text.Label(f"Score: {score}", x=10, y=WINDOW_HEIGHT-30)
    score_label.draw()

def update(dt):
    global score
    dino.update()
    
    for obstacle in obstacles:
        obstacle.update()
    if obstacles[-1].x < WINDOW_WIDTH - 200:
        obstacles.append(Obstacle(WINDOW_WIDTH))
    if obstacles[0].x < -20:
        obstacles.pop(0)
        score += 1

    distance_to_obstacle = obstacles[0].x - (DINO_X + 40)
    obstacle_height = obstacles[0].height
    dino_y = dino.y

    nn_input = np.array([distance_to_obstacle, obstacle_height, dino_y])
    prediction = nn.forward(nn_input)

    if prediction > 0.5:
        dino.jump()

    if DINO_X + 40 > obstacles[0].x and DINO_X < obstacles[0].x + obstacles[0].width:
        if dino.y < GROUND_HEIGHT + obstacles[0].height:
            print(f"Game Over! Score: {score}")
            pyglet.app.exit()

pyglet.clock.schedule_interval(update, 1/60)
pyglet.app.run()