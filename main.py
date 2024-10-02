import pygame, numpy as np, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from collections import deque, namedtuple
import random, math, time, json, os
from datetime import datetime

WINDOW_WIDTH, WINDOW_HEIGHT, FPS = 1200, 600, 60
COLORS = {'WHITE': (255,255,255), 'BLACK': (0,0,0), 'RED': (255,0,0), 'GREEN': (0,255,0), 'BLUE': (0,0,255), 'GRAY': (128,128,128)}
STATE_SIZE, HIDDEN_SIZE, ACTION_SIZE = 12, 256, 3
BATCH_SIZE, GAMMA, LEARNING_RATE = 128, 0.99, 0.0001
MEMORY_SIZE, UPDATE_TARGET_EVERY = 200000, 50
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Linear(channels, channels)
        )
    def forward(self, x): return F.relu(x + self.layers(x))

class AdvancedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            ResidualBlock(HIDDEN_SIZE),
            ResidualBlock(HIDDEN_SIZE)
        )
        self.advantage = nn.Sequential(ResidualBlock(HIDDEN_SIZE), nn.Linear(HIDDEN_SIZE, ACTION_SIZE))
        self.value = nn.Sequential(ResidualBlock(HIDDEN_SIZE), nn.Linear(HIDDEN_SIZE, 1))
        
    def forward(self, x):
        features = self.feature_extractor(x)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class PERBuffer:
    def __init__(self, capacity):
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size, self.capacity, self.next_idx = 0, capacity, 0
        self.beta = 0.4
        
    def add(self, priority, data):
        idx = self.next_idx + self.capacity - 1
        self.data[self.next_idx] = data
        self.update(idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree): break
            if s <= self.tree[left]: idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

    def sample(self, n):
        batch_idx, batch_priorities, batch = [], [], []
        segment = self.tree[0] / n
        self.beta = min(1, self.beta + 0.001)
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.get(s)
            batch_idx.append(idx)
            batch_priorities.append(priority)
            batch.append(data)
        sampling_probabilities = batch_priorities / self.tree[0]
        weights = (self.size * sampling_probabilities) ** -self.beta
        weights /= weights.max()
        return batch, batch_idx, weights

class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = AdvancedNN().to(self.device)
        self.target_net = AdvancedNN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = PERBuffer(MEMORY_SIZE)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.steps = 0
        self.epsilon = lambda: max(0.01, min(1, 1 - math.log10((self.steps + 1) / 5)))

    def act(self, state):
        if random.random() > self.epsilon():
            with torch.no_grad():
                return self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device)).max(1)[1].item()
        return random.randrange(ACTION_SIZE)

    def train(self):
        if self.memory.size < BATCH_SIZE: return
        experiences, indices, weights = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        expected_q = rewards + (1 - dones) * GAMMA * next_q

        td_error = torch.abs(current_q - expected_q.unsqueeze(1))
        for idx, error in zip(indices, td_error):
            self.memory.update(idx, error.item() ** 0.6)

        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q, expected_q.unsqueeze(1), reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        if self.steps % UPDATE_TARGET_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps += 1

    def save(self, score): 
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'score': score
        }, f"models/dino_{score}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

class Dino:
    def __init__(self):
        self.width, self.height = 40, 60
        self.x, self.y = 100, WINDOW_HEIGHT - 110
        self.dy, self.ducking = 0, False
        self.initial_height = self.height

    def jump(self):
        if self.y >= WINDOW_HEIGHT - self.height - 50: self.dy = -15

    def duck(self, should_duck):
        if should_duck and not self.ducking and self.y >= WINDOW_HEIGHT - self.height - 50:
            self.ducking = True
            self.height = self.initial_height // 2
            self.y = WINDOW_HEIGHT - self.height - 50
        elif not should_duck and self.ducking:
            self.ducking = False
            self.height = self.initial_height
            self.y = WINDOW_HEIGHT - self.height - 50

    def update(self):
        self.y += self.dy
        if self.y < WINDOW_HEIGHT - self.height - 50: self.dy += 0.8
        else: self.y, self.dy = WINDOW_HEIGHT - self.height - 50, 0

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Advanced AI Dino")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset()

    def reset(self):
        self.dino = Dino()
        self.obstacles = []
        self.speed = 5
        self.score = 0
        self.frame = 0
        self.night_mode = False
        return self._get_state()

    def _get_state(self):
        if not self.obstacles:
            return np.array([self.dino.y/WINDOW_HEIGHT, self.dino.dy/20, 1.0, 0.0, 0.0, 0.0, self.speed/20, 
                           1.0 if self.night_mode else 0.0] + [0]*4)
        obs = self.obstacles[0]
        next_obs = self.obstacles[1] if len(self.obstacles) > 1 else obs
        return np.array([
            self.dino.y/WINDOW_HEIGHT, self.dino.dy/20,
            (obs.x - self.dino.x)/WINDOW_WIDTH, obs.height/WINDOW_HEIGHT,
            obs.width/WINDOW_WIDTH, 1.0 if obs.type == 'bird' else 0.0,
            self.speed/20, 1.0 if self.night_mode else 0.0,
            (next_obs.x - self.dino.x)/WINDOW_WIDTH, next_obs.height/WINDOW_HEIGHT,
            next_obs.width/WINDOW_WIDTH, 1.0 if next_obs.type == 'bird' else 0.0
        ])

    def step(self, action):
        self.frame += 1
        if action == 0: self.dino.jump()
        self.dino.duck(action == 1)
        self.dino.update()
        
        if not self.obstacles or self.obstacles[-1].x < WINDOW_WIDTH - 300:
            obs_type = 'bird' if random.random() < 0.3 else 'cactus'
            if obs_type == 'bird':
                height, width = 40, 40
                y = WINDOW_HEIGHT - random.choice([50, 100, 150]) - height
            else:
                height, width = random.randint(40, 80), random.randint(20, 40)
                y = WINDOW_HEIGHT - height - 50
            self.obstacles.append(type('Obstacle', (), {'x': WINDOW_WIDTH, 'y': y, 'width': width, 
                                                      'height': height, 'type': obs_type}))

        self.obstacles = [o for o in self.obstacles if o.x + o.width > 0]
        for o in self.obstacles: o.x -= self.speed

        collision = any(self.dino.x < o.x + o.width and self.dino.x + self.dino.width > o.x and
                       self.dino.y < o.y + o.height and self.dino.y + self.dino.height > o.y 
                       for o in self.obstacles)

        self.score += 1
        self.speed += 0.001
        if self.score % 500 == 0: self.night_mode = not self.night_mode

        reward = -10 if collision else 0.1 + sum(1 for o in self.obstacles if o.x + o.width < self.dino.x)
        if self.obstacles: 
            dist = self.obstacles[0].x - (self.dino.x + self.dino.width)
            if 0 < dist < 200: reward += 0.2

        return self._get_state(), reward, collision or self.frame > 5000

    def draw(self):
        self.screen.fill(COLORS['GRAY'] if self.night_mode else COLORS['WHITE'])
        color = COLORS['WHITE'] if self.night_mode else COLORS['BLACK']
        pygame.draw.line(self.screen, color, (0, WINDOW_HEIGHT-50), (WINDOW_WIDTH, WINDOW_HEIGHT-50), 2)
        pygame.draw.rect(self.screen, color, (self.dino.x, self.dino.y, self.dino.width, self.dino.height))
        for o in self.obstacles:
            pygame.draw.rect(self.screen, color, (o.x, o.y, o.width, o.height))
        self.screen.blit(self.font.render(f'Score: {self.score}', True, color), (10, 10))
        pygame.display.flip()
        self.clock.tick(FPS)

def main():
    game = Game()
    agent = Agent()
    record = 0
    
    try:
        while True:
            state = game.reset()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: return

                action = agent.act(state)
                next_state, reward, done = game.step(action)
                agent.memory.add(abs(reward), Experience(state, action, reward, next_state, done))
                agent.train()
                state = next_state
                game.draw()

                if done:
                    if game.score > record:
                        record = game.score
                        agent.save(record)
                    print(f'Score: {game.score}, Record: {record}, Epsilon: {agent.epsilon():.4f}')
                    break
    except KeyboardInterrupt: pass

if __name__ == "__main__": main()