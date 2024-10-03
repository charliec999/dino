import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math
import time
import json
import os
from datetime import datetime

WINDOW_WIDTH, WINDOW_HEIGHT, FPS = 1200, 800, 60
GAME_HEIGHT = 600
COLORS = {
    'WHITE': (255,255,255),
    'BLACK': (0,0,0),
    'RED': (255,0,0),
    'GREEN': (0,255,0),
    'BLUE': (0,0,255),
    'GRAY': (128,128,128),
    'YELLOW': (255,255,0)
}

STATE_SIZE, HIDDEN_SIZE, ACTION_SIZE = 15, 256, 3
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
    
    def forward(self, x):
        return F.relu(x + self.layers(x))

class AdvancedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            ResidualBlock(HIDDEN_SIZE),
            ResidualBlock(HIDDEN_SIZE),
            ResidualBlock(HIDDEN_SIZE)
        ])
        self.advantage = nn.Sequential(ResidualBlock(HIDDEN_SIZE), nn.Linear(HIDDEN_SIZE, ACTION_SIZE))
        self.value = nn.Sequential(ResidualBlock(HIDDEN_SIZE), nn.Linear(HIDDEN_SIZE, 1))
        self.activations = []
    
    def forward(self, x):
        self.activations = []
        features = x
        for layer in self.layers:
            features = layer(features)
            self.activations.append(features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else features)
        advantage = self.advantage(features)
        value = self.value(features)
        self.activations.append(advantage.detach().cpu().numpy())
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class PERBuffer:
    def __init__(self, capacity):
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.capacity = capacity
        self.next_idx = 0
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
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

    def sample(self, n):
        batch_idx = []
        batch_priorities = []
        batch = []
        segment = self.tree[0] / n
        self.beta = min(1, self.beta + 0.001)
        
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.get(s)
            batch_idx.append(idx)
            batch_priorities.append(priority)
            batch.append(data)
            
        sampling_probabilities = np.array(batch_priorities) / self.tree[0]
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
        self.generation = 0
        self.epsilon = lambda: max(0.01, min(1, 1 - math.log10((self.steps + 1) / 5)))

    def act(self, state):
        if random.random() > self.epsilon():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                self.policy_net(state_tensor)
                return self.policy_net(state_tensor).max(1)[1].item()
        return random.randrange(ACTION_SIZE)

    def train(self):
        if self.memory.size < BATCH_SIZE:
            return
            
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
            'generation': self.generation,
            'score': score
        }, f"models/dino_{score}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

class Dino:
    def __init__(self):
        self.width, self.height = 40, 60
        self.x, self.y = 100, GAME_HEIGHT - 110
        self.dy, self.ducking = 0, False
        self.initial_height = self.height

    def jump(self):
        if self.y >= GAME_HEIGHT - self.height - 50:
            self.dy = -15

    def duck(self, should_duck):
        if should_duck and not self.ducking and self.y >= GAME_HEIGHT - self.height - 50:
            self.ducking = True
            self.height = self.initial_height // 2
            self.y = GAME_HEIGHT - self.height - 50
        elif not should_duck and self.ducking:
            self.ducking = False
            self.height = self.initial_height
            self.y = GAME_HEIGHT - self.height - 50

    def update(self):
        self.y += self.dy
        if self.y < GAME_HEIGHT - self.height - 50:
            self.dy += 0.8
        else:
            self.y = GAME_HEIGHT - self.height - 50
            self.dy = 0

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Advanced AI Dino")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset()
        self.visualization_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT - GAME_HEIGHT))
        self.generation = 1

    def reset(self):
        self.dino = Dino()
        self.obstacles = []
        self.speed = 5
        self.score = 0
        self.frame = 0
        self.night_mode = False
        self.difficulty_multiplier = 1.0
        return self._get_state()

    def _get_state(self):
        if not self.obstacles:
            return np.array([
                self.dino.y/GAME_HEIGHT, self.dino.dy/20, 1.0, 0.0, 0.0, 0.0,
                self.speed/20, 1.0 if self.night_mode else 0.0,
                self.difficulty_multiplier, 1.0, 0.0, 0.0, 0.0,
                math.sin(self.frame/100), math.cos(self.frame/100)
            ])
        
        obs = self.obstacles[0]
        next_obs = self.obstacles[1] if len(self.obstacles) > 1 else obs
        
        return np.array([
            self.dino.y/GAME_HEIGHT, self.dino.dy/20,
            (obs.x - self.dino.x)/WINDOW_WIDTH, obs.height/GAME_HEIGHT,
            obs.width/WINDOW_WIDTH, 1.0 if obs.type == 'bird' else 0.0,
            self.speed/20, 1.0 if self.night_mode else 0.0,
            self.difficulty_multiplier,
            (next_obs.x - self.dino.x)/WINDOW_WIDTH,
            next_obs.height/GAME_HEIGHT, next_obs.width/WINDOW_WIDTH,
            1.0 if next_obs.type == 'bird' else 0.0,
            math.sin(self.frame/100), math.cos(self.frame/100)
        ])

    def step(self, action):
        self.frame += 1
        if action == 0:
            self.dino.jump()
        self.dino.duck(action == 1)
        self.dino.update()
        
        if not self.obstacles or self.obstacles[-1].x < WINDOW_WIDTH - 300 * (1 - self.difficulty_multiplier * 0.3):
            obs_type = 'bird' if random.random() < 0.3 + self.difficulty_multiplier * 0.1 else 'cactus'
            if obs_type == 'bird':
                height, width = 40, 40
                y = GAME_HEIGHT - random.choice([50, 100, 150]) - height
            else:
                height, width = random.randint(40, 80), random.randint(20, 40)
                y = GAME_HEIGHT - height - 50
            self.obstacles.append(type('Obstacle', (), {'x': WINDOW_WIDTH, 'y': y, 'width': width, 
                                                      'height': height, 'type': obs_type}))

        self.obstacles = [o for o in self.obstacles if o.x + o.width > 0]
        for o in self.obstacles:
            o.x -= self.speed * (1 + self.difficulty_multiplier * 0.2)

        collision = any(self.dino.x < o.x + o.width and self.dino.x + self.dino.width > o.x and
                       self.dino.y < o.y + o.height and self.dino.y + self.dino.height > o.y 
                       for o in self.obstacles)

        self.score += 0.1
        self.speed += 0.001
        self.difficulty_multiplier = min(2.0, 1.0 + self.score / 1000)
        if int(self.score) % 50 == 0:
            self.night_mode = not self.night_mode

        reward = -10 if collision else 0.1 + sum(1 for o in self.obstacles if o.x + o.width < self.dino.x)
        if self.obstacles:
            dist = self.obstacles[0].x - (self.dino.x + self.dino.width)
            if 0 < dist < 200:
                reward += 0.2

        return self._get_state(), reward, collision or self.frame > 5000

    def draw_network(self, activations):
        self.visualization_surface.fill(COLORS['WHITE'])
        if not activations:
            return
        
        layers = [STATE_SIZE] + [HIDDEN_SIZE] * 3 + [ACTION_SIZE]
        max_neurons = max(layers)
        layer_spacing = WINDOW_WIDTH / (len(layers) + 1)
        neuron_spacing = min(30, (WINDOW_HEIGHT - GAME_HEIGHT) / max_neurons)
        
        for l, (layer_size, activation) in enumerate(zip(layers, activations)):
            x = layer_spacing * (l + 1)
            for n in range(min(layer_size, 40)):
                y = (WINDOW_HEIGHT - GAME_HEIGHT) / 2 + (n - min(layer_size, 40)/2) * neuron_spacing
                value = activation[0][n] if n < len(activation[0]) else 0
                color = tuple(max(0, min(255, int(128 + value * 128))) for _ in range(3))
                pygame.draw.circle(self.visualization_surface, color, (int(x), int(y)), 8)
                
                if l < len(layers) - 1:
                    next_layer_size = min(layers[l + 1], 40)
                    next_x = layer_spacing * (l + 2)
                    for next_n in range(next_layer_size):
                        next_y = (WINDOW_HEIGHT - GAME_HEIGHT) / 2 + (next_n - next_layer_size/2) * neuron_spacing
                        pygame.draw.line(self.visualization_surface, COLORS['GRAY'], 
                                       (int(x), int(y)), (int(next_x), int(next_y)), 1)

    def draw(self, activations=None):
        if activations:
            self.draw_network(activations)
        self.screen.blit(self.visualization_surface, (0, 0))
        
        game_surface = pygame.Surface((WINDOW_WIDTH, GAME_HEIGHT))
        game_surface.fill(COLORS['GRAY'] if self.night_mode else COLORS['WHITE'])
        
        color = COLORS['WHITE'] if self.night_mode else COLORS['BLACK']
        pygame.draw.line(game_surface, color, (0, GAME_HEIGHT-50), (WINDOW_WIDTH, GAME_HEIGHT-50), 2)
        pygame.draw.rect(game_surface, color, (self.dino.x, self.dino.y, self.dino.width, self.dino.height))
        
        for o in self.obstacles:
            pygame.draw.rect(game_surface, color, (o.x, o.y, o.width, o.height))
            
        score_text = self.font.render(f'Score: {int(self.score)}', True, color)
        speed_text = self.font.render(f'Speed: {self.speed:.1f}x', True, color)
        difficulty_text = self.font.render(f'Difficulty: {self.difficulty_multiplier:.1f}x', True, color)
        generation_text = self.font.render(f'Generation: {self.generation}', True, color)
        
        game_surface.blit(score_text, (10, 10))
        game_surface.blit(speed_text, (10, 50))
        game_surface.blit(difficulty_text, (10, 90))
        game_surface.blit(generation_text, (10, 130))
        
        self.screen.blit(game_surface, (0, WINDOW_HEIGHT - GAME_HEIGHT))
        pygame.display.flip()
        self.clock.tick(FPS)

def main():
    if not os.path.exists("models"):
        os.makedirs("models")
        
    game = Game()
    agent = Agent()
    record = 0
    
    try:
        while True:
            state = game.reset()
            game.generation = agent.generation
            
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    agent.policy_net(state_tensor)
                    activations = agent.policy_net.activations
                
                action = agent.act(state)
                next_state, reward, done = game.step(action)
                agent.memory.add(abs(reward), Experience(state, action, reward, next_state, done))
                agent.train()
                state = next_state
                
                game.draw(activations)

                if done:
                    if game.score > record:
                        record = game.score
                        agent.save(record)
                    print(f'Generation: {agent.generation}, Score: {int(game.score)}, Record: {int(record)}, Epsilon: {agent.epsilon():.4f}, Difficulty: {game.difficulty_multiplier:.1f}x')
                    break
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if game.score > 100:
            agent.save(game.score)

if __name__ == "__main__":
    main()