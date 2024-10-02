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


WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 600
FPS = 60


COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'GRAY': (128, 128, 128)
}


GRAVITY = 0.8
JUMP_SPEED = -15
INITIAL_GAME_SPEED = 5
SPEED_INCREMENT = 0.001
NIGHT_MODE_THRESHOLD = 500


STATE_SIZE = 8
HIDDEN_SIZE = 128
ACTION_SIZE = 3  
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0001
MEMORY_SIZE = 100000
UPDATE_TARGET_EVERY = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class AdvancedNeuralNetwork(nn.Module):
    def __init__(self):
        super(AdvancedNeuralNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.Dropout(0.2)
        )
        
        self.advantage_layer = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, ACTION_SIZE)
        )
        
        self.value_layer = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        advantages = self.advantage_layer(features)
        values = self.value_layer(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.alpha = 0.6  
        self.beta = 0.4   
        self.beta_increment = 0.001
        self.epsilon = 1e-5  
        
    def push(self, experience):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, weights
        
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + self.epsilon

class AdvancedDQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = AdvancedNeuralNetwork().to(self.device)
        self.target_net = AdvancedNeuralNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.episode_rewards = []
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(ACTION_SIZE)
            
    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
    def train(self):
        batch = self.memory.sample(BATCH_SIZE)
        if batch is None:
            return
            
        experiences, indices, weights = batch
        
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([exp.done for exp in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        
        td_errors = torch.abs(current_q_values - expected_q_values.unsqueeze(1)).detach().cpu().numpy()
        
        
        self.memory.update_priorities(indices, td_errors.squeeze())
        
        
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1), reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        
        if self.steps_done % UPDATE_TARGET_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.steps_done += 1
        
    def save_model(self, score):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'score': score
        }, f"{model_dir}/dino_model_{score}_{timestamp}.pth")

class Obstacle:
    def __init__(self, x, y, width, height, obstacle_type):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = obstacle_type  
        self.passed = False

    def update(self, game_speed):
        self.x -= game_speed
        
    def draw(self, screen, night_mode):
        color = COLORS['GRAY'] if night_mode else COLORS['BLACK']
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))

class Dino:
    def __init__(self):
        self.width = 40
        self.height = 60
        self.x = 100
        self.y = WINDOW_HEIGHT - self.height - 50
        self.dy = 0
        self.ducking = False
        self.initial_height = self.height
        
    def jump(self):
        if self.y >= WINDOW_HEIGHT - self.height - 50:
            self.dy = JUMP_SPEED
            
    def duck(self):
        if not self.ducking and self.y >= WINDOW_HEIGHT - self.height - 50:
            self.ducking = True
            self.height = self.initial_height // 2
            self.y = WINDOW_HEIGHT - self.height - 50
            
    def stop_duck(self):
        if self.ducking:
            self.ducking = False
            self.height = self.initial_height
            self.y = WINDOW_HEIGHT - self.height - 50
            
    def update(self):
        self.y += self.dy
        if self.y < WINDOW_HEIGHT - self.height - 50:
            self.dy += GRAVITY
        else:
            self.y = WINDOW_HEIGHT - self.height - 50
            self.dy = 0
            
    def draw(self, screen, night_mode):
        color = COLORS['GRAY'] if night_mode else COLORS['BLACK']
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Advanced AI Dino Game")
        self.clock = pygame.time.Clock()
        
        self.dino = Dino()
        self.obstacles = []
        self.game_speed = INITIAL_GAME_SPEED
        self.score = 0
        self.high_score = 0
        self.agent = AdvancedDQNAgent()
        self.frame_iteration = 0
        self.night_mode = False
        
        self.font = pygame.font.Font(None, 36)
        
    def reset(self):
        self.dino = Dino()
        self.obstacles = []
        self.game_speed = INITIAL_GAME_SPEED
        self.score = 0
        self.frame_iteration = 0
        self.night_mode = False
        return self._get_state()
        
    def _get_state(self):
        if not self.obstacles:
            return np.array([
                self.dino.y / WINDOW_HEIGHT,
                self.dino.dy / 20,
                1.0,  
                0.0,  
                0.0,  
                0.0,  
                self.game_speed / 20,
                1.0 if self.night_mode else 0.0
            ])
            
        next_obstacle = self.obstacles[0]
        return np.array([
            self.dino.y / WINDOW_HEIGHT,
            self.dino.dy / 20,
            (next_obstacle.x - self.dino.x) / WINDOW_WIDTH,
            next_obstacle.height / WINDOW_HEIGHT,
            next_obstacle.width / WINDOW_WIDTH,
            1.0 if next_obstacle.type == 'bird' else 0.0,
            self.game_speed / 20,
            1.0 if self.night_mode else 0.0
        ])
        
    def _generate_obstacle(self):
        if not self.obstacles or self.obstacles[-1].x < WINDOW_WIDTH - 300:
            obstacle_type = 'bird' if random.random() < 0.3 else 'cactus'
            
            if obstacle_type == 'bird':
                height = 40
                width = 40
                y = WINDOW_HEIGHT - random.choice([50, 100, 150]) - height
            else:
                height = random.randint(40, 80)
                width = random.randint(20, 40)
                y = WINDOW_HEIGHT - height - 50
                
            self.obstacles.append(Obstacle(WINDOW_WIDTH, y, width, height, obstacle_type))
            
    def _calculate_reward(self, collision):
        if collision:
            return -10
        
        reward = 0.1  
        
        
        for obstacle in self.obstacles:
            if not obstacle.passed and obstacle.x + obstacle.width < self.dino.x:
                obstacle.passed = True
                reward += 1
                
        
        if self.obstacles:
            next_obstacle = self.obstacles[0]
            distance = next_obstacle.x - (self.dino.x + self.dino.width)
            if 0 < distance < 200:  
                reward += 0.2
                
        return reward
        
    def step(self, action):
        self.frame_iteration += 1
        
        
        if action == 0:  
            self.dino.jump()
        elif action == 1:  
            self.dino.duck()
        else:  
            self.dino.stop_duck()
            
        
        self.dino.update()
        self._generate_obstacle()
        
        
        self.obstacles = [obs for obs in self.obstacles if obs.x + obs.width > 0]
        for obstacle in self.obstacles:
            obstacle.update(self.game_speed)
            
        
        collision = False
        for obstacle in self.obstacles:
            if (self.dino.x < obstacle.x + obstacle.width and
                self.dino.x + self.dino.width > obstacle.x and
                self.dino.y < obstacle.y + obstacle.height and
                self.dino.y + self.dino.height > obstacle.y):
                collision = True
                break
                
        
        
        self.score += 1
        self.game_speed += SPEED_INCREMENT
        
        
        if self.score % NIGHT_MODE_THRESHOLD == 0:
            self.night_mode = not self.night_mode
            
        
        reward = self._calculate_reward(collision)
        
        
        new_state = self._get_state()
        
        
        self.high_score = max(self.score, self.high_score)
        
        
        done = collision or self.frame_iteration > 2000
        
        return new_state, reward, done
        
    def draw(self):
        
        background_color = COLORS['GRAY'] if self.night_mode else COLORS['WHITE']
        self.screen.fill(background_color)
        
        
        ground_color = COLORS['WHITE'] if self.night_mode else COLORS['BLACK']
        pygame.draw.line(self.screen, ground_color, 
                        (0, WINDOW_HEIGHT - 50), 
                        (WINDOW_WIDTH, WINDOW_HEIGHT - 50), 2)
        
        
        self.dino.draw(self.screen, self.night_mode)
        
        
        for obstacle in self.obstacles:
            obstacle.draw(self.screen, self.night_mode)
        
        
        text_color = COLORS['WHITE'] if self.night_mode else COLORS['BLACK']
        score_text = self.font.render(f'Score: {self.score}', True, text_color)
        high_score_text = self.font.render(f'High Score: {self.high_score}', True, text_color)
        epsilon_text = self.font.render(f'Epsilon: {self.agent.epsilon:.4f}', True, text_color)
        speed_text = self.font.render(f'Speed: {self.game_speed:.1f}', True, text_color)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(high_score_text, (10, 50))
        self.screen.blit(epsilon_text, (10, 90))
        self.screen.blit(speed_text, (10, 130))
        
        
        pygame.display.flip()
        
class DinoTrainer:
    def __init__(self):
        self.game = Game()
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0
        self.record = 0
        self.n_games = 0
        
    def train(self):
        while True:
            
            state = self.game.reset()
            
            episode_reward = 0
            done = False
            
            while not done:
                
                action = self.game.agent.select_action(state)
                
                
                next_state, reward, done = self.game.step(action)
                
                
                experience = Experience(state, action, reward, next_state, done)
                self.game.agent.memory.push(experience)
                
                
                self.game.agent.train()
                
                
                state = next_state
                episode_reward += reward
                
                
                self.game.draw()
                self.game.clock.tick(FPS)
                
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                        
            
            self.n_games += 1
            self.game.agent.update_epsilon()
            
            
            self.plot_scores.append(self.game.score)
            self.total_score += self.game.score
            mean_score = self.total_score / self.n_games
            self.plot_mean_scores.append(mean_score)
            
            
            if self.game.score > self.record:
                self.record = self.game.score
                self.game.agent.save_model(self.record)
                
            
            print(f'Game {self.n_games}, Score {self.game.score}, Record {self.record}, '
                  f'Epsilon {self.game.agent.epsilon:.4f}, Mean Score {mean_score:.2f}')
                  
            
            self._save_statistics()
            
    def _save_statistics(self):
        stats = {
            'n_games': self.n_games,
            'record': self.record,
            'scores': self.plot_scores,
            'mean_scores': self.plot_mean_scores,
            'total_score': self.total_score,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        stats_dir = "statistics"
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
            
        with open(f"{stats_dir}/stats_{self.n_games}.json", 'w') as f:
            json.dump(stats, f)

class DinoVisualizer:
    def __init__(self, stats_file):
        with open(stats_file, 'r') as f:
            self.stats = json.load(f)
            
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Dino AI Training Visualization")
        self.font = pygame.font.Font(None, 36)
        
    def draw_graph(self):
        
        self.screen.fill(COLORS['WHITE'])
        
        
        pygame.draw.line(self.screen, COLORS['BLACK'], 
                        (50, WINDOW_HEIGHT - 50), 
                        (WINDOW_WIDTH - 50, WINDOW_HEIGHT - 50), 2)
        pygame.draw.line(self.screen, COLORS['BLACK'], 
                        (50, WINDOW_HEIGHT - 50), 
                        (50, 50), 2)
        
        
        scores = self.stats['scores']
        mean_scores = self.stats['mean_scores']
        
        if len(scores) > 1:
            max_score = max(max(scores), max(mean_scores))
            scale_x = (WINDOW_WIDTH - 100) / len(scores)
            scale_y = (WINDOW_HEIGHT - 100) / max_score
            
            
            points = [(50 + i * scale_x, WINDOW_HEIGHT - 50 - score * scale_y) 
                     for i, score in enumerate(scores)]
            if len(points) > 1:
                pygame.draw.lines(self.screen, COLORS['BLUE'], False, points, 2)
                
            
            mean_points = [(50 + i * scale_x, WINDOW_HEIGHT - 50 - score * scale_y) 
                          for i, score in enumerate(mean_scores)]
            if len(mean_points) > 1:
                pygame.draw.lines(self.screen, COLORS['RED'], False, mean_points, 2)
        
        
        stats_text = [
            f"Games: {self.stats['n_games']}",
            f"Record: {self.stats['record']}",
            f"Mean Score: {self.stats['mean_scores'][-1]:.2f}"
        ]
        
        for i, text in enumerate(stats_text):
            surface = self.font.render(text, True, COLORS['BLACK'])
            self.screen.blit(surface, (WINDOW_WIDTH - 300, 50 + i * 40))
            
        pygame.display.flip()
        
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            self.draw_graph()
            pygame.time.wait(100)
            
        pygame.quit()

def main():
    
    for directory in ["models", "statistics"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    
    trainer = DinoTrainer()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        
        trainer._save_statistics()
        
        
        latest_stats = max(os.listdir("statistics"), 
                          key=lambda x: int(x.split('_')[1].split('.')[0]))
        visualizer = DinoVisualizer(f"statistics/{latest_stats}")
        visualizer.run()

if __name__ == "__main__":
    main()
    
        