import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np

# Set device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Larger network
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_size)
        
        # Initialize weights for better training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(state).to(device),
            torch.LongTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).to(device)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Improved hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Slower decay
        self.learning_rate = 0.0001  # Lower learning rate
        self.batch_size = 64  # Smaller batch for stability
        self.memory = ReplayBuffer(200000)  # Larger buffer
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer with weight decay
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5  # L2 regularization
        )

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from replay buffer
        state, action, reward, next_state, done = self.memory.sample(
            self.batch_size
        )

        # Double DQN: Use policy net for action selection, target net for evaluation
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))

        with torch.no_grad():
            # Use policy net to select actions
            next_actions = self.policy_net(next_state).argmax(1)
            # Use target net to evaluate actions
            next_q_values = self.target_net(next_state).gather(1, next_actions.unsqueeze(1))
            target_q_values = reward + (1 - done) * self.gamma * next_q_values.squeeze(1)

        # Compute loss with gradient clipping
        loss = F.smooth_l1_loss(
            current_q_values.squeeze(1), target_q_values
        )
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_dqn():
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    episodes = 4000  # More episodes
    target_update = 50  # More frequent updates
    best_reward = float('-inf')
    
    # Performance tracking
    recent_rewards = []
    
    # Improved model saving
    save_interval = 500  # Save every 500 episodes
    best_models = []  # Keep track of top 3 models
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
        
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Update epsilon once per episode
        agent.epsilon = max(
            agent.epsilon_min, 
            agent.epsilon * agent.epsilon_decay
        )
        
        # Track recent performance
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        # Improved model saving strategy
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        # Save best model (highest single episode reward)
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                'model_state_dict': agent.policy_net.state_dict(),
                'episode': episode,
                'reward': total_reward,
                'epsilon': agent.epsilon,
                'avg_reward': avg_reward
            }, 'best_model.pth')
            print(f"🏆 New best model saved! Reward: {total_reward:.2f}")
        
        # Save checkpoint every 500 episodes
        if episode % save_interval == 0:
            checkpoint_path = f'checkpoint_episode_{episode}.pth'
            torch.save({
                'model_state_dict': agent.policy_net.state_dict(),
                'episode': episode,
                'reward': total_reward,
                'epsilon': agent.epsilon,
                'avg_reward': avg_reward,
                'best_reward': best_reward
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save top 3 models based on average reward (if we have enough data)
        if len(recent_rewards) >= 50:
            best_models.append({
                'episode': episode,
                'avg_reward': avg_reward,
                'single_reward': total_reward,
                'epsilon': agent.epsilon
            })
            # Keep only top 3
            best_models.sort(key=lambda x: x['avg_reward'], reverse=True)
            best_models = best_models[:3]
            
            # Save the best average model
            if best_models[0]['episode'] == episode:
                torch.save({
                    'model_state_dict': agent.policy_net.state_dict(),
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'single_reward': total_reward,
                    'epsilon': agent.epsilon
                }, 'best_avg_model.pth')
                print(f"📊 Best average model saved! Avg: {avg_reward:.2f}")
        
        # Print progress with average reward
        if episode % 50 == 0:
            print(
                f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.2f}, Best: {best_reward:.2f}"
            )
        else:
            print(
                f"Episode: {episode}, Total Reward: {total_reward}, "
                f"Epsilon: {agent.epsilon:.2f}"
            )
    
    # Save final model
    torch.save({
        'model_state_dict': agent.policy_net.state_dict(),
        'episode': episode,
        'reward': total_reward,
        'epsilon': agent.epsilon,
        'avg_reward': avg_reward,
        'best_reward': best_reward
    }, 'final_model.pth')
    
    print(f"\n🎯 Training Complete!")
    print(f"Best single episode reward: {best_reward:.2f}")
    print(f"Final average reward: {avg_reward:.2f}")
    print(f"Top 3 average models: {best_models}")
    
    env.close()

if __name__ == "__main__":
    train_dqn() 