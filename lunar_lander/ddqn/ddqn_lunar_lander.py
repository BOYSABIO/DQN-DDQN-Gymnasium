import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import os
import csv
import datetime
import matplotlib.pyplot as plt
import pandas as pd

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
            return 0.0  # Return loss of 0 if not enough samples

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
            next_q_values = self.target_net(next_state).gather(
                1, next_actions.unsqueeze(1)
            )
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
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def plot_training_progress(csv_path, window_size=50):
    """Plot training progress from CSV log file"""
    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract episode times (skip label "time")
    times = df.iloc[0, 1:].values
    time_fmt = "%H:%M:%S"
    start_time = datetime.datetime.strptime(times[0], time_fmt)
    end_time = datetime.datetime.strptime(times[-1], time_fmt)
    if end_time < start_time:  # handle wrap-around midnight
        end_time = end_time.replace(day=start_time.day + 1)
    duration = end_time - start_time

    # Extract reward values (skip label "reward")
    rewards_str = df.iloc[1, 1:].values
    rewards = pd.to_numeric(rewards_str, errors='coerce')
    rewards_series = pd.Series(rewards)

    # Compute sliding window average
    rolling_avg = rewards_series.rolling(window=window_size, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(rewards_series, label="Episode Reward", alpha=0.4)
    plt.plot(rolling_avg, label=f"Moving Average (window={window_size})", 
             linewidth=2, color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"DDQN Training Progress - Lunar Lander\nTotal Training Time: {duration}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def write_log(date_list, time_list, reward_list, length_list, 
              loss_list, epsilon_list, 
              log_filename='DDQN_training_log.csv'):
    """Write training log to CSV file"""
    log_dir = './logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    rows = [['date'] + date_list,
            ['time'] + time_list,
            ['reward'] + reward_list,
            ['length'] + length_list,
            ['loss'] + loss_list,
            ['epsilon'] + epsilon_list]
    
    with open(os.path.join(log_dir, log_filename), 'w', 
              newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
    
    print(f"📊 Training log saved to {os.path.join(log_dir, log_filename)}")


def save_model(agent, episode, reward, epsilon, avg_reward, best_reward, 
               save_name=None):
    """Save model checkpoint"""
    save_dir = './models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if save_name is None:
        save_name = f"DDQN_episode_{episode}"
    
    save_path = os.path.join(save_dir, f"{save_name}.pth")
    torch.save({
        'model_state_dict': agent.policy_net.state_dict(),
        'episode': episode,
        'reward': reward,
        'epsilon': epsilon,
        'avg_reward': avg_reward,
        'best_reward': best_reward
    }, save_path)
    print(f"💾 Model saved to {save_path}")


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
    
    # Training logging variables
    episode_epsilon_list = []
    episode_reward_list = []
    episode_length_list = []
    episode_loss_list = []
    episode_date_list = []
    episode_time_list = []
    
    # Improved model saving
    save_interval = 500  # Save every 500 episodes
    best_models = []  # Keep track of top 3 models
    
    print("🚀 Starting DDQN training with comprehensive logging...")
    print(f"📈 Training for {episodes} episodes")
    print(f"🎯 Target network update every {target_update} episodes")
    print(f"💾 Model save interval: {save_interval} episodes")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        episode_losses = []
        done = False
        
        # Episode loop
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train()
            if loss > 0:  # Only add non-zero losses
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            episode_length += 1
        
        # Update target network
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
        
        # Calculate average reward
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        # Record episode statistics for logging
        episode_reward_list.append(str(total_reward))
        episode_length_list.append(str(episode_length))
        episode_loss_list.append(str(np.mean(episode_losses) 
                                    if episode_losses else 0))
        episode_epsilon_list.append(str(agent.epsilon))
        
        # Record timestamp
        now_time = datetime.datetime.now()
        episode_date_list.append(now_time.date().strftime('%Y-%m-%d'))
        episode_time_list.append(now_time.time().strftime('%H:%M:%S'))
        
        # Save best model (highest single episode reward)
        if total_reward > best_reward:
            best_reward = total_reward
            save_model(agent, episode, total_reward, agent.epsilon, 
                      avg_reward, best_reward, "DDQN_best_model")
            print(f"🏆 New best model saved! Reward: {total_reward:.2f}")
        
        # Save checkpoint every save_interval episodes
        if episode % save_interval == 0:
            save_model(agent, episode, total_reward, agent.epsilon, 
                      avg_reward, best_reward, 
                      f"DDQN_checkpoint_episode_{episode}")
            print(f"💾 Checkpoint saved: episode {episode}")
        
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
                save_model(agent, episode, total_reward, agent.epsilon, 
                          avg_reward, best_reward, "DDQN_best_avg_model")
                print(f"📊 Best average model saved! Avg: {avg_reward:.2f}")
        
        # Write logs every 10 episodes
        if episode % 10 == 0:
            write_log(episode_date_list, episode_time_list, 
                     episode_reward_list, episode_length_list, 
                     episode_loss_list, episode_epsilon_list,
                     'DDQN_training_log.csv')
        
        # Print progress with average reward
        if episode % 50 == 0:
            print(
                f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.2f}, Best: {best_reward:.2f}"
            )
        else:
            print(
                f"Episode: {episode}, Total Reward: {total_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.2f}"
            )
    
    # Final evaluation and save
    print(f"\n🎯 Training Complete!")
    print(f"Best single episode reward: {best_reward:.2f}")
    print(f"Final average reward: {avg_reward:.2f}")
    print(f"Top 3 average models: {best_models}")
    
    # Final save
    save_model(agent, episode, total_reward, agent.epsilon, 
               avg_reward, best_reward, "DDQN_final_model")
    write_log(episode_date_list, episode_time_list, episode_reward_list,
             episode_length_list, episode_loss_list, episode_epsilon_list,
             'DDQN_final_log.csv')
    
    # Plot training progress
    log_path = os.path.join('./logs/', 'DDQN_final_log.csv')
    if os.path.exists(log_path):
        print("\n📈 Plotting training progress...")
        plot_training_progress(log_path)
    
    env.close()
    
    return episode_reward_list, episode_length_list


if __name__ == "__main__":
    train_dqn() 