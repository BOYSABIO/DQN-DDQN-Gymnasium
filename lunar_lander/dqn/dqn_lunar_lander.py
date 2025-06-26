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

# ───────────────────────── Device ─────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

# ─────────────────────── Network (DQN) ────────────────────
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ──────────────────── Replay Buffer ───────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch):
        s, a, r, s2, d = zip(*random.sample(self.buffer, batch))
        return (torch.FloatTensor(s).to(device),
                torch.LongTensor(a).to(device),
                torch.FloatTensor(r).to(device),
                torch.FloatTensor(s2).to(device),
                torch.FloatTensor(d).to(device))

    def __len__(self):
        return len(self.buffer)

# ───────────────────────── Agent ──────────────────────────
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.gamma = 0.99
        self.epsilon, self.eps_min, self.eps_decay = 1.0, 0.01, 0.9995
        self.lr, self.batch = 1e-4, 128

        self.policy_net = DQN(state_size, action_size).to(device)
        self.optim = optim.Adam(self.policy_net.parameters(), lr=self.lr,
                                weight_decay=1e-5)

        self.memory = ReplayBuffer(200000)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.policy_net.fc3.out_features)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.policy_net(state).argmax().item()

    def train(self):
        if len(self.memory) < self.batch:
            return 0.0
        s, a, r, s2, d = self.memory.sample(self.batch)

        q_curr = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # ─── Vanilla DQN TD target ───
            q_next = self.policy_net(s2).max(1)[0]
            q_targ = r + (1 - d) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_curr, q_targ)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optim.step()
        return loss.item()

# ──────────────────── Helper: CSV log ─────────────────────
def write_log(dates, times, rewards, lengths, losses, epsilons,
              fname='DQN_training_log.csv'):
    os.makedirs('logs', exist_ok=True)
    rows = [['date']   + dates,
            ['time']   + times,
            ['reward'] + rewards,
            ['length'] + lengths,
            ['loss']   + losses,
            ['epsilon']+ epsilons]
    path = os.path.join('logs', fname)
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f"📊 Log saved to {path}")

# ──────────────────── Helper: save model ──────────────────
def save_model(agent, ep, rew, eps, avg, best, name):
    os.makedirs('models', exist_ok=True)
    path = os.path.join('models', f"{name}.pth")
    torch.save({'model_state_dict': agent.policy_net.state_dict(),
                'episode': ep, 'reward': rew, 'epsilon': eps,
                'avg_reward': avg, 'best_reward': best}, path)
    print(f"💾 Model saved → {path}")

# ──────────────────── Helper: plot training progress ─────

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


# ───────────────────── Training Loop ──────────────────────
def train_dqn():
    env = gym.make('LunarLander-v2')
    state_size, action_size = env.observation_space.shape[0], env.action_space.n
    agent = DQNAgent(state_size, action_size)

    EPISODES, SAVE_INT, BEST_REW = 5000, 500, -float('inf')
    rec_rewards, best_models = [], []

    # logging containers
    d_list, t_list, r_list, l_list, loss_list, eps_list = [],[],[],[],[],[]

    print("🚀 Starting **DQN** training …")
    for ep in range(EPISODES):
        s, _ = env.reset()
        tot_r, ep_len, ep_losses = 0, 0, []
        done = False
        while not done:
            a = agent.select_action(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.memory.push(s, a, r, s2, done)
            loss = agent.train()
            if loss: ep_losses.append(loss)
            s, tot_r, ep_len = s2, tot_r + r, ep_len + 1

        # epsilon decay
        agent.epsilon = max(agent.eps_min, agent.epsilon * agent.eps_decay)

        rec_rewards.append(tot_r)
        if len(rec_rewards) > 100: rec_rewards.pop(0)
        avg_r = np.mean(rec_rewards)
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0

        # ─── logging lists ───
        now = datetime.datetime.now()
        d_list.append(now.strftime('%Y-%m-%d'))
        t_list.append(now.strftime('%H:%M:%S'))
        r_list.append(str(tot_r))
        l_list.append(str(ep_len))
        loss_list.append(str(avg_loss))
        eps_list.append(str(round(agent.epsilon,4)))

        # ─── best / checkpoints ───
        if tot_r > BEST_REW:
            BEST_REW = tot_r
            save_model(agent, ep, tot_r, agent.epsilon, avg_r, BEST_REW,
                       "DQN_best_model")

        if ep % SAVE_INT == 0:
            save_model(agent, ep, tot_r, agent.epsilon, avg_r, BEST_REW,
                       f"DQN_checkpoint_ep_{ep}")

        # top-3 by avg reward
        if len(rec_rewards) >= 50:
            best_models.append({'ep': ep,'avg': avg_r,'single': tot_r})
            best_models = sorted(best_models, key=lambda x: x['avg'],
                                 reverse=True)[:3]
            if best_models[0]['ep'] == ep:
                save_model(agent, ep, tot_r, agent.epsilon, avg_r, BEST_REW,
                           "DQN_best_avg_model")

        # ─── console prints ───
        if ep % 50 == 0:
            print(f"Episode {ep:4d} | Avg100 {avg_r:7.2f} | "
                  f"Epsilon {agent.epsilon:.3f} | Best {BEST_REW:7.2f}")
        else:
            print(f"Episode {ep:4d} | Reward {tot_r:7.2f} | "
                  f"Epsilon {agent.epsilon:.3f}")

        # write CSV every 10 eps
        if ep % 10 == 0:
            write_log(d_list, t_list, r_list, l_list, loss_list, eps_list)

    # final save & plot
    save_model(agent, EPISODES, tot_r, agent.epsilon, avg_r, BEST_REW,
               "DQN_final_model")
    write_log(d_list, t_list, r_list, l_list, loss_list, eps_list,
              'DQN_final_log.csv')

    log_path = os.path.join('logs', 'DQN_final_log.csv')
    if os.path.exists(log_path):
        print("\n📈 Plotting training progress …")
        plot_training_progress(log_path)

    env.close()

# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_dqn()
