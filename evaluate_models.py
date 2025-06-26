#!/usr/bin/env python3
"""
Model Evaluation Script
=======================

A flexible script to evaluate two trained models on various Gymnasium environments.
Supports both DQN and DDQN models for LunarLander and CarRacing environments.

Usage:
    python evaluate_models.py --model1 path/to/model1.pth --model2 path/to/model2.pth --env LunarLander-v2
    python evaluate_models.py --model1 path/to/model1.pth --model2 path/to/model2.pth --env CarRacing-v2
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import gymnasium as gym
import cv2
from collections import deque
from datetime import datetime

# Add project paths to sys.path
sys.path.append(
    os.path.join(os.path.dirname(__file__), 'lunar_lander', 'dqn')
)
sys.path.append(
    os.path.join(os.path.dirname(__file__), 'lunar_lander', 'ddqn')
)

# Import model classes
try:
    from dqn_lunar_lander import DQN as LunarLanderDQN
except ImportError:
    LunarLanderDQN = None

try:
    from lunar_lander.ddqn.ddqn_lunar_lander import DQN as LunarLanderDDQN
except ImportError:
    LunarLanderDDQN = None

# Set device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

class LunarLanderDQN3Layer(torch.nn.Module):
    """DQN network for LunarLander with 3 layers (128, 128, 4) - matches saved model"""
    def __init__(self, input_size, output_size):
        super(LunarLanderDQN3Layer, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

class LunarLanderDQN3Layer256(torch.nn.Module):
    """DQN network for LunarLander with 3 layers (256, 256, 4) - matches newer saved model"""
    def __init__(self, input_size, output_size):
        super(LunarLanderDQN3Layer256, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, output_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

class CarRacingDQN(torch.nn.Module):
    """DQN network for CarRacing environment"""
    def __init__(self, in_dim, out_dim):
        super(CarRacingDQN, self).__init__()
        channel_n, height, width = in_dim
        
        if height != 84 or width != 84:
            raise ValueError(
                f"CarRacing DQN model requires input of (84, 84)-shape. "
                f"Got ({height}, {width})"
            )
        
        # Updated architecture to match the saved models
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channel_n, out_channels=16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2592, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)

class CarRacingWrapper:
    """Wrapper for CarRacing environment with preprocessing"""
    def __init__(self, env, skip_frames=4, stack_frames=4):
        self.env = env
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.frame_stack = deque(maxlen=stack_frames)
        
    def reset(self):
        obs, info = self.env.reset()
        # Initialize frame stack
        for _ in range(self.stack_frames):
            self.frame_stack.append(obs)
        return self._get_observation(), info
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip_frames):
            # For CarRacing-v3 with continuous=False, action is already discrete
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        
        self.frame_stack.append(obs)
        return self._get_observation(), total_reward, terminated, truncated, info
    
    def _get_observation(self):
        # Convert to grayscale and resize
        frames = []
        for frame in self.frame_stack:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            frames.append(resized)
        
        # Stack frames (no normalization, keep as uint8)
        stacked = np.stack(frames, axis=0)
        return stacked
    
    def close(self):
        """Close the environment"""
        if hasattr(self.env, 'close'):
            self.env.close()

class ModelEvaluator:
    """Main evaluation class"""
    
    def __init__(self, env_name: str, num_episodes: int = 10, render: bool = False, 
                 record_episodes: bool = False, record_dir: str = "episode_recordings",
                 fps: int = 30):
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.render = render
        self.record_episodes = record_episodes
        self.record_dir = record_dir
        self.fps = fps
        self.results = {}
        
        # Create recording directory if needed
        if self.record_episodes:
            os.makedirs(self.record_dir, exist_ok=True)
        
        # Environment setup
        if env_name == "LunarLander-v2":
            self.env = gym.make(env_name, render_mode='human' if render else None)
            self.state_size = self.env.observation_space.shape[0]
            self.action_size = self.env.action_space.n
            self.is_image_env = False
        elif env_name == "CarRacing-v2":
            # Use CarRacing-v2 with discrete actions to match training
            self.env = CarRacingWrapper(
                gym.make("CarRacing-v2", continuous=False, render_mode='human' if render else None)
            )
            self.state_size = (4, 84, 84)  # 4 stacked grayscale frames
            self.action_size = 5  # Discrete actions: [noop, brake, gas, left, right]
            self.is_image_env = True
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
    
    def load_model(self, model_path: str, model_name: str) -> torch.nn.Module:
        """Load a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading {model_name} from: {model_path}")
        
        # Load model weights with PyTorch 2.6 compatibility
        try:
            # First try with weights_only=True (safer)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e:
            if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
                # If weights_only fails, try with weights_only=False (for older checkpoints)
                print(f"  - Using weights_only=False for compatibility")
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            else:
                raise e
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check if it's a direct state dict (like the DQN model)
            if all(key.startswith(('fc', 'conv', 'net.')) for key in checkpoint.keys()):
                # Direct state dict
                state_dict = checkpoint
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'upd_model_state_dict' in checkpoint:
                state_dict = checkpoint['upd_model_state_dict']
            elif 'frz_model_state_dict' in checkpoint:
                state_dict = checkpoint['frz_model_state_dict']
            else:
                # Try to find any key that contains 'state_dict'
                state_dict_keys = [k for k in checkpoint.keys() if 'state_dict' in k.lower()]
                if state_dict_keys:
                    state_dict = checkpoint[state_dict_keys[0]]
                else:
                    raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
        else:
            # Direct state dict
            state_dict = checkpoint
        
        # Handle state dict keys for different model formats
        if self.env_name == "CarRacing-v2":
            # For CarRacing models, check if keys need the 'net.' prefix
            if any(key.isdigit() for key in state_dict.keys()):
                # Keys are like "0.weight", "2.weight" - need to add "net." prefix
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_state_dict[f"net.{key}"] = value
                state_dict = new_state_dict
        else:
            # For other environments, handle "net." prefix removal if present
            if any(key.startswith('net.') for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('net.'):
                        new_state_dict[key[4:]] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
        
        # Create appropriate model based on environment and architecture
        if self.env_name == "LunarLander-v2":
            if LunarLanderDQN is None:
                raise ImportError("LunarLander DQN model not available")
            # Detect architecture from state dict by checking layer sizes
            # 3-layer (128): fc1.weight [128, 8], no fc4.weight
            # 3-layer (256): fc1.weight [256, 8], no fc4.weight  
            # 4-layer: fc1.weight [512, 8], fc4.weight exists
            if 'fc1.weight' in state_dict:
                fc1_shape = state_dict['fc1.weight'].shape
                if fc1_shape[0] == 128 and 'fc4.weight' not in state_dict:
                    print(f"  - Detected DQN architecture (3 layers: 128->128->4)")
                    model = LunarLanderDQN3Layer(self.state_size, self.action_size).to(device)
                elif fc1_shape[0] == 256 and 'fc4.weight' not in state_dict:
                    print(f"  - Detected DQN architecture (3 layers: 256->256->4)")
                    model = LunarLanderDQN3Layer256(self.state_size, self.action_size).to(device)
                elif fc1_shape[0] == 512 and 'fc4.weight' in state_dict:
                    print(f"  - Detected DQN/DDQN architecture (4 layers: 512->512->256->4)")
                    model = LunarLanderDQN(self.state_size, self.action_size).to(device)
                else:
                    raise ValueError(f"Unknown architecture: fc1 has {fc1_shape[0]} outputs, fc4 present: {'fc4.weight' in state_dict}")
            else:
                raise ValueError("Could not detect architecture: fc1.weight not found in state dict")
        elif self.env_name == "CarRacing-v2":
            model = CarRacingDQN(self.state_size, self.action_size).to(device)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Print model info if available
        if isinstance(checkpoint, dict) and 'episode' in checkpoint:
            print(f"  - Trained for {checkpoint['episode']} episodes")
            print(f"  - Best reward: {checkpoint.get('reward', 'N/A')}")
            print(f"  - Average reward: {checkpoint.get('avg_reward', 'N/A')}")
        
        return model
    
    def record_episode(self, model: torch.nn.Module, model_name: str, 
                      episode_num: int, reward: float, is_best: bool = False, 
                      is_worst: bool = False) -> None:
        """Record an episode and save it as an MP4 video file"""
        if not self.record_episodes:
            return
        
        # Create a recording environment
        if self.env_name == "LunarLander-v2":
            record_env = gym.make(self.env_name, render_mode='rgb_array')
        elif self.env_name == "CarRacing-v2":
            record_env = CarRacingWrapper(
                gym.make("CarRacing-v2", continuous=False, render_mode='rgb_array')
            )
        
        # Determine filename with environment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_short = "LunarLander" if self.env_name == "LunarLander-v2" else "CarRacing"
        
        if is_best:
            filename = f"{model_name}_{env_short}_best_episode_{episode_num}_reward_{reward:.1f}_{timestamp}.mp4"
        elif is_worst:
            filename = f"{model_name}_{env_short}_worst_episode_{episode_num}_reward_{reward:.1f}_{timestamp}.mp4"
        else:
            filename = f"{model_name}_{env_short}_episode_{episode_num}_reward_{reward:.1f}_{timestamp}.mp4"
        
        filepath = os.path.join(self.record_dir, filename)
        
        # Get video dimensions
        state, _ = record_env.reset()
        if self.is_image_env:
            # For CarRacing, we need to get the original RGB frame
            frame = record_env.env.render()
            height, width = frame.shape[:2]
        else:
            # For LunarLander, render a frame to get dimensions
            frame = record_env.render()
            height, width = frame.shape[:2]
        
        # Initialize video writer with H264 codec (more compatible)
        # Note: For CarRacing, you can use a lower FPS (e.g., 15 or 10) to slow down the recording
        # by setting fps=10 or fps=15 when creating the ModelEvaluator
        try:
            # Try H264 codec first (most compatible)
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
        except Exception:
            try:
                # Fallback to mp4v if H264 fails
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
            except Exception:
                # Final fallback to XVID
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
        
        # Record the episode
        state, _ = record_env.reset()
        done = False
        
        while not done:
            # Select action using the model
            with torch.no_grad():
                if self.is_image_env:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            # Take action and get frame
            next_state, reward, terminated, truncated, _ = record_env.step(action)
            done = terminated or truncated
            
            # Get the frame
            if self.is_image_env:
                frame = record_env.env.render()
            else:
                frame = record_env.render()
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            state = next_state
        
        out.release()
        record_env.close()
        
        print(f"  📹 Saved recording: {filename}")
    
    def evaluate_model(self, model: torch.nn.Module, model_name: str) -> Dict:
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")
        
        episode_rewards = []
        episode_lengths = []
        best_episode = None
        worst_episode = None
        best_reward = float('-inf')
        worst_reward = float('inf')
        
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Select action using the model
                with torch.no_grad():
                    if self.is_image_env:
                        # For image-based environments
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    else:
                        # For vector-based environments
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
                
                # Take action in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                total_reward += reward
                episode_length += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            # Track best and worst episodes
            if total_reward > best_reward:
                best_reward = total_reward
                best_episode = episode
            if total_reward < worst_reward:
                worst_reward = total_reward
                worst_episode = episode
            
            print(f"  Episode {episode + 1}: Reward = {total_reward:.2f}, Length = {episode_length}")
        
        # Record best and worst episodes (only at the end)
        if self.record_episodes and best_episode is not None:
            self.record_episode(model, model_name, best_episode + 1, 
                              episode_rewards[best_episode], is_best=True)
        if self.record_episodes and worst_episode is not None:
            self.record_episode(model, model_name, worst_episode + 1, 
                              episode_rewards[worst_episode], is_worst=True)
        
        # Calculate statistics
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'best_episode': best_episode + 1 if best_episode is not None else None,
            'worst_episode': worst_episode + 1 if worst_episode is not None else None
        }
        
        print(f"  Results for {model_name}:")
        print(f"    Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"    Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
        print(f"    Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        if best_episode is not None:
            print(f"    Best Episode: {best_episode + 1} (Reward: {best_reward:.2f})")
        if worst_episode is not None:
            print(f"    Worst Episode: {worst_episode + 1} (Reward: {worst_reward:.2f})")
        
        return results
    
    def compare_models(self, model1_path: str, model2_path: str, 
                      model1_name: str = "Model 1", model2_name: str = "Model 2"):
        """Compare two models"""
        print(f"\n{'='*60}")
        print(f"COMPARING MODELS ON {self.env_name}")
        print(f"{'='*60}")
        
        # Load and evaluate both models
        model1 = self.load_model(model1_path, model1_name)
        model2 = self.load_model(model2_path, model2_name)
        
        results1 = self.evaluate_model(model1, model1_name)
        results2 = self.evaluate_model(model2, model2_name)
        
        self.results = {
            model1_name: results1,
            model2_name: results2
        }
        
        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {model1_name:<15} {model2_name:<15} {'Difference':<15}")
        print("-" * 65)
        print(f"{'Mean Reward':<20} {results1['mean_reward']:<15.2f} {results2['mean_reward']:<15.2f} {results2['mean_reward'] - results1['mean_reward']:<15.2f}")
        print(f"{'Std Reward':<20} {results1['std_reward']:<15.2f} {results2['std_reward']:<15.2f} {results2['std_reward'] - results1['std_reward']:<15.2f}")
        print(f"{'Min Reward':<20} {results1['min_reward']:<15.2f} {results2['min_reward']:<15.2f} {results2['min_reward'] - results1['min_reward']:<15.2f}")
        print(f"{'Max Reward':<20} {results1['max_reward']:<15.2f} {results2['max_reward']:<15.2f} {results2['max_reward'] - results1['max_reward']:<15.2f}")
        print(f"{'Mean Length':<20} {results1['mean_length']:<15.1f} {results2['mean_length']:<15.1f} {results2['mean_length'] - results1['mean_length']:<15.1f}")
        
        # Determine winner
        if results1['mean_reward'] > results2['mean_reward']:
            winner = model1_name
            margin = results1['mean_reward'] - results2['mean_reward']
        elif results2['mean_reward'] > results1['mean_reward']:
            winner = model2_name
            margin = results2['mean_reward'] - results1['mean_reward']
        else:
            winner = "Tie"
            margin = 0
        
        print(f"\n🏆 Winner: {winner}")
        if winner != "Tie":
            print(f"   Margin: {margin:.2f} points")
        
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot comparison results"""
        if not self.results:
            print("No results to plot. Run compare_models() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Comparison on {self.env_name}', fontsize=16)
        
        model_names = list(self.results.keys())
        colors = ['#1f77b4', '#ff7f0e']
        
        # Reward comparison
        ax1 = axes[0, 0]
        rewards = [self.results[name]['episode_rewards'] for name in model_names]
        bp1 = ax1.boxplot(rewards, labels=model_names, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        ax1.set_title('Episode Rewards Distribution')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Episode length comparison
        ax2 = axes[0, 1]
        lengths = [self.results[name]['episode_lengths'] for name in model_names]
        bp2 = ax2.boxplot(lengths, labels=model_names, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        ax2.set_title('Episode Lengths Distribution')
        ax2.set_ylabel('Length')
        ax2.grid(True, alpha=0.3)
        
        # Reward progression
        ax3 = axes[1, 0]
        for i, name in enumerate(model_names):
            ax3.plot(self.results[name]['episode_rewards'], 
                    label=name, color=colors[i], marker='o', alpha=0.7)
        ax3.set_title('Reward Progression Across Episodes')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Statistical summary
        ax4 = axes[1, 1]
        metrics = ['mean_reward', 'std_reward', 'min_reward', 'max_reward']
        metric_labels = ['Mean', 'Std', 'Min', 'Max']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, name in enumerate(model_names):
            values = [self.results[name][metric] for metric in metrics]
            ax4.bar(x + i*width, values, width, label=name, color=colors[i], alpha=0.7)
        
        ax4.set_title('Reward Statistics')
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Reward')
        ax4.set_xticks(x + width/2)
        ax4.set_xticklabels(metric_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def close(self):
        """Close the environment"""
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare two trained models')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model')
    parser.add_argument('--env', type=str, required=True, 
                       choices=['LunarLander-v2', 'CarRacing-v2'], 
                       help='Environment to evaluate on')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes per model')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--name1', type=str, default='Model 1', help='Name for first model')
    parser.add_argument('--name2', type=str, default='Model 2', help='Name for second model')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--save-plot', type=str, help='Path to save the comparison plot')
    parser.add_argument('--record', action='store_true', help='Record best and worst episodes')
    parser.add_argument('--record-dir', type=str, default='episode_recordings', 
                       help='Directory to save episode recordings')
    parser.add_argument('--fps', type=int, default=30, 
                       help='FPS for video recordings (default: 30). Use lower FPS (e.g., 10-15) to slow down CarRacing recordings.')
    
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(
            env_name=args.env,
            num_episodes=args.episodes,
            render=args.render,
            record_episodes=args.record,
            record_dir=args.record_dir,
            fps=args.fps
        )
        
        # Compare models
        results = evaluator.compare_models(
            model1_path=args.model1,
            model2_path=args.model2,
            model1_name=args.name1,
            model2_name=args.name2
        )
        
        # Generate plots if requested
        if args.plot or args.save_plot:
            evaluator.plot_results(save_path=args.save_plot)
        
        evaluator.close()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 