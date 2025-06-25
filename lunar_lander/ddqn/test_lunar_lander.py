import gymnasium as gym
import torch
from dqn_lunar_lander import DQN, device

def test_model():
    # Create environment with rendering
    env = gym.make('LunarLander-v2', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create and load the model
    model = DQN(state_size, action_size).to(device)
    
    # Load the saved model (which includes metadata)
    checkpoint = torch.load('final_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Print model info
    print(f"Loaded model from episode {checkpoint['episode']}")
    print(f"Model was saved with reward: {checkpoint['reward']:.2f}")
    print(f"Average reward at save time: {checkpoint['avg_reward']:.2f}")
    print(f"Epsilon at save time: {checkpoint['epsilon']:.4f}")
    
    # Run 10 test episodes
    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action using the model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
        
        print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    test_model() 