# Model Evaluation Script

This script allows you to evaluate and compare two trained models on various Gymnasium environments. It supports both DQN and DDQN models for LunarLander and CarRacing environments.

## Features

- **Multi-environment support**: LunarLander-v2 and CarRacing-v2
- **Flexible model loading**: Handles different checkpoint formats
- **Comprehensive evaluation**: Episode rewards, lengths, and statistics
- **Visual comparison**: Generate plots comparing model performance
- **Episode recording**: Save best and worst episodes as WebM video files
- **Customizable**: Set custom model names, number of episodes, and rendering options

## Usage

### Basic Usage

```bash
# Compare two models on LunarLander
python evaluate_models.py --model1 path/to/model1.pth --model2 path/to/model2.pth --env LunarLander-v2

# Compare two models on CarRacing
python evaluate_models.py --model1 path/to/model1.pth --model2 path/to/model2.pth --env CarRacing-v2
```

### Advanced Usage

```bash
# With custom names and more episodes
python evaluate_models.py \
    --model1 lunar_lander/dqn/model.pth \
    --model2 lunar_lander/ddqn/final_model.pth \
    --env LunarLander-v2 \
    --name1 "DQN" \
    --name2 "DDQN" \
    --episodes 20 \
    --render \
    --plot \
    --save-plot comparison_results.png
```

### Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model1` | str | Yes | - | Path to first model file |
| `--model2` | str | Yes | - | Path to second model file |
| `--env` | str | Yes | - | Environment name (LunarLander-v2 or CarRacing-v2) |
| `--episodes` | int | No | 10 | Number of episodes to evaluate per model |
| `--render` | flag | No | False | Render the environment during evaluation |
| `--name1` | str | No | "Model 1" | Custom name for first model |
| `--name2` | str | No | "Model 2" | Custom name for second model |
| `--plot` | flag | No | False | Generate comparison plots |
| `--save-plot` | str | No | - | Path to save the comparison plot |
| `--record` | flag | No | False | Record best and worst episodes as videos |
| `--record-dir` | str | No | "episode_recordings" | Directory to save episode recordings |
| `--fps` | int | No | 30 | FPS for video recordings |

## Examples

### Example 1: Compare DQN vs DDQN on LunarLander

```bash
python evaluate_models.py \
    --model1 lunar_lander/dqn/model.pth \
    --model2 lunar_lander/ddqn/final_model.pth \
    --env LunarLander-v2 \
    --name1 "DQN" \
    --name2 "DDQN" \
    --episodes 15 \
    --plot
```

### Example 2: Compare CarRacing models with rendering

```bash
python evaluate_models.py \
    --model1 Car_racing/training/saved_models/DQN_740863.pt \
    --model2 Car_racing/training/saved_models/DDQN_743266.pt \
    --env CarRacing-v2 \
    --name1 "DQN CarRacing" \
    --name2 "DDQN CarRacing" \
    --episodes 5 \
    --render
```

### Example 3: Save results to file

```bash
python evaluate_models.py \
    --model1 lunar_lander/dqn/model.pth \
    --model2 lunar_lander/ddqn/final_model.pth \
    --env LunarLander-v2 \
    --name1 "DQN" \
    --name2 "DDQN" \
    --episodes 20 \
    --plot \
    --save-plot lunar_lander_comparison.png
```

### Example 4: Record best and worst episodes

```bash
python evaluate_models.py \
    --model1 lunar_lander/dqn/model.pth \
    --model2 lunar_lander/ddqn/final_model.pth \
    --env LunarLander-v2 \
    --name1 "DQN" \
    --name2 "DDQN" \
    --episodes 15 \
    --record \
    --record-dir my_recordings
```

### Example 5: Record with custom FPS

```bash
python evaluate_models.py \
    --model1 lunar_lander/dqn/model.pth \
    --model2 lunar_lander/ddqn/final_model.pth \
    --env LunarLander-v2 \
    --name1 "DQN" \
    --name2 "DDQN" \
    --episodes 10 \
    --record \
    --fps 60 \
    --record-dir high_fps_recordings
```

## Output

The script provides:

1. **Model Loading Information**: Shows which models are being loaded and their training metadata
2. **Episode-by-Episode Results**: Individual episode rewards and lengths for each model
3. **Statistical Summary**: Mean, standard deviation, min, and max rewards for each model
4. **Comparison Table**: Side-by-side comparison of all metrics
5. **Winner Declaration**: Which model performed better and by what margin
6. **Best/Worst Episode Tracking**: Identifies and reports the best and worst episodes for each model
7. **Episode Recordings** (if `--record` is used): WebM video files of the best and worst episodes
8. **Visual Plots** (if `--plot` is used):
   - Episode rewards distribution (box plots)
   - Episode lengths distribution (box plots)
   - Reward progression across episodes (line plots)
   - Statistical summary (bar charts)

## Supported Model Formats

The script can handle various checkpoint formats:

- **Direct state dict**: `torch.save(model.state_dict(), 'model.pth')`
- **Checkpoint with metadata**: 
  ```python
  torch.save({
      'model_state_dict': model.state_dict(),
      'episode': episode,
      'reward': reward,
      'avg_reward': avg_reward
  }, 'model.pth')
  ```
- **CarRacing format**: `torch.save({'upd_model_state_dict': model.state_dict()}, 'model.pth')`

## Environment Support

### LunarLander-v2
- **State space**: 8-dimensional vector
- **Action space**: 4 discrete actions
- **Model architecture**: Fully connected neural network
- **Supported models**: DQN, DDQN

### CarRacing-v2
- **State space**: 4 stacked grayscale frames (84x84)
- **Action space**: 5 discrete actions
- **Model architecture**: Convolutional neural network
- **Supported models**: DQN, DDQN

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install torch gymnasium matplotlib numpy opencv-python
```

## Troubleshooting

### Common Issues

1. **Import Error for LunarLander models**: Make sure the `lunar_lander` directory is in your Python path
2. **Model loading errors**: Check that the model file exists and is compatible with the environment
3. **Environment errors**: Ensure you have the correct Gymnasium version installed
4. **Plotting errors**: Make sure matplotlib is properly installed and configured

### Getting Help

If you encounter issues:

1. Check that all model paths are correct
2. Verify that the environment name matches exactly
3. Ensure all dependencies are installed
4. Try running with `--render` to see if the environment loads correctly

## Example Output

```
============================================================
COMPARING MODELS ON LunarLander-v2
============================================================
Loading DQN from: lunar_lander/dqn/model.pth
Loading DDQN from: lunar_lander/ddqn/final_model.pth
  - Trained for 1000 episodes
  - Best reward: 250.45
  - Average reward: 200.12

Evaluating DQN...
  Episode 1: Reward = 180.25, Length = 450
  Episode 2: Reward = 195.67, Length = 520
  ...

Evaluating DDQN...
  Episode 1: Reward = 220.45, Length = 480
  Episode 2: Reward = 235.12, Length = 510
  ...

============================================================
COMPARISON SUMMARY
============================================================
Metric               DQN            DDQN           Difference      
-----------------------------------------------------------------
Mean Reward          185.45         225.67         40.22          
Std Reward           15.23          12.45          -2.78          
Min Reward           160.12         200.34         40.22          
Max Reward           210.78         250.90         40.12          
Mean Length           485.2          495.6          10.4           

🏆 Winner: DDQN
   Margin: 40.22 points
``` 