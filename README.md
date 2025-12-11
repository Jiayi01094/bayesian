# Plane War AI: Autonomous Game Control under Uncertainty

### Group Members
- Eddie Tian, Jiayi Wang, Sebastian De La Cruz

> **Project Goal:** To train a RL agent to master a chaotic Atari Asteroids like shooter game from scratch using only raw pixel input and learning to estimate value and risk from raw pixel input."

## Intro & Background
Video games like Plane War act as stochastic environments requiring quick reflexes, visual recognition, and implicit probability estimation. Our AI starts with zero knowledge:
- It doesn't know what a "plane" is;
- It doesn't know "bullets" represent a probability of death;
- It receives only a stream of noisy visual data and a sparse reward signal;
The Mission: Build an end-to-end Reinforcement Learning pipeline that evolves from random move to predictive dodging.

## The Environment ([plane_env.py](game/plane_env.py))
We utilized a custom Python-based game engine wrapped in a Gym-compatible interface for RL training.

**Input** 
- Raw screen pixels converted to 84x84 Grayscale; 

**State Space** 
- A stack of 4 consecutive frames (Shape: 4x84x84);
- A single frame is static, so we stack 4 frames so the agent can perceive velocity and bullet direction.

**Action Space (The Controls)**
- 5 discrete actions: UP, Down, Left, Right, Do Nothing
- a more simplified version than the atari Asteroids game.

**Reward**
- The agent controls a plane and receives immediate feedback to encourage survival and high scores while avoiding risk.

| Action / State | Reward Value |
| :--- | :--- |
| **Survival** (Per frame) | `+0.1` |
| **Dodge Enemy** (Enemy passes safely) | `+0.2` |
| **Kill Enemy** (Based on score increase) | `+ (Score / 500)` |
| **Wall Penalty** (Too close to edges) | `-0.2` |
| **Danger Penalty** (Too close to enemy) | `-0.3` |
| **Death** (Episode Ends) | `-50.0` |

> Total Return: The sum of rewards accumulated over `frame_skip` steps per action.

$$ 
R_t = \underbrace{0.1}_{\text{Survival}} + \underbrace{0.2 \cdot N_{\text{dodge}}+ \min\left(\frac{\Delta \text{Score}}{500}, 5\right)}_{\text{Performance}}-\underbrace{\left(0.2 \cdot I_{\text{wall}}+ 0.3 \cdot N_{\text{danger}}+ 50 \cdot I_{\text{death}}\right)}_{\text{Penalties}}
$$


## Quick Demo Time (Baby Model)

## Model Architecture and Implementation ([train_dqn.py](game/train_dqn.py))
**A. The Network Architecture**

```python
class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        # 1. Visual Cortex (Convolutional Layers)
        # Input: (Batch, 4, 84, 84) -> Captures motion from stacked frames
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # Detects edges/shapes
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Detects objects (bullets)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Detects trajectories
            nn.ReLU(),
        )
        
        # 2. Decision Making (Fully Connected Layers)
        # Flatten: 64 channels * 7 * 7 = 3136 features
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),       # Latent feature processing
            nn.ReLU(),
            nn.Linear(512, n_actions),  # Output: Q-value for each action
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```
**1. Feature Extraction (CNN)**
   - Since the game is a grid of 84x84 pixels, we use Convolutional layers to "see the game;
   - They function like the human vision, identifying edges, bullets, and enemy shapes from the raw image.
  
**2. Q-Value $$Q(s,a)$$**

$$
Q(s, a) = R + \gamma \max_{a'} Q(s', a')
$$

- the Q-value is thus the immediate reward (R) we get plus the estimated best possible future rewards from the next state;
- but discounted by a factor to prioritize immediate survival.
                                    

**B. Experience Replay Buffer**
- to stablize training, we store the agent's experiences in a replay buffer, which breaks the temporal correlation between consecutive frames
- and prevents the agent from overfitting to the current specific situation
```python
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # Randomly sample a batch to train the network
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
```
- capacity: 50,000 steps
- data sctore: (State, Action, Reward, Next_State, Done)
- Sampling: random batches of 64

**C. Hyperparameters**
| Parameter        | Value   | Purpose |
|------------------|---------|---------|
| GAMMA            | 0.99    | **Discount Factor:** Prioritizes long-term survival over immediate small rewards. |
| LR               | 5e-5    | **Learning Rate:** Kept small to ensure stable convergence of the loss function. |
| EPS_DECAY        | 150,000 | **Exploration Horizon:** Epsilon decays from `1.0` → `0.05` over 150k frames. |
| TARGET_UPDATE    | 5000    | **Sync Frequency:** Copies weights from Policy Net to Target Net every 5k steps. |
| Optimizer        | Adam    | Adaptive moment estimation for efficient gradient descent. |
| Loss Function    | MSE     | Mean Squared Error between predicted Q-values and target Q-values. |

**D. Optimization Strategy**
We use the Adam Optimizer with Mean Squared Error (MSE) loss.

1. Prediction: The Main Network predicts Q(s,a)

2. Target: The Target Network calculates Reward + Q 

3. Loss: MSE(Prediction, Target).

4. Gradient Clipping: We clip gradients to 10.0 to prevent exploding gradients when the reward signal is unstable.

## Human Play V.S. Final Model

## How to Run

```zsh
pip install pygame torch numpy opencv-python
py main.py
```

## Reference
This project implements a RL agent trained to play a bullet-hell shooter game.

The simulation environment is adapted from **[PlaneGame](game/PlaneGame)**, an open-source Python/Pygame shooter originally created by **Yongyu Yan** and **Yue Zhuo**. We modified the source code to support frame stacking, reward shaping, and a Gym-compatible interface for AI training.

