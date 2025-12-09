# Plane War AI: Autonomous Game Control under Uncertainty (working readme file)

### Group Members
- Eddie Tian, Jiayi Wang, Sebastian De La Cruz

> **Project Goal:** To train a RL agent to master a chaotic Atari Asteroids like shooter game from scratch using only raw pixel input and learning to estimate value and risk from raw pixel input."

## Intro & Background
Video games like Plane War act as stochastic environments requiring quick reflexes, visual recognition, and implicit probability estimation. Our AI starts with zero knowledge:
- It doesn't know what a "plane" is;
- It doesn't know "bullets" represent a probability of death;
- It receives only a stream of noisy visual data and a sparse reward signal;
The Mission: Build an end-to-end Reinforcement Learning pipeline that evolves from "random mvoe" to "predictive dodging."

## The Environment ([plane_env.py](game/plane_env.py))
We utilized a custom Python-based game engine wrapped in a Gym-compatible interface for RL training.

**Input (The Eyes)** 
- Raw screen pixels converted to 84x84 Grayscale; 

**State Space** 
- A stack of 4 consecutive frames (Shape: 4x84x84);

**Action Space (The Controls)**
- 5 discrete actions: UP, Down, Left, Right, Do Nothing

## Quick Demo Time (Baby Model)


## Model Architecture ([train_dqn.py](game/train_dqn.py))


## Training Pipeline


## Human Play V.S. Final Model


## How to Run

```zsh
pip install pygame torch numpy opencv-python
py main.py
```

## Reference

