import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # 🟢 关闭画面渲染，大幅加速

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from plane_env import PlaneGameEnv


# ====================== DQN 网络 ======================
class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ====================== Replay Buffer ======================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ====================== 超参数 ======================
GAMMA = 0.99
BATCH_SIZE = 256                  # 🟢 更大 batch
BUFFER_CAPACITY = 50000
LR = 1e-4

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100000               # 🟢 epsilon 降得更慢，更稳定

TARGET_UPDATE = 2000
MAX_FRAMES = 500000              # 🟢 训练 50 万步
MAX_EPISODE_STEPS = 4000


# ====================== epsilon 动作选择 ======================
def select_action(state, policy_net, steps, n_actions, device):
    eps = EPS_END + (EPS_START - EPS_END) * max(
        0.0, (EPS_DECAY - steps) / EPS_DECAY
    )

    if random.random() < eps:
        return random.randrange(n_actions), eps

    with torch.no_grad():
        state_v = torch.from_numpy(state).float().to(device)
        q_values = policy_net(state_v)
        action = int(torch.argmax(q_values).item())
    return action, eps


# ====================== 优化器更新 ======================
def optimize(policy, target, optimizer, buffer, device):
    if len(buffer) < BATCH_SIZE:
        return 0.0

    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    states_v = torch.from_numpy(states).float().to(device)
    next_states_v = torch.from_numpy(next_states).float().to(device)
    actions_v = torch.from_numpy(actions).long().to(device)
    rewards_v = torch.from_numpy(rewards).to(device)
    dones_v = torch.from_numpy(dones).to(device)

    q = policy(states_v)
    state_action_values = q.gather(1, actions_v.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target(next_states_v).max(1)[0]
        target_values = rewards_v + GAMMA * next_q * (1 - dones_v)

    loss = nn.MSELoss()(state_action_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()

    return float(loss.item())


# ====================== 主训练循环 ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    # 🟢 frame_skip=6 大幅加速训练，减少渲染开销
    env = PlaneGameEnv(frame_skip=6)

    n_actions = env.n_actions
    policy = DQN(n_actions).to(device)
    target = DQN(n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    steps = 0
    episode = 0

    while steps < MAX_FRAMES:
        obs = env.reset()
        state = obs[np.newaxis, ...]

        episode_reward = 0

        for t in range(MAX_EPISODE_STEPS):
            action, eps = select_action(state, policy, steps, n_actions, device)

            next_obs, reward, done, info = env.step(action)
            next_state = next_obs[np.newaxis, ...]

            buffer.push(state[0], action, reward, next_state[0], done)

            loss = optimize(policy, target, optimizer, buffer, device)

            state = next_state
            episode_reward += reward
            steps += 1

            if steps % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())

            if steps % 5000 == 0:
                print(f"[{steps}] eps={eps:.3f}, loss={loss:.4f}, reward={episode_reward}, score={info['score']}")

            if done or steps >= MAX_FRAMES:
                break

        episode += 1
        print(f"Episode {episode} ended reward={episode_reward}, steps={steps}")

    torch.save(policy.state_dict(), "dqn_500k.pth")
    print("Training finished. Model saved: dqn_500k.pth")


if __name__ == "__main__":
    main()
