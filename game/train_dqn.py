import random
import collections
from typing import Deque, Tuple
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from plane_env import PlaneGameEnv


# ====================== DQN 网络 ======================
class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        # 输入是 4 帧堆叠：(4,84,84)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # -> (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64,7,7)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        # x: (B,4,84,84)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ====================== Replay Buffer ======================
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state / next_state: (4,84,84)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),          # (B,4,84,84)
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ====================== 超参数（30 万步版本） ======================
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_CAPACITY = 50000
LR = 5e-5

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRAMES = 150000   # 3 万步内从 1 -> 0.1

TARGET_UPDATE_INTERVAL = 5000  # 每 1000 步拷贝一次 target
MAX_FRAMES = 300000             # 这就是你选的 A：5 万步小训练
MAX_EPISODE_STEPS = 2000       # 每局最多步数，防止死循环


def select_action(state, policy_net, steps_done, n_actions, device):
    """
    epsilon-greedy 动作选择
    state: (1,4,84,84) numpy
    """
    eps = EPS_END + (EPS_START - EPS_END) * max(
        0.0, (EPS_DECAY_FRAMES - steps_done) / EPS_DECAY_FRAMES
    )

    if random.random() < eps:
        return random.randrange(n_actions), eps
    else:
        with torch.no_grad():
            state_v = torch.from_numpy(state).float().to(device)  # (1,4,84,84)
            q_values = policy_net(state_v)                        # (1,n_actions)
            action = int(torch.argmax(q_values, dim=1).item())
        return action, eps


def optimize_model(
    policy_net, target_net, optimizer, replay_buffer, device
):
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0

    states, actions, rewards, next_states, dones = replay_buffer.sample(
        BATCH_SIZE
    )

    # 转 tensor
    states_v = torch.from_numpy(states).float().to(device)         # (B,4,84,84)
    next_states_v = torch.from_numpy(next_states).float().to(device)
    actions_v = torch.from_numpy(actions).long().to(device)        # (B,)
    rewards_v = torch.from_numpy(rewards).to(device)               # (B,)
    dones_v = torch.from_numpy(dones).to(device)                   # (B,)

    # 当前 Q(s,a)
    q_values = policy_net(states_v)                                # (B,n_actions)
    state_action_values = q_values.gather(1, actions_v.unsqueeze(1)).squeeze(1)

    # 目标 Q
    with torch.no_grad():
        next_q_values = target_net(next_states_v).max(1)[0]        # (B,)
        expected_q_values = rewards_v + GAMMA * next_q_values * (1 - dones_v)

    loss = nn.MSELoss()(state_action_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return float(loss.item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    writer = SummaryWriter(log_dir="runs/PlaneDQN")

    # 创建环境（用你写好的 PlaneGameEnv）
    env = PlaneGameEnv(frame_skip=2, death_penalty=-50.0)
    n_actions = env.n_actions

    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    steps_done = 0
    episode_idx = 0

    while steps_done < MAX_FRAMES:
        obs = env.reset()          # (4,84,84)
        state = obs[np.newaxis, ...]  # (1,4,84,84)

        episode_reward = 0.0

        for t in range(MAX_EPISODE_STEPS):
            action, eps = select_action(
                state, policy_net, steps_done, n_actions, device
            )
            with torch.no_grad():
                state_v = torch.from_numpy(state).float().to(device)
                q_values = policy_net(state_v)
                writer.add_scalar("Q/max", q_values.max().item(), steps_done)
                writer.add_scalar("Q/mean", q_values.mean().item(), steps_done)


            next_obs, reward, done, info = env.step(action)
            next_state = next_obs[np.newaxis, ...]  # (1,4,84,84)

            replay_buffer.push(
                state[0], action, reward, next_state[0], done
            )

            loss = optimize_model(
                policy_net, target_net, optimizer, replay_buffer, device
            )
            writer.add_scalar("Loss/loss", loss, steps_done)


            state = next_state
            episode_reward += reward
            steps_done += 1

            # 更新 target 网络
            if steps_done % TARGET_UPDATE_INTERVAL == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # 每 1000 步打印一次训练信息
            if steps_done % 1000 == 0:
                print(
                    f"[global_step={steps_done}] "
                    f"eps={eps:.3f}, "
                    f"last_loss={loss:.4f}, "
                    f"episode_reward={episode_reward:.2f}, "
                    f"score={info['score']}, life={info['life']}"
                )

            if done or steps_done >= MAX_FRAMES:
                break

        episode_idx += 1
        # ---------- TensorBoard: 每个 episode ----------
        writer.add_scalar("Episode/reward", episode_reward, episode_idx)
        writer.add_scalar("Episode/score", info["score"], episode_idx)
        writer.add_scalar("Episode/epsilon", eps, episode_idx)

        print(
            f"Episode {episode_idx} finished: "
            f"reward={episode_reward:.2f}, "
            f"steps_done={steps_done}"
        )

    env.close()
    torch.save(policy_net.state_dict(), "dqn_plane_50k_3.pth")
    print("Training done, model saved to dqn_plane_300k.pth")
    writer.close()

if __name__ == "__main__":
    main()
