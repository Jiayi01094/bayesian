import torch
import numpy as np
from plane_env import PlaneGameEnv
from train_dqn import DQN   # 复用同样的网络结构


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PlaneGameEnv(frame_skip=2)
    n_actions = env.n_actions

    # 创建模型
    policy_net = DQN(n_actions).to(device)
    policy_net.load_state_dict(torch.load("dqn_plane_300k.pth", map_location=device))
    policy_net.eval()

    obs = env.reset()
    state = obs[np.newaxis, ...]

    episode_reward = 0

    while True:
        with torch.no_grad():
            state_v = torch.from_numpy(state).float().to(device)
            q = policy_net(state_v)
            action = int(torch.argmax(q, dim=1).item())

        next_obs, reward, done, info = env.step(action)
        next_state = next_obs[np.newaxis, ...]

        episode_reward += reward
        state = next_state

        if done:
            print("Episode finished.")
            print(f"Total reward: {episode_reward}")
            print(f"Final score: {info['score']}")
            break

    env.close()


if __name__ == "__main__":
    test()
