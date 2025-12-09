from plane_env import PlaneGameEnv
import random

env = PlaneGameEnv(frame_skip=2)
obs = env.reset()

done = False
ep_reward = 0

while not done:
    action = random.randint(0, 4)  # random policy
    obs, reward, done, info = env.step(action)
    ep_reward += reward
    print(f"reward={reward:.3f}, score={info['score']}, life={info['life']}")

print("Episode finished:", ep_reward)
env.close()
