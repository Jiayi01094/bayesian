# plane_env.py
import numpy as np
from PIL import Image
import pygame

from main import Game
import variables as var


class PlaneGameEnv:
    """
    A minimal Gym-like environment wrapper for your PlaneGame.

    Action space (B-mode):
        0: no-op
        1: move left
        2: move right
        3: move up
        4: move down

    Observation:
        (1, 84, 84) grayscale image, normalized to [0,1]

    Reward:
        Δscore / 1000  +  life_loss_penalty

    Episode ends when life_num == 0
    """

    def __init__(self, frame_skip=2, death_penalty=50.0):
        self.n_actions = 5
        self.frame_skip = frame_skip
        self.death_penalty = death_penalty

        self.frame_stack = 4  # number of frames
        self.frames = None  # buffer for stacked frames

        # Create game
        self.game = Game()
        self.clock = pygame.time.Clock()

        # First init
        var.init(self.game)
        self._force_rl_mode()

        self._last_score = self.game.score

    # ================================================================
    # Public API
    # ================================================================
    def reset(self):
        # 每次都重新创建全新的 Game()，确保所有速度、敌机数量、level全部归零
        from main import Game
        self.game = Game()

        # 跳过开始菜单
        self.game.start = False
        self.game._help = False
        self.game.paused = False
        self.game.transition = False

        # 获取初始 observation
        obs = self._get_obs()

        # 初始化 frame stack
        self.frames = [obs for _ in range(self.frame_stack)]
        return np.concatenate(self.frames, axis=0)

    def step(self, action):
        total_reward = 0.0
        done = False

        for _ in range(self.frame_skip):

            before_score = self.game.score
            before_life = self.game.life_num

            # Update game frame
            self.game.step_frame(
                action=action,
                use_keyboard=False,
                clock=self.clock
            )

            # ---------------------------
            # Reward Shaping
            # ---------------------------

            reward = 0.05  # 基础生存奖励

            # 1. 距离敌机的奖励
            mx, my = self.game.me.rect.centerx, self.game.me.rect.centery
            min_dist = 9999

            for e in self.game.enemies:
                ex, ey = e.rect.centerx, e.rect.centery
                d = np.sqrt((mx - ex) ** 2 + (my - ey) ** 2)
                if d < min_dist:
                    min_dist = d

            # 平滑的距离奖励（核心）
            reward += np.tanh((min_dist - 120) / 40) * 0.3

            # 强惩罚
            if min_dist < 60:
                reward -= 4.0

            # 稍微奖励安全距离
            elif min_dist > 130:
                reward += 0.1

            # 2. Left/right wall penalty
            if self.game.me.rect.left < 40 or self.game.me.rect.right > self.game.width - 40:
                reward -= 0.4

            # 3. Kill bonus
            delta_score = self.game.score - before_score
            if delta_score > 0:
                reward += min(delta_score / 500.0, 5.0)

            # 4. Life loss
            if self.game.life_num < before_life:
                reward -= self.death_penalty
                done = True

        # Generate observation
        obs = self._get_obs()

        self.frames.pop(0)
        self.frames.append(obs)
        stacked = np.concatenate(self.frames, axis=0)

        info = {
            "score": self.game.score,
            "life": self.game.life_num
        }

        return stacked, float(total_reward), done, info

    def render(self):
        """Optional: allow external rendering, do nothing here."""
        pass

    def close(self):
        pygame.quit()

    # ================================================================
    # Internal helpers
    # ================================================================
    def _get_obs(self):
        """
        Capture the current screen, convert to grayscale 84x84.
        Output shape: (1,84,84)
        """
        frame = pygame.surfarray.array3d(self.game.screen)   # (W,H,3)
        frame = np.transpose(frame, (1, 0, 2))               # (H,W,3)

        img = Image.fromarray(frame)
        img = img.convert("L")       # grayscale
        img = img.resize((84, 84))   # reduce resolution

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr[None, :, :]  # (1,84,84)
        return arr

    def _force_rl_mode(self):
        """
        Skip start screens, help screens, pause, transitions.
        Always enter direct gameplay.
        """
        self.game.start = False
        self.game._help = False
        self.game.paused = False
        self.game.transition = False
