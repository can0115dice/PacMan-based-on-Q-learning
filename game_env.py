"""
src/game_env.py  ──  小型 Pac-Man 核心逻辑
地图: 7×7 格子，无墙壁（开放地图）
角色: Pac-Man（玩家）+ 1只鬼（纯随机游走）
豆子: 随机分布在地图上，Pac-Man 吃完所有豆子即胜利
鬼: 完全随机游走，碰到 Pac-Man 则游戏结束

状态空间设计（对 Q-learning 友好）:
  - 鬼的方向:           8 （上/下/左/右/左上/右上/左下/右下 + 同格=8）
  - 鬼的距离等级:       3 （近=1格, 中=2-3格, 远=4+格）
  - 最近豆子的方向:     4 （正上/正下/正左/正右，优先同行/列）
  - 最近豆子距离等级:   3 （紧邻=1格, 近=2-3格, 远=4+格）
  总计: 8 × 3 × 4 × 3 = 288 状态  ← 极小，10000局完全收敛
"""

import numpy as np
import random

# ─── 地图参数 ─────────────────────────────────────────────────────────────────
GRID_SIZE   = 7
NUM_GHOSTS  = 1
NUM_DOTS    = 15
MAX_STEPS   = 300

# ─── 动作 ─────────────────────────────────────────────────────────────────────
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTIONS      = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
ACTION_DELTA = {
    ACTION_UP:    (-1, 0),
    ACTION_DOWN:  ( 1, 0),
    ACTION_LEFT:  ( 0,-1),
    ACTION_RIGHT: ( 0, 1),
}
ACTION_NAMES = ["↑ UP", "↓ DOWN", "← LEFT", "→ RIGHT"]

# ─── 奖励值 ───────────────────────────────────────────────────────────────────
REWARD_DOT        =  20
REWARD_WIN        = 300
REWARD_DEAD       = -100
REWARD_STEP       =   0
REWARD_NEAR_GHOST =  -5   # 距离1格
REWARD_CLOSE_GHOST = -2   # 距离2-3格


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _ghost_state(pac, ghost):
    """
    鬼的方向（8方向+同格）× 距离等级（3级）。
    方向: 0=上 1=下 2=左 3=右 4=左上 5=右上 6=左下 7=右下 (8=同格归入最近距离)
    距离: 0=1格(危险) 1=2-3格(警戒) 2=4+格(安全)
    返回 (direction_idx, dist_level)
    """
    dr = ghost[0] - pac[0]
    dc = ghost[1] - pac[1]
    dist = abs(dr) + abs(dc)

    if dist == 0:
        return (0, 0)   # 同格，最危险

    if dist <= 1:
        dist_level = 0
    elif dist <= 3:
        dist_level = 1
    else:
        dist_level = 2

    if dr < 0 and dc == 0:   direction = 0  # 正上
    elif dr > 0 and dc == 0: direction = 1  # 正下
    elif dr == 0 and dc < 0: direction = 2  # 正左
    elif dr == 0 and dc > 0: direction = 3  # 正右
    elif dr < 0 and dc < 0:  direction = 4  # 左上
    elif dr < 0 and dc > 0:  direction = 5  # 右上
    elif dr > 0 and dc < 0:  direction = 6  # 左下
    else:                     direction = 7  # 右下

    return (direction, dist_level)


def _dot_state(pac, dots):
    """
    最近豆子的方向（4个正方向）× 距离等级（3级）。
    优先选同行/同列的豆子（AI 可以直线走到），否则选曼哈顿最近。
    方向: 0=上 1=下 2=左 3=右
    距离: 0=1格 1=2-3格 2=4+格
    """
    if not dots:
        return (0, 2)   # 无豆，默认远处

    pr, pc = pac

    # 优先找同行或同列的豆子（AI能直线到达）
    same_line = [d for d in dots if d[0] == pr or d[1] == pc]
    candidates = same_line if same_line else list(dots)

    nearest = min(candidates, key=lambda d: abs(d[0]-pr) + abs(d[1]-pc))
    dr = nearest[0] - pr
    dc = nearest[1] - pc
    dist = abs(dr) + abs(dc)

    if dist <= 1:   dist_level = 0
    elif dist <= 3: dist_level = 1
    else:           dist_level = 2

    # 方向：以绝对偏移大的轴为主
    if abs(dr) >= abs(dc):
        direction = 0 if dr < 0 else 1   # 上/下
    else:
        direction = 2 if dc < 0 else 3   # 左/右

    return (direction, dist_level)


class PacManEnv:

    @staticmethod
    def get_state_size() -> int:
        """状态总数量: 8 × 3 × 4 × 3 = 288"""
        return 8 * 3 * 4 * 3

    @staticmethod
    def get_state_space() -> tuple:
        """Q-table shape: (鬼方向, 鬼距离, 豆方向, 豆距离)"""
        return (8, 3, 4, 3)

    @staticmethod
    def get_action_size() -> int:
        return len(ACTIONS)

    def reset(self):
        self.step_count = 0
        self.done       = False
        self.won        = False

        all_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        random.shuffle(all_cells)

        self.pac = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))

        ghost_candidates = [
            (r, c) for r, c in all_cells
            if abs(r - self.pac[0]) + abs(c - self.pac[1]) >= 4
        ]
        random.shuffle(ghost_candidates)
        self.ghost = ghost_candidates[0]

        occupied = {self.pac, self.ghost}
        dot_candidates = [cell for cell in all_cells if cell not in occupied]
        self.dots = set(random.sample(dot_candidates,
                                      min(NUM_DOTS, len(dot_candidates))))
        return self._get_state()

    def step(self, action: int):
        if self.done:
            return self._get_state(), 0, True, {}

        dr, dc   = ACTION_DELTA[action]
        nr       = _clamp(self.pac[0] + dr, 0, GRID_SIZE - 1)
        nc       = _clamp(self.pac[1] + dc, 0, GRID_SIZE - 1)
        self.pac = (nr, nc)

        reward = REWARD_STEP
        info   = {}

        if self.pac in self.dots:
            self.dots.discard(self.pac)
            reward += REWARD_DOT
            info["ate_dot"] = True

        if not self.dots:
            reward    += REWARD_WIN
            self.done  = True
            self.won   = True
            info["win"] = True
            return self._get_state(), reward, self.done, info

        self.ghost = self._move_ghost(self.ghost)

        if self.pac == self.ghost:
            reward      += REWARD_DEAD
            self.done    = True
            info["dead"] = True
            return self._get_state(), reward, self.done, info

        # 分级鬼惩罚
        ghost_dist = abs(self.ghost[0]-self.pac[0]) + abs(self.ghost[1]-self.pac[1])
        if ghost_dist == 1:
            reward += REWARD_NEAR_GHOST
        elif ghost_dist <= 3:
            reward += REWARD_CLOSE_GHOST

        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            reward         -= 50
            self.done       = True
            info["timeout"] = True

        return self._get_state(), reward, self.done, info

    def _move_ghost(self, ghost):
        move = random.choice(list(ACTION_DELTA.values()))
        nr   = _clamp(ghost[0] + move[0], 0, GRID_SIZE - 1)
        nc   = _clamp(ghost[1] + move[1], 0, GRID_SIZE - 1)
        return (nr, nc)

    def _get_state(self) -> tuple:
        ghost_dir, ghost_dist = _ghost_state(self.pac, self.ghost)
        dot_dir,   dot_dist   = _dot_state(self.pac, self.dots)
        return (ghost_dir, ghost_dist, dot_dir, dot_dist)

    def get_pac(self)        -> tuple: return self.pac
    def get_ghosts(self)     -> list:  return [self.ghost]
    def get_dots(self)       -> set:   return set(self.dots)
    def get_step_count(self) -> int:   return self.step_count
    def is_done(self)        -> bool:  return self.done
    def is_won(self)         -> bool:  return self.won
    def dots_remaining(self) -> int:   return len(self.dots)

