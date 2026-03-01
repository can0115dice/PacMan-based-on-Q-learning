"""
src/q_agent.py  ──  标准 Q-learning 智能体
Q-table 索引: (pac位置, 鬼1象限, 鬼2象限, 最近豆方向) → 4个动作

核心公式:
  Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

超参数说明:
  α (alpha)         学习率，控制每次更新的步幅
  γ (gamma)         折扣因子，控制对未来奖励的重视程度
  ε (epsilon)       探索率，epsilon-greedy 中随机探索的概率
  epsilon_decay     每局结束后 ε 的衰减系数
  epsilon_min       ε 的下限，始终保留少量探索
"""

import numpy as np
import random
import pickle


class QLearningAgent:

    def __init__(
        self,
        state_space:    tuple,
        action_size:    int,
        alpha:          float = 0.15,
        gamma:          float = 0.95,
        epsilon:        float = 1.0,
        epsilon_decay:  float = 0.995,
        epsilon_min:    float = 0.05,
    ):
        self.state_space   = state_space
        self.action_size   = action_size
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Q-table: shape = (*state_space, action_size)
        # 例: (49, 5, 5, 4, 4) — 最后一维是4个动作
        self.q_table = np.zeros((*state_space, action_size), dtype=np.float64)

    # ── Epsilon-greedy 动作选择 ────────────────────────────────────────────────
    def select_action(self, state: tuple) -> int:
        """
        以概率 ε 随机探索，以概率 1-ε 选择 Q 值最大的动作。
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state]))

    # ── Q-learning 更新 ───────────────────────────────────────────────────────
    def update(
        self,
        state:      tuple,
        action:     int,
        reward:     float,
        next_state: tuple,
        done:       bool,
    ) -> float:
        """
        标准 Q-learning 更新（off-policy TD）。
        返回 |TD error|，用于监控收敛情况。

        公式: Q(s,a) ← Q(s,a) + α * [r + γ * max_a'Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target_q - current_q
        self.q_table[state][action] += self.alpha * td_error
        return abs(td_error)

    # ── Epsilon 衰减 ──────────────────────────────────────────────────────────
    def decay_epsilon(self):
        """每局结束后调用，让探索率从高到低衰减。"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── 辅助 ──────────────────────────────────────────────────────────────────
    def get_best_action(self, state: tuple) -> int:
        return int(np.argmax(self.q_table[state]))

    def get_q_values(self, state: tuple) -> np.ndarray:
        return self.q_table[state].copy()

    def get_policy_summary(self) -> dict:
        visited = int(np.sum(np.any(self.q_table != 0, axis=-1)))
        total   = int(np.prod(self.state_space))
        return {
            "visited_states": visited,
            "total_states":   total,
            "coverage":       visited / max(total, 1),
            "mean_q":         float(np.mean(self.q_table)),
            "max_q":          float(np.max(self.q_table)),
        }

    # ── 保存 / 加载 ───────────────────────────────────────────────────────────
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "q_table":       self.q_table,
                "epsilon":       self.epsilon,
                "alpha":         self.alpha,
                "gamma":         self.gamma,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min":   self.epsilon_min,
            }, f)
        print(f"[Agent] Q-table 已保存 → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table       = data["q_table"]
        self.epsilon       = data["epsilon"]
        self.alpha         = data.get("alpha",         self.alpha)
        self.gamma         = data.get("gamma",         self.gamma)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.epsilon_min   = data.get("epsilon_min",   self.epsilon_min)
        print(f"[Agent] Q-table 已加载 ← {path}")
