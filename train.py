"""
train.py  ──  训练 Q-learning 智能体玩小型 Pac-Man（7×7，1只鬼）
用法: python train.py
训练结束后保存 model.pkl，并生成 training_result.png
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from game_env import PacManEnv, GRID_SIZE, NUM_DOTS, ACTION_NAMES
from q_agent  import QLearningAgent

# ─── 超参数 ───────────────────────────────────────────────────────────────────
NUM_EPISODES   = 10000
ALPHA          = 0.15    # 学习率 α
GAMMA          = 0.95    # 折扣因子 γ
EPSILON_START  = 1.0     # 初始探索率 ε
EPSILON_DECAY  = 0.995   # ε 每局衰减系数
EPSILON_MIN    = 0.05    # ε 最小值
SAVE_PATH      = os.path.join(os.path.dirname(__file__), "model.pkl")
PLOT_PATH      = os.path.join(os.path.dirname(__file__), "training_result.png")
PRINT_EVERY    = 1000
# ─────────────────────────────────────────────────────────────────────────────


def moving_average(data, window=200):
    if len(data) < window:
        return np.array(data, dtype=float)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def train():
    env   = PacManEnv()
    agent = QLearningAgent(
        state_space   = env.get_state_space(),
        action_size   = env.get_action_size(),
        alpha         = ALPHA,
        gamma         = GAMMA,
        epsilon       = EPSILON_START,
        epsilon_decay = EPSILON_DECAY,
        epsilon_min   = EPSILON_MIN,
    )

    print(f"地图: {GRID_SIZE}×{GRID_SIZE}  |  豆子: {NUM_DOTS}  |  "
          f"状态空间: {env.get_state_space()}  |  "
          f"Q-table大小: {env.get_state_size() * env.get_action_size()} 个值")
    print(f"超参数: α={ALPHA}, γ={GAMMA}, ε={EPSILON_START}→{EPSILON_MIN} "
          f"(decay={EPSILON_DECAY})")
    print(f"开始训练，共 {NUM_EPISODES} 局...\n")

    # 记录指标
    rewards_hist   = []
    steps_hist     = []
    dots_eaten_hist= []
    win_hist       = []
    epsilon_hist   = []
    td_hist        = []

    best_reward = -np.inf

    for ep in range(1, NUM_EPISODES + 1):
        state     = env.reset()
        ep_reward = 0
        ep_tds    = []

        while True:
            action                         = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            td                             = agent.update(state, action, reward,
                                                          next_state, done)
            ep_reward += reward
            ep_tds.append(td)
            state = next_state
            if done:
                break

        agent.decay_epsilon()

        dots_eaten = NUM_DOTS - env.dots_remaining()
        won        = 1 if env.is_won() else 0

        rewards_hist.append(ep_reward)
        steps_hist.append(env.get_step_count())
        dots_eaten_hist.append(dots_eaten)
        win_hist.append(won)
        epsilon_hist.append(agent.epsilon)
        td_hist.append(np.mean(ep_tds))

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(SAVE_PATH)

        if ep % PRINT_EVERY == 0:
            r_win  = np.mean(win_hist[-PRINT_EVERY:]) * 100
            r_avg  = np.mean(rewards_hist[-PRINT_EVERY:])
            d_avg  = np.mean(dots_eaten_hist[-PRINT_EVERY:])
            summ   = agent.get_policy_summary()
            print(
                f"Ep {ep:>6}/{NUM_EPISODES}  "
                f"avg_reward={r_avg:>8.1f}  "
                f"win_rate={r_win:>5.1f}%  "
                f"avg_dots={d_avg:>4.1f}/{NUM_DOTS}  "
                f"ε={agent.epsilon:.3f}  "
                f"Q覆盖={summ['coverage']*100:.1f}%"
            )

    print(f"\n训练完成！最高单局奖励: {best_reward:.1f}")
    print(f"最终 Q-table: {agent.get_policy_summary()}")

    # ─── 绘图 ─────────────────────────────────────────────────────────────────
    BG     = "#0f0f1a"
    AX_BG  = "#1a1a2e"
    COLORS = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff", "#ff9f43"]
    TEXTC  = "#e0e0e0"
    GRIDC  = "#2a2a4a"
    MA_W   = 200

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle(f"Q-Learning Pac-Man {GRID_SIZE}×{GRID_SIZE}  ·  Training Results",
                 fontsize=17, color=TEXTC, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.32,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    def style(ax, title, xlabel="Episode", ylabel=""):
        ax.set_facecolor(AX_BG)
        ax.set_title(title, color=TEXTC, fontsize=11, pad=7)
        ax.set_xlabel(xlabel, color=TEXTC, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXTC, fontsize=9)
        ax.tick_params(colors=TEXTC, labelsize=8)
        ax.grid(color=GRIDC, linewidth=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    ep_x = np.arange(1, NUM_EPISODES + 1)

    # 1. 总奖励
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep_x, rewards_hist, alpha=0.12, color=COLORS[0], lw=0.5)
    ma = moving_average(rewards_hist, MA_W)
    ax1.plot(np.arange(MA_W, NUM_EPISODES + 1), ma,
             color=COLORS[0], lw=1.8, label=f"MA-{MA_W}")
    ax1.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXTC)
    style(ax1, "Total Reward per Episode", ylabel="Reward")

    # 2. 胜率（吃完所有豆子）
    ax2 = fig.add_subplot(gs[0, 1])
    win_ma = moving_average(win_hist, MA_W) * 100
    ax2.plot(np.arange(MA_W, NUM_EPISODES + 1), win_ma,
             color=COLORS[3], lw=1.8)
    ax2.set_ylim(0, 105)
    ax2.axhline(50, color=COLORS[3], lw=0.8, ls="--", alpha=0.4)
    style(ax2, f"Win Rate — All Dots Eaten (MA-{MA_W}, %)", ylabel="%")

    # 3. 平均吃豆数
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ep_x, dots_eaten_hist, alpha=0.12, color=COLORS[5], lw=0.5)
    ma3 = moving_average(dots_eaten_hist, MA_W)
    ax3.plot(np.arange(MA_W, NUM_EPISODES + 1), ma3,
             color=COLORS[5], lw=1.8, label=f"MA-{MA_W}")
    ax3.axhline(NUM_DOTS, color="white", lw=0.8, ls="--", alpha=0.4,
                label=f"Max={NUM_DOTS}")
    ax3.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXTC)
    style(ax3, "Dots Eaten per Episode", ylabel="Dots")

    # 4. Epsilon 衰减
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(ep_x, epsilon_hist, color=COLORS[2], lw=1.5)
    ax4.set_ylim(0, 1.05)
    style(ax4, "Epsilon Decay (Exploration → Exploitation)", ylabel="ε")

    # 5. TD 误差收敛
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(ep_x, td_hist, alpha=0.12, color=COLORS[4], lw=0.5)
    ma5 = moving_average(td_hist, MA_W)
    ax5.plot(np.arange(MA_W, NUM_EPISODES + 1), ma5,
             color=COLORS[4], lw=1.8, label=f"MA-{MA_W}")
    ax5.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXTC)
    style(ax5, "Mean |TD Error| (↓ = converging)", ylabel="|TD Error|")

    # 6. 步数分布
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(steps_hist, bins=40, color=COLORS[1], alpha=0.8, edgecolor="#992222")
    ax6.axvline(np.mean(steps_hist), color="white", lw=1.2, ls="--",
                label=f"mean={np.mean(steps_hist):.0f}")
    ax6.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXTC)
    style(ax6, "Steps per Episode Distribution", xlabel="Steps", ylabel="Frequency")

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"训练图表已保存 → {PLOT_PATH}")


if __name__ == "__main__":
    train()
