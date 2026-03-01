"""
test.py  ──  Pygame 可视化测试已训练的 Q-learning Pac-Man 智能体
用法: python test.py
快捷键:
  SPACE     ── 暂停 / 继续
  ENTER/R   ── 下一局（游戏结束后）
  ↑ ↓       ── 调整 AI 步进速度
  Q / ESC   ── 退出
"""

import sys
import os
import time
import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from game_env import PacManEnv, GRID_SIZE, NUM_DOTS, ACTION_NAMES
from q_agent  import QLearningAgent

# ─── 路径 & 测试局数 ──────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model.pkl")
MAX_EPISODES = 10

# ─── 窗口 & 布局 ──────────────────────────────────────────────────────────────
CELL      = 80           # 每格像素
MARGIN    = 30           # 棋盘左/上边距
TOP_BAR   = 90           # 顶部标题栏高度
INFO_W    = 280          # 右侧信息面板宽度
GAP       = 4            # 格子间距

BOARD_PX  = GRID_SIZE * CELL + (GRID_SIZE - 1) * GAP
WIN_W     = MARGIN * 2 + BOARD_PX + INFO_W + 20
WIN_H     = TOP_BAR + BOARD_PX + MARGIN + 20

# ─── 颜色 ────────────────────────────────────────────────────────────────────
BG_COLOR    = (10,  10,  25)
BOARD_BG    = (18,  20,  45)
CELL_COLOR  = (25,  30,  60)
WALL_COLOR  = (40,  50,  90)
DOT_COLOR   = (255, 230, 100)
PAC_COLOR   = (255, 220,  30)
GHOST_COLORS= [(255, 80, 80), (255, 140, 50)]
HEADER_C    = (0,  210, 255)
TEXT_W      = (220, 220, 220)
WIN_COLOR   = (80,  255, 120)
LOSE_COLOR  = (255,  80,  80)
SPEED_LEVELS= [0.06, 0.2, 0.5, 1.0]


def board_xy(row, col):
    """格子(row,col)的像素左上角坐标。"""
    x = MARGIN + col * (CELL + GAP)
    y = TOP_BAR + row * (CELL + GAP)
    return x, y


def draw_board(screen, env, fonts):
    """绘制棋盘、豆子、Pac-Man、鬼。"""
    pac    = env.get_pac()
    ghosts = env.get_ghosts()
    dots   = env.get_dots()

    # 棋盘底板
    board_rect = pygame.Rect(MARGIN - GAP, TOP_BAR - GAP,
                             BOARD_PX + GAP * 2, BOARD_PX + GAP * 2)
    pygame.draw.rect(screen, BOARD_BG, board_rect, border_radius=10)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y = board_xy(r, c)
            rect = pygame.Rect(x, y, CELL, CELL)
            pygame.draw.rect(screen, CELL_COLOR, rect, border_radius=6)
            pygame.draw.rect(screen, WALL_COLOR, rect, width=1, border_radius=6)

            # 豆子
            if (r, c) in dots:
                pygame.draw.circle(screen, DOT_COLOR,
                                   (x + CELL // 2, y + CELL // 2), 6)

    # 鬼
    for i, ghost in enumerate(ghosts):
        gx, gy = board_xy(ghost[0], ghost[1])
        cx = gx + CELL // 2
        cy = gy + CELL // 2
        gc = GHOST_COLORS[i % len(GHOST_COLORS)]
        # 鬼身体（圆形 + 方块底部）
        pygame.draw.circle(screen, gc, (cx, cy - 5), CELL // 2 - 8)
        pygame.draw.rect(screen, gc,
                         (gx + 8, cy - 5, CELL - 16, CELL // 2 - 2))
        # 鬼眼睛
        pygame.draw.circle(screen, (255, 255, 255), (cx - 7, cy - 10), 5)
        pygame.draw.circle(screen, (255, 255, 255), (cx + 7, cy - 10), 5)
        pygame.draw.circle(screen, (0,  0, 180),    (cx - 6, cy - 10), 3)
        pygame.draw.circle(screen, (0,  0, 180),    (cx + 8, cy - 10), 3)
        # 鬼编号
        gi = fonts["tiny"].render(f"G{i+1}", True, (255, 255, 255))
        screen.blit(gi, (cx - gi.get_width() // 2, gy + CELL - 20))

    # Pac-Man
    px, py = board_xy(pac[0], pac[1])
    cx = px + CELL // 2
    cy = py + CELL // 2
    pygame.draw.circle(screen, PAC_COLOR, (cx, cy), CELL // 2 - 6)
    pygame.draw.circle(screen, (200, 160, 0), (cx, cy), CELL // 2 - 6, 2)
    # 眼睛
    pygame.draw.circle(screen, (0, 0, 0), (cx + 5, cy - 8), 4)
    lbl = fonts["tiny"].render("AI", True, (40, 30, 0))
    screen.blit(lbl, (cx - lbl.get_width() // 2, cy + 4))


def draw_info(screen, d, fonts):
    """绘制信息面板"""
    rx = MARGIN * 2 + BOARD_PX + 10
    ry = TOP_BAR - GAP
    rw = INFO_W - 10
    rh = BOARD_PX + GAP * 2

    pygame.draw.rect(screen, (15, 18, 42), (rx, ry, rw, rh), border_radius=10)
    pygame.draw.rect(screen, (45, 65, 130), (rx, ry, rw, rh), width=2, border_radius=10)

    y = ry + 24

    def row(label, val, vc=TEXT_W):
        nonlocal y
        ls = fonts["small"].render(label, True, (125, 145, 200))
        screen.blit(ls, (rx + 14, y))
        vs = fonts["info"].render(str(val), True, vc)
        screen.blit(vs, (rx + rw - vs.get_width() - 14, y))
        y += 38

    def sep():
        nonlocal y
        pygame.draw.line(screen, (35, 45, 85),
                         (rx + 14, y + 2), (rx + rw - 14, y + 2))
        y += 16

    row("Episode",    f"{d['episode']} / {MAX_EPISODES}")
    row("Step",       d["step"])
    sep()
    rc = (100, 230, 100) if d["reward"] >= 0 else (255, 100, 100)
    row("Reward",     f"{d['reward']:+.0f}", rc)
    sep()
    row("Score",      int(d["score"]))
    row("Best Score", int(d["best_score"]), (255, 215, 45))


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到模型: {MODEL_PATH}\n请先运行 train.py 训练模型。")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Pac-Man")
    clock  = pygame.time.Clock()

    def load_font(size):
        for name in ["microsoftyahei", "simhei", "notosanscjk"]:
            try:
                return pygame.font.SysFont(name, size)
            except Exception:
                pass
        return pygame.font.Font(None, size)

    fonts = {
        "title": load_font(28),
        "info":  load_font(19),
        "small": load_font(16),
        "tiny":  load_font(14),
    }

    env   = PacManEnv()
    agent = QLearningAgent(env.get_state_space(), env.get_action_size())
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0   # 纯利用

    state       = env.reset()
    episode     = 1
    step        = 0
    last_reward = 0
    last_action = "-"
    total_score = 0
    best_score  = 0
    end_reason  = ""     # "win" / "dead" / "timeout"
    paused      = False
    speed_idx   = 1
    last_step_t = time.time()
    running     = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_RETURN, pygame.K_r):
                    if env.is_done():
                        if episode >= MAX_EPISODES:
                            running = False
                        else:
                            state       = env.reset()
                            episode    += 1
                            step        = 0
                            last_reward = 0
                            last_action = "-"
                            total_score = 0
                            end_reason  = ""
                elif event.key == pygame.K_UP:
                    speed_idx = max(0, speed_idx - 1)
                elif event.key == pygame.K_DOWN:
                    speed_idx = min(len(SPEED_LEVELS) - 1, speed_idx + 1)

        # ── AI 步进 ────────────────────────────────────────────────────────────
        now = time.time()
        if not paused and not env.is_done() and \
                (now - last_step_t) >= SPEED_LEVELS[speed_idx]:
            last_step_t = now
            action      = agent.select_action(state)
            last_action = ACTION_NAMES[action]
            next_state, reward, done, info = env.step(action)
            last_reward  = reward
            total_score += reward
            state        = next_state
            step        += 1
            if info.get("timeout"):
                end_reason = "timeout"
            elif info.get("dead"):
                end_reason = "dead"
            elif info.get("win"):
                end_reason = "win"
            if total_score > best_score:
                best_score = total_score

        # ── 绘制 ──────────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Title
        title = fonts["title"].render("Pac-Man", True, HEADER_C)
        screen.blit(title, (WIN_W // 2 - title.get_width() // 2, 14))
        hint  = fonts["small"].render(
            "SPACE=Pause  ENTER=Next Game  ↑↓=Speed  Q=Quit", True, (90, 110, 160))
        screen.blit(hint, (WIN_W // 2 - hint.get_width() // 2, 56))

        # board
        draw_board(screen, env, fonts)

        draw_info(screen, {
            "episode":    episode,
            "step":       step,
            "reward":     last_reward,
            "score":      total_score,
            "best_score": best_score,
        }, fonts)

        # 游戏结束遮罩
        if env.is_done():
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))

            if end_reason == "win":
                color = WIN_COLOR
                msg   = "YOU WIN!"
            elif end_reason == "dead":
                color = LOSE_COLOR
                msg   = "被鬼抓住了！"
            else:
                color = (255, 180, 50)
                msg   = "超时结束！"

            ms = fonts["title"].render(msg, True, color)
            screen.blit(ms, (WIN_W // 2 - ms.get_width() // 2, WIN_H // 2 - 50))

            sc_s = fonts["info"].render(f"Score: {int(total_score)}", True, TEXT_W)
            screen.blit(sc_s, (WIN_W // 2 - sc_s.get_width() // 2, WIN_H // 2 + 2))

            if episode < MAX_EPISODES:
                ht = f"ENTER = Next new game ({episode}/{MAX_EPISODES})    Q = Quit"
            else:
                ht = f"已完成全部 {MAX_EPISODES} 局    Q = 退出"
            hs = fonts["small"].render(ht, True, (195, 195, 195))
            screen.blit(hs, (WIN_W // 2 - hs.get_width() // 2, WIN_H // 2 + 40))

        pygame.display.flip()

    pygame.quit()
    print("已退出。")


if __name__ == "__main__":
    main()
