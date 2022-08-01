from pathlib import Path
from random import choice, seed, uniform

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# posは[y, x]の順番


class Agent:
    def __init__(self) -> None:
        self.log: list[tuple[int, int]] = []

    def init_pos(self, start: tuple[int, int]) -> None:
        self.pos = start


class TD:
    road = 0
    goal = 1
    start = 2
    wall = 3

    def __init__(self, field_path: Path, epsilon: float = 0.1, gamma: float = 0.95, alpha: float = 0.2) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.field_path = field_path
        self.read_field()
        h, w = self.field.shape
        self.values = np.zeros((h, w), dtype=np.float64)

        self.agent = Agent()
        self.initialize()

    def read_field(self) -> None:
        with open(self.field_path, "r") as f:
            lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        h = len(lines)
        w = len(lines[0])
        self.field = np.zeros((h, w), dtype=np.int8)
        for i in range(h):
            line = lines[i]
            self.field[i] = np.fromiter(map(int, list(self._convert_line(line))), dtype=np.int8)

    def _convert_line(self, line: str) -> str:
        return line.replace("S", str(self.start)).replace("G", str(self.goal)).replace(" ", str(self.road)).replace("#", str(self.wall))

    def next_positions(self) -> list[tuple[int, int]]:
        pos_now = self.agent.pos
        poss = []
        for direction in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            next_pos = (pos_now[0] + direction[0], pos_now[1] + direction[1])
            if not self.is_wall(next_pos):
                poss.append(next_pos)
        return poss

    @property
    def start_pos(self) -> tuple[int, int]:
        start_arr = np.where(self.field == TD.start)
        return (start_arr[0][0], start_arr[1][0])

    def is_wall(self, pos: tuple[int, int]) -> bool:
        return self.field[pos] == TD.wall

    def is_goal(self, pos: tuple[int, int]) -> bool:
        return self.field[pos] == TD.goal

    def initialize(self):
        start_pos = self.start_pos
        self.agent.init_pos(start_pos)
        self.agent.log = [start_pos]  # ゴールまで辿り着くごとにログを初期化

    def act(self) -> tuple[int, int]:
        poss = self.next_positions()
        if self.epsilon < uniform(0, 1):
            next_values: list[float] = [(1 if self.is_goal(pos) else 0) + self.gamma * self.values[pos] for pos in poss]
            # when several actions get the same value, select one from them randomly
            next_pos_idx = choice([i for i, next_val in enumerate(next_values) if next_val == max(next_values)])
        else:
            next_pos_idx = int(uniform(0, 1) * len(poss))

        next_pos = poss[next_pos_idx]
        self.agent.log.append(next_pos)
        return next_pos

    def update(self, next_pos: tuple[int, int]) -> None:
        pos = self.agent.pos
        r = 1 if self.is_goal(next_pos) else 0
        self.values[pos] += self.alpha * (r + self.gamma * self.values[next_pos] - self.values[pos])
        self.agent.pos = next_pos

    def episode(self) -> int:
        step = 0
        self.initialize()
        while not self.is_goal(self.agent.pos):
            next_pos = self.act()
            self.update(next_pos)
            step += 1
        return step

    def train(self) -> tuple[list[int], int]:
        least_repeat = 50
        max_repeat = 500
        th = 0.01
        episode_num = 0
        steps = []
        while episode_num < least_repeat:
            step = self.episode()
            steps.append(step)
            episode_num += 1
        prev_values = np.zeros_like(self.values)
        while episode_num < max_repeat and np.abs(self.values - prev_values).sum() > th:
            prev_values = self.values.copy()
            step = self.episode()
            steps.append(step)
            episode_num += 1
        return steps, episode_num

    def show_values(self) -> None:
        cmap = ListedColormap(["#00000000", "skyblue", "pink", "black"])
        plt.figure(figsize=(10, 10))
        plt.gca().set_aspect("equal")
        plt.xticks([])
        plt.yticks([])
        plt.gca().invert_yaxis()
        plt.pcolormesh(self.values, cmap="Reds", edgecolors="gray", linewidth=0.1)
        plt.pcolormesh(self.field, cmap=cmap, edgecolors="gray", linewidth=0.1)
        plt.show()
        plt.tight_layout()
        if str(self.field_path)[-5] == "2":
            plt.savefig(f"TD_values_0{str(self.epsilon)[2:]}_2.png", dpi=150)
        else:
            plt.savefig(f"TD_values_0{str(self.epsilon)[2:]}.png", dpi=150)

    def show_log(self) -> None:
        cmap = ListedColormap(["white", "skyblue", "pink", "black"])
        _, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.pcolormesh(self.field, cmap=cmap, edgecolors="gray", linewidth=0.1)
        prev_pos = self.agent.log[-1]
        for pos in reversed(self.agent.log[:-1]):
            ax.arrow(
                x=pos[1] + 0.5,
                y=pos[0] + 0.5,
                dx=prev_pos[1] - pos[1],
                dy=prev_pos[0] - pos[0],
                width=0.01,
                head_width=0.2,
                length_includes_head=True,
                color="orangered",
            )
            prev_pos = pos
        plt.gca().invert_yaxis()
        plt.show()
        plt.tight_layout()
        if str(self.field_path)[-5] == "2":
            plt.savefig(f"TD_log_0{str(self.epsilon)[2:]}_2.png", dpi=150)
        else:
            plt.savefig(f"TD_log_0{str(self.epsilon)[2:]}.png", dpi=150)

    def show_field(self) -> None:
        cmap = ListedColormap(["white", "skyblue", "pink", "black"])
        _, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.pcolormesh(self.field, cmap=cmap, edgecolors="gray", linewidth=0.1)
        plt.gca().invert_yaxis()
        plt.show()
        plt.tight_layout()
        if str(self.field_path)[-5] == "2":
            plt.savefig(f"field2.png", dpi=150)
        else:
            plt.savefig(f"field.png", dpi=150)


if __name__ == "__main__":
    import sys

    # python TD.py field.txt 0.2

    epsilon = float(sys.argv[2])

    seed(100)
    field_path = Path(sys.argv[1])
    td = TD(field_path, epsilon=epsilon)
    steps, episode_num = td.train()
    td.show_values()
    td.show_log()
    td.show_field()

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(episode_num), steps, "-")
    ax.set_title(rf"Learning curve of $\epsilon$-greedy TD(0) ($\epsilon={{{epsilon}}}$)", size=18)
    ax.set_xlabel("the number of episodes", size=15)
    ax.set_ylabel("the number of steps", size=15)
    ax.text(0.7, 0.2, f"final step: {steps[-1]}", transform=ax.transAxes, size=18)
    plt.show()
    plt.tight_layout()
    if str(field_path)[-5] == "2":
        plt.savefig(f"TD_learning_0{str(epsilon)[2:]}_2.png", dpi=150)
    else:
        plt.savefig(f"TD_learning_0{str(epsilon)[2:]}.png", dpi=150)
