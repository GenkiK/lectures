from pathlib import Path
from random import choice, seed, uniform

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# posは[y, x]の順番


class Field:
    road = 0
    goal = 1
    start = 2
    wall = 3

    def read_field(self, field_path: Path) -> None:
        with open(field_path, "r") as f:
            lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        h = len(lines)
        w = len(lines[0])
        self.field = np.zeros((h, w), dtype=np.int8)

        for i in range(h):
            line = lines[i]
            self.field[i] = np.fromiter(map(int, list(self._convert_line(line))), dtype=np.int8)

    @property
    def start_pos(self) -> tuple[int, int]:
        start_arr = np.where(self.field == Field.start)
        return (start_arr[0][0], start_arr[1][0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.field.shape

    def _convert_line(self, line: str) -> str:
        return line.replace("S", str(Field.start)).replace("G", str(Field.goal)).replace(" ", str(Field.road)).replace("#", str(Field.wall))

    def is_goal(self, pos: tuple[int, int]) -> bool:
        return self.field[pos] == Field.goal

    def is_wall(self, pos: tuple[int, int]) -> bool:
        return self.field[pos] == Field.wall

    def show_values(self, values: np.ndarray) -> None:
        cmap = ListedColormap(["#00000000", "skyblue", "pink", "black"])
        plt.gca().set_aspect("equal")
        plt.xticks([])
        plt.yticks([])
        plt.gca().invert_yaxis()
        plt.pcolormesh(values, cmap="Greens", edgecolors="gray", linewidth=0.1)
        plt.pcolormesh(self.field, cmap=cmap, edgecolors="gray", linewidth=0.1)
        plt.show()

    def show_log(self, log: list[tuple[int, int]]) -> None:
        cmap = ListedColormap(["white", "skyblue", "pink", "black"])
        _, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.pcolormesh(self.field, cmap=cmap, edgecolors="gray", linewidth=0.1)
        prev_pos = log[-1]
        for pos in reversed(log[:-1]):
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
        plt.savefig("TD.png", dpi=150)


class Agent:
    def __init__(self) -> None:
        self.log: list[tuple[int, int]] = []

    def init_pos(self, start: tuple[int, int]) -> None:
        self.pos = start

    def next_positions(self, field: Field) -> list[tuple[int, int]]:
        poss = []
        for direction in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            next_pos = (self.pos[0] + direction[0], self.pos[1] + direction[1])
            if not field.is_wall(next_pos):
                poss.append(next_pos)
        return poss


class TD:
    def __init__(self, field_path: Path, epsilon: float = 0.1, gamma: float = 0.95, alpha: float = 0.2) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.field = Field()
        self.field.read_field(field_path)
        h, w = self.field.shape
        self.values = np.zeros((h, w), dtype=np.float64)

        self.agent = Agent()
        self.initialize()

    def initialize(self):
        start_pos = self.field.start_pos
        self.agent.init_pos(start_pos)
        self.agent.log = [start_pos]  # ゴールまで辿り着くごとにログを初期化

    def act(self) -> tuple[int, int]:
        poss = self.agent.next_positions(self.field)
        if self.epsilon < uniform(0, 1):
            next_values: list[float] = [(1 if self.field.is_goal(pos) else 0) + self.gamma * self.values[pos] for pos in poss]
            # when several actions get the same value, select one from them randomly
            next_pos_idx = choice([i for i, next_val in enumerate(next_values) if next_val == max(next_values)])
        else:
            next_pos_idx = int(uniform(0, 1) * len(poss))

        next_pos = poss[next_pos_idx]
        self.agent.log.append(next_pos)
        return next_pos

    def update(self, next_pos: tuple[int, int]) -> None:
        pos = self.agent.pos
        r = 1 if self.field.is_goal(next_pos) else 0
        self.values[pos] += self.alpha * (r + self.gamma * self.values[next_pos] - self.values[pos])
        self.agent.pos = next_pos

    def episode(self) -> int:
        step = 0
        self.initialize()
        while not self.field.is_goal(self.agent.pos):
            next_pos = self.act()
            self.update(next_pos)
            step += 1
        return step

    def train(self) -> tuple[list[int], int]:
        least_repeat = 100
        max_repeat = 500
        th = 0.08
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
        self.field.show_log(self.agent.log)
        self.field.show_values(self.values)
        return steps, episode_num


if __name__ == "__main__":
    import sys

    epsilon = float(sys.argv[2])

    seed(100)
    field_path = Path(sys.argv[1])
    td = TD(field_path, epsilon=epsilon)
    steps, episode_num = td.train()
    _, ax = plt.subplots()
    ax.plot(np.arange(episode_num), steps, "-")
    ax.set_title(r"Learning curve of $\epsilon$-greedy TD(0)", size=18)
    ax.set_xlabel("the number of episodes", size=15)
    ax.set_ylabel("the number of steps", size=15)
    ax.text(0.65, 0.3, f"final step: {steps[-1]}", transform=ax.transAxes, size=15)
    plt.show()
