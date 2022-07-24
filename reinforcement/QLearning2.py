from pathlib import Path
from random import choice, seed, uniform

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# posは[y, x]の順番

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

action2direction: dict[int, tuple[int, int]] = {RIGHT: (0, 1), LEFT: (0, -1), UP: (-1, 0), DOWN: (1, 0)}


class Agent:
    def __init__(self) -> None:
        self.log: list[tuple[int, int]] = []

    def init_pos(self, start: tuple[int, int]) -> None:
        self.pos = start


class QLearning2:
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
        self.q_values = np.zeros((h, w, 4), dtype=np.float64)

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

    @property
    def start_pos(self) -> tuple[int, int]:
        start_arr = np.where(self.field == QLearning2.start)
        return (start_arr[0][0], start_arr[1][0])

    def is_wall(self, pos: tuple[int, int]) -> bool:
        return self.field[pos] == QLearning2.wall

    def is_goal(self, pos: tuple[int, int]) -> bool:
        return self.field[pos] == QLearning2.goal

    def initialize(self):
        start_pos = self.start_pos
        self.agent.init_pos(start_pos)
        self.agent.log = [start_pos]  # ゴールまで辿り着くごとにログを初期化

    def next_actions(self) -> list[int]:
        pos_now = self.agent.pos
        actions = []
        for action, direction in action2direction.items():
            next_pos = (pos_now[0] + direction[0], pos_now[1] + direction[1])
            if not self.is_wall(next_pos):
                actions.append(action)
        return actions

    def act(self) -> tuple[int, tuple[int, int]]:
        actions = self.next_actions()
        pos = self.agent.pos
        pos_q_values: np.ndarray = self.q_values[pos][actions]  # get only movable action by indexing with list ([actions])
        if self.epsilon < uniform(0, 1):
            action: int = choice([action for action, q_value in zip(actions, pos_q_values) if q_value == max(pos_q_values)])
        else:
            action = choice(actions)

        direction = action2direction[action]
        next_pos = (pos[0] + direction[0], pos[1] + direction[1])
        self.agent.log.append(next_pos)
        return action, next_pos

    def update(self, action: int, next_pos: tuple[int, int]) -> None:
        pos = self.agent.pos
        r = 1 if self.is_goal(next_pos) else 0
        self.q_values[pos][action] += self.alpha * (r + self.gamma * max(self.q_values[next_pos]) - self.q_values[pos][action])
        self.agent.pos = next_pos

    def episode(self) -> int:
        step = 0
        self.initialize()
        while not self.is_goal(self.agent.pos):
            action, next_pos = self.act()
            self.update(action, next_pos)
            step += 1
        return step

    def train(self) -> tuple[list[int], int]:
        least_repeat = 60
        prev_values = np.zeros_like(self.q_values)
        th = 0.01
        episode_num = 0
        steps = []
        while episode_num < least_repeat or np.abs(self.q_values - prev_values).sum() > th:
            prev_values = self.q_values.copy()
            step = self.episode()
            steps.append(step)
            episode_num += 1
        return steps, episode_num

    def show_q_values(self) -> None:
        cmap = ListedColormap(["#00000000", "skyblue", "pink", "black"])
        plt.gca().set_aspect("equal")
        plt.xticks([])
        plt.yticks([])
        plt.gca().invert_yaxis()

        box = np.zeros((3, 3))
        h, w, _ = self.q_values.shape
        for y in np.arange(h):
            for x in np.arange(w):
                for action, q_value in enumerate(self.q_values[y, x]):
                    if action == LEFT:
                        box[1, 0] = q_value
                    elif action == RIGHT:
                        box[1, 2] = q_value
                    elif action == UP:
                        box[0, 1] = q_value
                    else:
                        box[2, 1] = q_value
                box_w = 1 / 3
                grid = np.arange(box_w / 2, 3 * box_w, box_w)
                plt.pcolormesh(grid + x, grid + y, box, cmap="Greens", edgecolors="lightgray", linewidth=0.05)
        plt.pcolormesh(np.arange(w) + 0.5, np.arange(h) + 0.5, self.field, cmap=cmap, edgecolors="gray", linewidth=0.1)
        plt.show()

    def show_log(self) -> None:
        cmap = ListedColormap(["white", "skyblue", "pink", "black"])
        _, ax = plt.subplots()
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


if __name__ == "__main__":
    import sys

    epsilon = float(sys.argv[2])
    seed(100)

    field_path = Path(sys.argv[1])
    q_learning = QLearning2(field_path, epsilon=epsilon)
    steps, episode_num = q_learning.train()
    q_learning.show_q_values()
    q_learning.show_log()
    _, ax = plt.subplots()
    ax.set_title(r"Learning curve of Q-learning", size=18)
    ax.plot(np.arange(episode_num), steps, "-")
    ax.set_xlabel("the number of episodes", size=15)
    ax.set_ylabel("the number of steps", size=15)
    ax.text(0.65, 0.3, f"final step: {steps[-1]}", transform=ax.transAxes, size=15)
    plt.show()
