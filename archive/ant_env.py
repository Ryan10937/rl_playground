"""AntForagingEnv — a Gymnasium env where an ant searches a 2D grid for food
with a limited sight radius and randomly placed walls.

Tile encoding (uint8) used throughout:
    0 = empty
    1 = wall
    2 = food
    3 = ant   (only ever appears in the *rendered* view, never the base map)

Observation:
    A (2r+1) x (2r+1) egocentric window around the ant. Cells outside the
    map are encoded as walls (1). The ant's own cell is encoded as 3.

Action space (Discrete(4)):
    0 = up (-y)
    1 = right (+x)
    2 = down (+y)
    3 = left (-x)

Rewards:
    step_penalty (default -1.0) on every step
    food_reward  (default +50.0) when the ant moves onto food
    wall_penalty (default -5.0)  when the ant attempts to leave the map
                                   *or* walk into a wall — the ant stays put

Episode ends (terminated) when all food has been consumed.
A `max_steps` truncation is also supported.

`info` dict on every step contains:
    "food_eaten":      cumulative count
    "distance_traveled": cumulative count of *successful* moves
    "wall_hits":       cumulative count of out-of-bounds *or* wall attempts
    "food_remaining":  food still on the map
    "ant_pos":         (x, y) tuple
    "moved":           bool — did this step actually move the ant
    "ate_food":        bool — did this step consume food
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Tile constants
EMPTY = 0
WALL = 1
FOOD = 2
ANT = 3

# Action constants
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

# (dx, dy) per action, with y increasing downward (row index)
_ACTION_DELTAS = {
    ACTION_UP:    (0, -1),
    ACTION_RIGHT: (1,  0),
    ACTION_DOWN:  (0,  1),
    ACTION_LEFT:  (-1, 0),
}


@dataclass
class AntForagingConfig:
    """Portable configuration for an AntForagingEnv.

    The env can either *generate* a map from these params (when `grid` is
    None) or use the supplied `grid` verbatim. Either way the config object
    fully describes the environment and can be pickled / JSON-roundtripped.
    """
    # Map sizing
    width: int = 20
    height: int = 20
    sight_radius: int = 3

    # Procedural generation (used when `grid` is None)
    n_food: int = 5
    wall_density: float = 0.15       # fraction of non-ant tiles that are walls
    ant_start: Optional[tuple[int, int]] = None  # (x, y); None => random empty cell

    # Rewards
    step_penalty: float = -1.0
    food_reward: float = 50.0
    wall_penalty: float = -5.0

    # Episode
    max_steps: int = 500

    # Optional fully-specified map. If provided, width/height/n_food/wall_density
    # are inferred from it and the generator is bypassed. Must contain only
    # values in {EMPTY, WALL, FOOD}. Shape is (height, width).
    grid: Optional[list[list[int]]] = field(default=None)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AntForagingConfig":
        return cls(**d)


class AntForagingEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 8}

    def __init__(self, config: AntForagingConfig | dict | None = None, render_mode: str | None = None):
        super().__init__()

        if config is None:
            config = AntForagingConfig()
        elif isinstance(config, dict):
            config = AntForagingConfig.from_dict(config)
        self.config = config
        self.render_mode = render_mode

        # If a grid was supplied, lock width/height to it.
        if config.grid is not None:
            grid_arr = np.asarray(config.grid, dtype=np.uint8)
            assert grid_arr.ndim == 2, "grid must be 2D"
            self.height, self.width = grid_arr.shape
        else:
            self.width = int(config.width)
            self.height = int(config.height)
        assert self.width > 0 and self.height > 0

        self.sight_radius = int(config.sight_radius)
        assert self.sight_radius >= 0

        # Spaces
        win = 2 * self.sight_radius + 1
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(win, win), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

        # Filled in by reset()
        self._grid: np.ndarray | None = None
        self._ant_xy: tuple[int, int] | None = None
        self._food_eaten = 0
        self._distance = 0
        self._wall_hits = 0
        self._food_remaining = 0
        self._steps = 0

    # ----- map generation ------------------------------------------------
    def _generate_map(self, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
        cfg = self.config

        if cfg.grid is not None:
            grid = np.asarray(cfg.grid, dtype=np.uint8).copy()
        else:
            grid = np.zeros((self.height, self.width), dtype=np.uint8)
            total = self.width * self.height
            n_walls = int(round(cfg.wall_density * total))
            n_walls = max(0, min(n_walls, total - 1 - cfg.n_food))  # leave room
            flat_idx = rng.choice(total, size=n_walls + cfg.n_food, replace=False)
            wall_idx = flat_idx[:n_walls]
            food_idx = flat_idx[n_walls:]
            ys, xs = np.unravel_index(wall_idx, (self.height, self.width))
            grid[ys, xs] = WALL
            ys, xs = np.unravel_index(food_idx, (self.height, self.width))
            grid[ys, xs] = FOOD

        # Pick ant start
        if cfg.ant_start is not None:
            ax, ay = cfg.ant_start
            assert 0 <= ax < self.width and 0 <= ay < self.height, "ant_start out of bounds"
            # Don't let the ant start on a wall; if so, clear that cell.
            if grid[ay, ax] == WALL:
                grid[ay, ax] = EMPTY
            # If it lands on food, eat it immediately would be weird — just clear.
            if grid[ay, ax] == FOOD:
                grid[ay, ax] = EMPTY
            ant_xy = (ax, ay)
        else:
            empties = np.argwhere(grid == EMPTY)
            if len(empties) == 0:
                # extreme edge case: make room
                grid[0, 0] = EMPTY
                ant_xy = (0, 0)
            else:
                pick = empties[rng.integers(len(empties))]
                ant_xy = (int(pick[1]), int(pick[0]))

        return grid, ant_xy

    # ----- gym API -------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random  # seeded by super().reset

        # Optional per-episode override of the grid via options
        if options is not None and "grid" in options and options["grid"] is not None:
            saved = self.config.grid
            self.config.grid = options["grid"]
            try:
                self._grid, self._ant_xy = self._generate_map(rng)
            finally:
                self.config.grid = saved
            self.height, self.width = self._grid.shape
        else:
            self._grid, self._ant_xy = self._generate_map(rng)

        self._food_eaten = 0
        self._distance = 0
        self._wall_hits = 0
        self._food_remaining = int(np.sum(self._grid == FOOD))
        self._steps = 0

        return self._get_obs(), self._get_info(moved=False, ate_food=False)

    def step(self, action: int):
        assert self._grid is not None, "Call reset() before step()."
        assert self.action_space.contains(int(action)), f"Invalid action {action}"

        cfg = self.config
        dx, dy = _ACTION_DELTAS[int(action)]
        ax, ay = self._ant_xy
        nx, ny = ax + dx, ay + dy

        reward = float(cfg.step_penalty)
        moved = False
        ate_food = False

        out_of_bounds = not (0 <= nx < self.width and 0 <= ny < self.height)
        blocked_by_wall = (not out_of_bounds) and self._grid[ny, nx] == WALL

        if out_of_bounds or blocked_by_wall:
            reward += float(cfg.wall_penalty)
            self._wall_hits += 1
            # Ant stays still
        else:
            self._ant_xy = (nx, ny)
            self._distance += 1
            moved = True
            if self._grid[ny, nx] == FOOD:
                reward += float(cfg.food_reward)
                self._grid[ny, nx] = EMPTY
                self._food_eaten += 1
                self._food_remaining -= 1
                ate_food = True

        self._steps += 1
        terminated = self._food_remaining <= 0
        truncated = self._steps >= int(cfg.max_steps) and not terminated

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(moved=moved, ate_food=ate_food),
        )

    # ----- helpers -------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Return a (2r+1, 2r+1) egocentric view. Off-map cells are walls."""
        r = self.sight_radius
        size = 2 * r + 1
        ax, ay = self._ant_xy
        obs = np.full((size, size), WALL, dtype=np.uint8)

        # Compute overlap between the window and the grid
        x0, x1 = ax - r, ax + r + 1  # grid x range
        y0, y1 = ay - r, ay + r + 1  # grid y range
        gx0, gx1 = max(0, x0), min(self.width, x1)
        gy0, gy1 = max(0, y0), min(self.height, y1)
        if gx0 < gx1 and gy0 < gy1:
            ox0, oy0 = gx0 - x0, gy0 - y0
            ox1, oy1 = ox0 + (gx1 - gx0), oy0 + (gy1 - gy0)
            obs[oy0:oy1, ox0:ox1] = self._grid[gy0:gy1, gx0:gx1]

        # Mark ant at center
        obs[r, r] = ANT
        return obs

    def _get_info(self, moved: bool, ate_food: bool) -> dict[str, Any]:
        return {
            "food_eaten": int(self._food_eaten),
            "distance_traveled": int(self._distance),
            "wall_hits": int(self._wall_hits),
            "food_remaining": int(self._food_remaining),
            "ant_pos": tuple(self._ant_xy) if self._ant_xy is not None else None,
            "moved": bool(moved),
            "ate_food": bool(ate_food),
            "steps": int(self._steps),
        }

    # ----- rendering -----------------------------------------------------
    _GLYPHS = {EMPTY: ".", WALL: "#", FOOD: "*", ANT: "A"}

    def render(self):
        if self.render_mode == "ansi" or self.render_mode is None:
            return self._render_ansi()
        if self.render_mode == "human":
            print(self._render_ansi())
            return None
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        raise ValueError(f"Unknown render mode {self.render_mode!r}")

    def _render_ansi(self) -> str:
        assert self._grid is not None
        view = self._grid.copy()
        ax, ay = self._ant_xy
        view[ay, ax] = ANT
        lines = ["".join(self._GLYPHS[int(c)] for c in row) for row in view]
        return "\n".join(lines)

    def _render_rgb(self) -> np.ndarray:
        """Return an HxWx3 uint8 image: empty=white, wall=black, food=green, ant=red."""
        assert self._grid is not None
        palette = {
            EMPTY: (245, 245, 245),
            WALL:  (30, 30, 30),
            FOOD:  (60, 180, 75),
            ANT:   (220, 50, 50),
        }
        view = self._grid.copy()
        ax, ay = self._ant_xy
        view[ay, ax] = ANT
        img = np.zeros((*view.shape, 3), dtype=np.uint8)
        for code, col in palette.items():
            img[view == code] = col
        return img
