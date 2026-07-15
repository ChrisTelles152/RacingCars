"""Shared pygame rendering for watch.py and play.py.

Lives OUTSIDE the racing/ package on purpose: the simulation core is headless
and never imports pygame. Viewers are pure observers — they read SimState
arrays and draw; nothing here feeds back into the simulation.

The track is rendered directly from the sensor occupancy grid, so the picture
on screen is exactly the world the cars sense and crash against (not a
separate drawing that could quietly disagree with the physics).
"""

from __future__ import annotations

import numpy as np
import pygame

from racing.sensors import ray_geometry, sense
from racing.track import Track

BG = (16, 17, 22)
TRACK = (58, 60, 70)
START = (235, 235, 235)
CENTER = (74, 77, 90)
CAR = (80, 190, 255)
BEST = (255, 205, 60)
DEAD = (105, 62, 62)
RAY = (110, 220, 130)
RAY_HIT = (240, 120, 90)
TEXT = (225, 225, 225)


class Viewer:
    def __init__(self, world_size: int, window: int = 880, title: str = "RacingCars"):
        pygame.init()
        self.world_size = world_size
        self.window = window
        self.scale = window / world_size
        self.screen = pygame.display.set_mode((window, window))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("menlo, monaco, monospace", 14)
        self.clock = pygame.time.Clock()
        self._track_surface: pygame.Surface | None = None
        self._track_id: int | None = None

    # --- coordinate transform -------------------------------------------------
    def w2s(self, pts: np.ndarray) -> np.ndarray:
        """World -> screen pixels. World y points down, same as the screen."""
        return np.asarray(pts, dtype=np.float64) * self.scale

    # --- track ------------------------------------------------------------------
    def _build_track_surface(self, track: Track) -> pygame.Surface:
        g = track.occ_sensor.shape[0]
        rgb = np.empty((g, g, 3), dtype=np.uint8)
        rgb[track.occ_sensor] = BG
        rgb[~track.occ_sensor] = TRACK
        # surfarray expects (x, y, 3); our grid is (y, x) — transpose.
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        surf = pygame.transform.smoothscale(surf, (self.window, self.window))

        # Faint centerline checkpoints + a start/finish line.
        for p in self.w2s(track.centerline[::8]):
            surf.set_at((int(p[0]), int(p[1])), CENTER)
        hw0 = (track.half_widths[0] if track.half_widths is not None
               else track.half_width)  # local width at the start line
        a = self.w2s(track.centerline[0] + track.normals[0] * hw0)
        b = self.w2s(track.centerline[0] - track.normals[0] * hw0)
        pygame.draw.line(surf, START, a, b, 2)
        return surf

    def draw_track(self, track: Track) -> None:
        if self._track_id != id(track):
            self._track_surface = self._build_track_surface(track)
            self._track_id = id(track)
        self.screen.blit(self._track_surface, (0, 0))

    # --- cars ---------------------------------------------------------------
    def draw_cars(self, pos: np.ndarray, heading: np.ndarray, alive: np.ndarray,
                  best_idx: int | None = None) -> None:
        """Cars as heading-aligned triangles; dead cars faded, best highlighted."""
        shape = np.array([[12.0, 0.0], [-7.0, 6.0], [-7.0, -6.0]])  # car frame, px
        cos, sin = np.cos(heading), np.sin(heading)
        order = np.argsort(alive.astype(int))  # draw dead first, alive on top
        for i in order:
            rot = np.array([[cos[i], -sin[i]], [sin[i], cos[i]]])
            tri = self.w2s(pos[i] + shape @ rot.T)
            if best_idx is not None and i == best_idx:
                color = BEST
            elif alive[i]:
                color = CAR
            else:
                color = DEAD
            pygame.draw.polygon(self.screen, color, tri)

    def draw_rays(self, track: Track, pos: np.ndarray, heading: np.ndarray,
                  sensor_cfg, indices) -> None:
        """Sensor rays for selected cars, recomputed just for display."""
        rel, ts, lengths = ray_geometry(sensor_cfg)
        idx = np.atleast_1d(indices)
        dist = sense(pos[idx], heading[idx], track.occ_sensor, sensor_cfg,
                     rel, ts, lengths)
        for row, i in enumerate(idx):
            angles = heading[i] + rel
            for r in range(len(rel)):
                d = dist[row, r] * lengths[r]
                end = pos[i] + d * np.array([np.cos(angles[r]), np.sin(angles[r])])
                hit = dist[row, r] < 0.999
                pygame.draw.line(self.screen, RAY_HIT if hit else RAY,
                                 self.w2s(pos[i]), self.w2s(end), 1)

    # --- chrome ---------------------------------------------------------------
    def draw_hud(self, lines: list[str]) -> None:
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, TEXT), (10, 8 + 18 * i))

    def flip(self, fps: int | None = None) -> None:
        pygame.display.flip()
        if fps:
            self.clock.tick(fps)

    @staticmethod
    def poll() -> tuple[bool, list[int]]:
        """Pump events. Returns (keep_running, key_presses this frame)."""
        keys: list[int] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, keys
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, keys
                keys.append(event.key)
        return True, keys
