"""Neuroevolution racing simulator core.

Hard rule: nothing in this package imports pygame or matplotlib. The core is
headless, deterministic under a seed, and fully vectorized with numpy — the
renderers (watch.py, play.py) are pure observers that live outside the package.
"""
