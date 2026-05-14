#!/usr/bin/env python3
"""Minimal example: load a Risky Overcooked layout and run random agents.

Example:
    python examples/random_agent_example.py --layout risky_mixed_coordination --horizon 50
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from risky_overcooked_py.agents.agent import AgentPair, RandomAgent  # noqa: E402
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv  # noqa: E402
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld  # noqa: E402


def run_random_rollout(layout, horizon, seed, p_slip=None, all_actions=True):
    random.seed(seed)
    np.random.seed(seed)

    mdp_kwargs = {}
    if p_slip is not None:
        mdp_kwargs["p_slip"] = p_slip

    mdp = OvercookedGridworld.from_layout_name(layout, **mdp_kwargs)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)

    agents = AgentPair(
        RandomAgent(all_actions=all_actions),
        RandomAgent(all_actions=all_actions),
    )
    agents.set_mdp(env.mdp)

    env.reset(regen_mdp=False)
    agents.reset()

    done = False
    total_reward = 0.0
    steps = 0
    info = {}

    while not done:
        joint_action_and_infos = agents.joint_action(env.state)
        joint_action, joint_action_info = zip(*joint_action_and_infos)
        _next_state, reward, done, info = env.step(joint_action, joint_action_info)
        total_reward += reward
        steps += 1

    episode = info.get("episode", {})
    return {
        "layout": layout,
        "horizon": horizon,
        "seed": seed,
        "p_slip": mdp.p_slip,
        "steps": steps,
        "total_reward": total_reward,
        "episode_sparse_reward": episode.get("ep_sparse_r"),
        "episode_shaped_reward": episode.get("ep_shaped_r"),
        "subgoals": sorted(mdp.terrain_pos_dict.get("G", []), key=lambda p: (p[1], p[0])),
        "waters": sorted(mdp.water_disable_timers.keys(), key=lambda p: (p[1], p[0])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a random-agent rollout in a Risky Overcooked layout."
    )
    parser.add_argument("--layout", default="risky_mixed_coordination")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-slip", type=float, default=None, help="Override layout p_slip")
    parser.add_argument(
        "--motion-only",
        action="store_true",
        help="Sample only movement/stay actions instead of all actions.",
    )
    args = parser.parse_args()

    result = run_random_rollout(
        layout=args.layout,
        horizon=args.horizon,
        seed=args.seed,
        p_slip=args.p_slip,
        all_actions=not args.motion_only,
    )

    print("=== Random Agent Rollout ===")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
