from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from ...games.tetris import TetrisConfig, TetrisPlacementEnv
from ...network import AdamOptimizer, MLPQNetwork
from .afterstate_utils import evaluate_afterstate_policy
from .expert import choose_expert_action


def parse_hidden_sizes(raw: str) -> Tuple[int, ...]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("hidden sizes cannot be empty")
    return tuple(int(v) for v in values)


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    frac = step / max(decay_steps, 1)
    return start + frac * (end - start)


@dataclass
class Transition:
    state_feat: np.ndarray
    reward: float
    done: bool
    next_feats: list[np.ndarray]


class AfterstateReplay:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.data: list[Transition] = []
        self.ptr = 0

    def add(self, item: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(item)
            return
        self.data[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        if batch_size > len(self.data):
            raise ValueError("batch_size cannot exceed current replay size")
        idx = rng.integers(0, len(self.data), size=batch_size)
        return [self.data[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.data)


def pick_action_training(
    env: TetrisPlacementEnv,
    network: MLPQNetwork,
    rng: np.random.Generator,
    epsilon: float,
    expert_prob: float,
    force_expert: bool,
    include_max_height: bool,
) -> tuple[int, np.ndarray]:
    candidates = env.legal_afterstates(include_max_height=include_max_height)
    if not candidates:
        legal = env.legal_actions()
        action = int(rng.choice(legal)) if legal else 0
        return action, np.zeros(5, dtype=np.float32)

    action_to_feat = {a: feat for a, feat in candidates}
    legal_actions = [a for a, _ in candidates]

    if force_expert:
        action = choose_expert_action(env, placement_actions=True)
        if action not in action_to_feat:
            action = int(legal_actions[0])
        return action, action_to_feat[action]

    if rng.random() < epsilon:
        if rng.random() < expert_prob:
            action = choose_expert_action(env, placement_actions=True)
            if action in action_to_feat:
                return action, action_to_feat[action]
        action = int(rng.choice(legal_actions))
        return action, action_to_feat[action]

    feats = np.stack([feat for _, feat in candidates], axis=0).astype(np.float32)
    values = network.predict_batch(feats)
    best_idx = int(np.argmax([v[0] for v in values]))
    action, feat = candidates[best_idx]
    return int(action), feat


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tetris afterstate DDQN (placement actions)")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--buffer-size", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--target-clip", type=float, default=100.0)
    parser.add_argument(
        "--double-dqn",
        action="store_true",
        default=True,
        help="Use Double DQN target selection for afterstates.",
    )
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.03)
    parser.add_argument("--eps-decay-steps", type=int, default=50_000)
    parser.add_argument("--expert-start", type=float, default=0.80)
    parser.add_argument("--expert-end", type=float, default=0.15)
    parser.add_argument("--expert-decay-steps", type=int, default=50_000)
    parser.add_argument("--expert-warmup-steps", type=int, default=12_000)
    parser.add_argument("--hidden-sizes", type=str, default="256,256")
    parser.add_argument(
        "--state-features",
        choices=["classic4", "extended5"],
        default="classic4",
        help="classic4: lines/height/holes/bumpiness, extended5 adds max_height.",
    )
    parser.add_argument(
        "--reward-style",
        choices=["score_delta", "env"],
        default="score_delta",
        help="score_delta matches common afterstate Tetris RL setups.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="models/tetris_afterstate")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    config = TetrisConfig(height=args.height, width=args.width, max_steps=args.max_steps)
    env = TetrisPlacementEnv(config=config, seed=args.seed)
    env.reset(seed=args.seed)
    include_max_height = args.state_features == "extended5"
    sample_candidates = env.legal_afterstates(include_max_height=include_max_height)
    if not sample_candidates:
        raise RuntimeError("No legal afterstates available after reset")
    feature_dim = sample_candidates[0][1].shape[0]

    online = MLPQNetwork(
        input_dim=feature_dim,
        output_dim=1,
        hidden_sizes=hidden_sizes,
        seed=args.seed,
        dueling=False,
    )
    target = MLPQNetwork(
        input_dim=feature_dim,
        output_dim=1,
        hidden_sizes=hidden_sizes,
        seed=args.seed + 1,
        dueling=False,
    )
    target.copy_from(online)

    optimizer = AdamOptimizer(lr=args.lr, max_grad_norm=args.grad_clip)
    replay = AfterstateReplay(capacity=args.buffer_size)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval = -float("inf")

    print(
        f"Tetris afterstate training | board={args.height}x{args.width} max_steps={args.max_steps} "
        f"feature_dim={feature_dim} reward_style={args.reward_style} double_dqn={args.double_dqn}"
    )

    for episode in range(1, args.episodes + 1):
        env.reset(seed=args.seed + episode)
        done = False
        steps = 0
        episode_reward = 0.0
        total_lines_cleared = 0
        losses: list[float] = []
        td_abs_vals: list[float] = []
        info = {"score": 0, "lines": 0, "lines_cleared": 0}

        while not done and steps < args.max_steps:
            epsilon = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
            expert_prob = epsilon_by_step(global_step, args.expert_start, args.expert_end, args.expert_decay_steps)
            force_expert = global_step < args.expert_warmup_steps
            action, after_feat = pick_action_training(
                env=env,
                network=online,
                rng=rng,
                epsilon=epsilon,
                expert_prob=expert_prob,
                force_expert=force_expert,
                include_max_height=include_max_height,
            )

            _, env_reward, done, info = env.step(action)
            total_lines_cleared += int(info["lines_cleared"])
            lines_cleared = int(info["lines_cleared"])
            if args.reward_style == "score_delta":
                reward = 1.0 + float((lines_cleared * lines_cleared) * args.width)
                if done:
                    reward -= 1.0
            else:
                reward = float(env_reward)

            next_pairs = env.legal_afterstates(include_max_height=include_max_height) if not done else []
            next_feats = [feat for _, feat in next_pairs]
            replay.add(
                Transition(
                    state_feat=after_feat.astype(np.float32),
                    reward=float(reward),
                    done=bool(done),
                    next_feats=next_feats,
                )
            )

            episode_reward += float(reward)
            steps += 1
            global_step += 1

            can_train = len(replay) >= args.batch_size and global_step >= args.warmup_steps
            if can_train and global_step % args.train_every == 0:
                batch = replay.sample(args.batch_size, rng)
                states = np.stack([t.state_feat for t in batch], axis=0).astype(np.float32)
                targets: list[float] = []
                for t in batch:
                    next_q = 0.0
                    if not t.done and t.next_feats:
                        next_arr = np.stack(t.next_feats, axis=0).astype(np.float32)
                        if args.double_dqn:
                            online_next = online.predict_batch(next_arr)
                            best_idx = int(np.argmax([v[0] for v in online_next]))
                            target_next = target.predict_batch(next_arr)
                            next_q = float(target_next[best_idx][0])
                        else:
                            next_values = target.predict_batch(next_arr)
                            next_q = float(max(v[0] for v in next_values))
                        if args.target_clip > 0.0:
                            next_q = float(np.clip(next_q, -args.target_clip, args.target_clip))
                    td_t = float(t.reward) + args.gamma * next_q
                    if args.target_clip > 0.0:
                        td_t = float(np.clip(td_t, -args.target_clip, args.target_clip))
                    targets.append(td_t)

                loss, td_abs = online.train_batch(
                    states=states,
                    actions=np.zeros(args.batch_size, dtype=np.int64),
                    targets=np.asarray(targets, dtype=np.float32),
                    optimizer=optimizer,
                    loss="huber",
                    huber_delta=args.huber_delta,
                )
                losses.append(float(loss))
                td_abs_vals.append(float(td_abs))

            if global_step % args.target_update == 0:
                target.copy_from(online)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_td_abs = float(np.mean(td_abs_vals)) if td_abs_vals else 0.0
        print(
            f"ep={episode:4d} steps={steps:4d} score={int(info['score']):4d} lines={int(info['lines']):4d} "
            f"episode_lines={total_lines_cleared:3d} reward={episode_reward:8.2f} "
            f"eps={epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps):.3f} "
            f"expert={epsilon_by_step(global_step, args.expert_start, args.expert_end, args.expert_decay_steps):.3f} "
            f"loss={mean_loss:.5f} td_abs={mean_td_abs:.5f}"
        )

        if episode % args.eval_every == 0:
            eval_env = TetrisPlacementEnv(config=config, seed=args.seed + 999)
            stats = evaluate_afterstate_policy(
                eval_env,
                online,
                episodes=args.eval_episodes,
                seed_start=args.seed + 7000,
                max_steps=args.max_steps,
            )
            print(
                f"  eval: avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} "
                f"avg_lines={stats.avg_lines:.2f} avg_steps={stats.avg_steps:.2f}"
            )
            if stats.avg_score > best_eval:
                best_eval = stats.avg_score
                best_path = save_dir / "tetris_afterstate_best.json"
                online.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    final_path = save_dir / "tetris_afterstate_final.json"
    online.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
