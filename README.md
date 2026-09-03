# Adaptive Personalized Learning Recommendation System (EdNet-KT3)
### Comparative Analysis of Q-Learning and Monte Carlo Control

This repository/notebook implements an offline reinforcement learning
pipeline that recommends the next learning activity (quiz, practice, or
revision) to a student, trained and evaluated on the [EdNet-KT3
dataset](https://arxiv.org/abs/1912.03072). Two model-free RL algorithms —
**Q-Learning** and **Monte Carlo Control** — are trained on the same
environment and compared against each other and a random-policy baseline.

## Contents

| File | Description |
|---|---|
| `Q-Learning_and_Monte_Carlo_Control.ipynb` | Full pipeline: data loading, preprocessing, RL environment construction, agent training, evaluation |

## Pipeline overview

The notebook runs end-to-end in the following stages:

1. **Data loading** — reads EdNet-KT3 question/lecture metadata and raw
   per-student interaction logs (`MAX_STUDENTS` students by default, for
   fast iteration on Kaggle).
2. **Question attempt reconstruction** — converts low-level `enter`/`respond`/`submit`
   events into scored question attempts, merged with question metadata (skill tags, part, bundle).
3. **Skill-level mastery tracking** — expands multi-skill questions into
   per-skill records and computes skill mastery and **recency-safe**
   recent-accuracy features (no use of future information).
4. **Lecture and explanation sessions** — reconstructs lecture-viewing and
   explanation/revision sessions from raw events, with completion-ratio estimates.
5. **RL state and action construction**
   - **State**: `(mastery_level, recent_accuracy_level, lesson_level, weak_skill_id)` — discretized, built only from information available *before* each bundle attempt (leakage-safe).
   - **Action space**: `QUIZ`, `PRACTICE`, `REVISION`. `LESSON` is intentionally **excluded** — lecture events have no graded outcome to compute a reward from, so it is not represented as a bundle-level action (see the action-coverage diagnostic cell for details).
   - **Reward**: `2 × Δaccuracy + mastery bonus (5.0 on crossing the mastery threshold)`.
6. **Transition table construction** — links each state to the next state,
   producing `(state, action, reward, next_state, done)` rows per student.
7. **Student-level train/validation/test split** — split by `user_id` (not
   by row), so no student's transitions appear in more than one split.
8. **RL environment (`StudentEnv`)** — an empirical MDP simulator built by
   resampling observed `(reward, next_state, done)` outcomes for each
   `(state, action)` pair seen in the training data.
   - **Action masking**: agents are restricted to `available_actions(state)`
     — only actions with real logged support for that state. This was a
     deliberate fix: an earlier version fell back to a same-state
     "self-loop" for unsupported actions, which let agents farm reward
     without genuine progress (see the Results section below).
9. **Agents** — tabular **Q-Learning** (off-policy TD(0), epsilon-greedy)
   and **Monte Carlo Control** (on-policy, first-visit, epsilon-greedy),
   trained online against the simulator with matched hyperparameters.
10. **Evaluation**
    - Learning curves (rolling-mean training reward) and per-algorithm
      convergence plots (`qlearning_convergence.png`, `mc_convergence.png`).
    - Held-out validation/test mean reward under the learned greedy policy.
    - Recommendation accuracy: agreement with the logged (historical) action
      and with the empirically best-reward action per state.
    - Completion-rate proxy: fraction of episodes reaching top mastery
      level under the greedy policy vs. a random-action baseline.

## How to run

1. Open the notebook on Kaggle with the EdNet-KT3 dataset attached as an
   input (or point `EDNET_ROOT` at a local copy).
2. Run all cells top to bottom. Processed tables, figures, and the final
   comparison CSV are written to `OUTPUT_DIR` (`/kaggle/working/cse753_processed/`
   by default).
3. To reduce runtime while iterating, lower `MAX_STUDENTS` (data pipeline)
   or `N_EPISODES` (agent training) near the top of the relevant cells.

## Key hyperparameters (main run)

| Parameter | Value |
|---|---|
| `N_EPISODES` | 3000 |
| `MAX_STEPS` per episode | 30 |
| `GAMMA` (discount factor) | 0.95 |
| `ALPHA` (Q-Learning learning rate) | 0.1 |
| Epsilon schedule | 1.0 → 0.05, decay 0.995/episode |
| `RANDOM_STATE` (seed) | fixed for reproducibility |

## Results (main run, 100-student dev subset)

| Metric | Q-Learning | Monte Carlo Control | Random baseline |
|---|---|---|---|
| Train reward (last 100 episodes) | 37.43 | 26.96 | — |
| Test mean reward | 30.04 | 28.52 | — |
| Recommendation accuracy vs. logged action | 0.879 | 0.809 | — |
| Recommendation accuracy vs. empirically-best action | 0.927 | 0.855 | — |
| Completion-rate proxy | **0.494** | 0.460 | 0.424 |
| Training time | 0.45s | 0.32s | — |

Both trained agents outperform the random baseline on the completion-rate
proxy, with Q-Learning ahead of Monte Carlo Control.

## Known limitations

- **Action space imbalance**: `PRACTICE` accounts for ~91% of logged
  transitions, `QUIZ` ~8%, `REVISION` <1%. `REVISION` in particular has very
  little data to learn from.
- **State sparsity**: ~600 unique states over ~25,700 transitions on the
  100-student dev subset; coverage would improve with more students
  (`MAX_STUDENTS` increased) or a coarser state discretization.
- **`LESSON` action excluded** — see Pipeline overview, step 5.
- **Single-seed run** for the headline comparison table above; hyperparameter
  sensitivity and an action-masking ablation study are provided separately
  to support robustness claims (multiple `alpha`/`gamma` configurations per
  algorithm, and a masked-vs-unmasked environment comparison).
- Evaluation uses a **simulated environment** fit from empirical training
  transitions, not a live student population — results describe policy
  quality within this offline model, not a guarantee of real-world outcomes.

## Dataset

[EdNet-KT3](https://arxiv.org/abs/1912.03072) — Choi et al., "EdNet: A
Large-Scale Hierarchical Dataset in Education," AIED 2020.
