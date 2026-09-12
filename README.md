# Adaptive Personalized Learning Recommendation Using Offline Reinforcement Learning

**A Comparative Analysis of Q-Learning and Monte Carlo Control on EdNet-KT3**

This project formulates personalized learning recommendation as an **offline reinforcement learning (RL)** problem. Student interaction histories from the [EdNet-KT3](https://github.com/riiid/ednet) dataset are transformed into bundle-level learning trajectories with leakage-safe states, observed learning actions, rewards, and next states. Two classical model-free RL algorithms — **Q-Learning** and **Monte Carlo Control** — are trained and evaluated under an identical empirical offline environment, with a **Random policy** baseline for comparison.

---

## Overview

Traditional e-learning systems often rely on static or predefined learning sequences that don't adapt to a student's changing performance. This project instead treats learning recommendation as a sequential decision-making problem: at each step, an agent observes a student's learning state and recommends one of three directly observable activities — **QUIZ**, **PRACTICE**, or **REVISION** — with the goal of maximizing long-term learning outcomes.

Because EdNet-KT3 is a historical interaction log rather than an interactive simulator, an **empirical offline transition model** is built from training-student trajectories, and **action masking** restricts each agent to state-action pairs actually supported by the data — avoiding fabricated outcomes for actions that were never observed.

## Key Contributions

- **Bundle-level offline RL formulation** — question interactions are grouped into learning bundles rather than treated as independent low-level events.
- **Leakage-safe temporal state construction** — each state is built only from a student's history *before* the current bundle; the bundle's outcome is used solely for the reward, never the state.
- **Controlled comparison of Q-Learning vs. Monte Carlo Control** under identical state representation, action space, reward function, and student-level data split.
- **Empirical offline transition environment** built exclusively from training students, with validation/test students held out from environment construction.
- **Comprehensive experimental analysis**: hyperparameter sensitivity, convergence analysis, train/validation/test generalization tracking, ablation study, and multi-seed evaluation (5 seeds).
- **Explicit discussion of offline-evaluation limitations**, including the lack of counterfactual outcomes and the difference between behavioral agreement and true recommendation accuracy.

## Dataset

- **Source:** [EdNet-KT3](https://github.com/riiid/ednet) (Choi et al., AIED 2020) — student interaction logs from the Santa AI-based self-learning platform.
- **Scope used:** 100 student interaction files (for computational feasibility).
- **After preprocessing:** 51,523 question attempts, 1,547 lecture sessions, 40,037 explanation sessions.
- **Final RL dataset:** 25,736 transitions from 98/100 students (2 students excluded for having fewer than 5 transitions), spanning ~600 unique discretized student states and 3 observable actions.

## Methodology

| Component | Description |
|---|---|
| **State** | Discretized (0/1/2) recent accuracy, skill mastery, learning progression, and a weak-skill "learning history" identifier — built only from information available before the current bundle |
| **Actions** | `QUIZ`, `PRACTICE`, `REVISION` (the fourth conceptual action, `LESSON`, was never observed in the source data and is not used for training) |
| **Reward** | `r_t = 2·ΔAccuracy_t + 5·𝟙(Accuracy_t ≥ 0.80 ∧ Accuracy_{t-1} < 0.80)` — rewards accuracy improvement plus a bonus for crossing an 80% mastery threshold |
| **Transition model** | Empirical, resampled from observed `(state, action) → (reward, next_state, done)` outcomes in the training split only |
| **Split** | Student-level 68% train / 15% validation / 15% test (no student appears in more than one split) |

### Hyperparameters (main configuration)

| Parameter | Value |
|---|---|
| Discount factor (γ) | 0.95 |
| Learning rate (α, Q-Learning only) | 0.10 |
| Training episodes | 3,000 |
| Max steps per episode | 30 |
| Epsilon schedule | 1.00 → 0.05 (decay 0.995) |
| Random seeds | 5 |

## Results

**Single-seed headline comparison** (random state 42):

| Metric | Q-Learning | Monte Carlo Control |
|---|---|---|
| Test Average Return | 30.04 | 28.52 |
| Logged-Action Agreement | 87.9% | 80.9% |
| Empirical-Best Agreement | 92.7% | 85.5% |
| Mastery Completion Proxy | 46.8% | 44.2% |
| Training Time | 0.51s | 0.37s |

**Multi-seed evaluation** (5 seeds, mean ± SD) — the more reliable comparison:

| Metric | Q-Learning | Monte Carlo Control |
|---|---|---|
| Test Average Return | 31.88 ± 0.92 | 30.86 ± 0.78 |
| Mastery Success Rate | 43.1% ± 2.7% | 42.0% ± 2.4% |

Both algorithms outperform a Random baseline (42.4% mastery completion proxy). Averaged across seeds, **Q-Learning shows a modest, seed-consistent advantage in average return**, while the two algorithms are **not reliably distinguishable on mastery success rate**. An ablation study further shows that removing the weak-skill "learning history" dimension from the state produces by far the largest and most seed-robust *increase* in test return (+6.31), pointing to state-space fragmentation as a key limitation of the current tabular representation.

See the full paper (`QLMC.pdf`) for hyperparameter sensitivity, convergence curves, and the complete ablation study.

## Repository Structure

```
.
├── QLMC.ipynb              # End-to-end pipeline: preprocessing, RL transition
│                            # construction, Q-Learning, Monte Carlo Control,
│                            # baseline comparison, and evaluation
├── QLMC.pdf                # Full write-up / report
└── README.md
```

## How to Run

The notebook is designed to run on **Kaggle**, reading EdNet-KT3 data from `/kaggle/input/` and writing processed outputs to `/kaggle/working/`.

1. Upload the EdNet-KT3 `questions.csv`, `lectures.csv`, and per-student `u*.csv` interaction files as a Kaggle dataset (or adjust `INPUT_DIR` for a local run).
2. Open `QLMC.ipynb` and run all cells top to bottom.
3. Key configuration knobs are set near the top of the notebook:
   - `MAX_STUDENTS` — number of student files to process (default: 100)
   - `MIN_EPISODE_LENGTH` — minimum transitions required to keep a student (default: 5)
   - `RANDOM_STATE` — seed for the main single-seed run (default: 42)
4. Processed datasets, comparison tables, and figures are saved under `/kaggle/working/cse753_processed/`.

## Limitations

- **Offline evaluation only** — policies are evaluated against an empirical simulator built from historical data, not through live deployment to real students.
- **No counterfactual outcomes** — the dataset only records what students actually did, so causal superiority of one recommendation over another cannot be established.
- **Restricted action space** — only QUIZ, PRACTICE, and REVISION are used as trainable actions; lecture/explanation activities are excluded to avoid fabricated outcomes.
- **Small student sample** (100 students) limits generalizability.
- **Tabular state representation** is discretized and simplified relative to a student's actual learning condition.

See Section 4 of the paper for the full discussion of limitations and future directions (larger datasets, richer sequential state representations, offline RL methods such as CQL/BCQ, and eventual online/human-subject evaluation).

## Citation

If you use this work, please cite:

```
Islam, S., & Nidhi, R. S. Adaptive Personalized Learning Recommendation Using
Reinforcement Learning: A Comparative Analysis of Q-Learning and Monte Carlo Control.
```

## References

1. Tang, X., Chen, Y., Li, X., Liu, J., & Ying, Z. (2019). A reinforcement learning approach to personalized learning recommendation systems. *British Journal of Mathematical and Statistical Psychology*, 72(1), 108–135.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
3. Piech, C., et al. (2015). Deep knowledge tracing. *NeurIPS*, 28.
4. Tan, C., Han, R., Ye, R., & Chen, K. (2020). Adaptive learning recommendation strategy based on deep Q-learning. *Applied Psychological Measurement*, 44(4), 251–266.
5. Chen, Y., Li, X., Liu, J., & Ying, Z. (2018). Recommendation system for adaptive learning. *Applied Psychological Measurement*, 42(1), 24–41.
6. Riedmann, A., Schaper, P., & Lugrin, B. (2025). Reinforcement learning in education: A systematic literature review. *International Journal of Artificial Intelligence in Education*.
7. Choi, Y., et al. (2020). EdNet: A large-scale hierarchical dataset in education. *AIED*.

## Authors

- Shamazda Islam
- Ramisa Sharar Nidhi
