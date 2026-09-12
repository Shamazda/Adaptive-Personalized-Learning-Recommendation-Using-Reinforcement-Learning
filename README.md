# Adaptive Personalized Learning Recommendation Using Offline Reinforcement Learning:

**A Comparative Analysis of Q-Learning and Monte Carlo Control**

This repository contains the complete implementation of an offline reinforcement learning (RL) framework for personalized learning recommendation, built on the [EdNet-KT3](https://github.com/riiid/ednet) dataset. Two classical model-free RL algorithms — **Q-Learning** and **Monte Carlo Control** — are trained and compared against a **Random policy baseline** on the task of recommending the next educational activity (QUIZ, PRACTICE, or REVISION) to a student, based on their learning state.

---

## Overview

Online learning platforms give students access to large collections of educational resources, but students differ in pace, prior knowledge, and performance — a fixed activity sequence is rarely optimal for everyone. This project formulates personalized learning recommendation as a **sequential decision-making problem**: an RL agent observes a student's current learning state, selects an educational activity, receives feedback based on the outcome, and updates its recommendation policy.

The pipeline processes raw EdNet-KT3 interaction logs into leakage-safe RL transitions `(sₜ, aₜ, rₜ, sₜ₊₁)`, trains Q-Learning and Monte Carlo Control on an empirical offline transition model, and evaluates both algorithms across multiple metrics, hyperparameter configurations, ablations, and random seeds.

*(See Figure 1 in the report for the full nine-stage pipeline: EdNet-KT3 → Preprocessing → Question-Bundle Construction → Student Performance/Skill Features → Leakage-Safe State Construction → RL Transition Generation → Student-Level Data Split → Q-Learning/Monte Carlo Control → Offline Evaluation and Comparison.)*

---

## Key Contributions

1. **Adaptive RL-based personalized learning framework** built from historical EdNet-KT3 student interaction data.
2. **Leakage-safe student state construction** — each state is built only from information available *before* the current question bundle begins, preventing target leakage from bundle outcomes.
3. **Observable, non-artificial action space** — `{QUIZ, PRACTICE, REVISION}`, derived directly from observed EdNet `source` values rather than assigning synthetic outcomes to unsupported activities (e.g., lectures).
4. **Empirical offline transition model** built entirely from training-student data, with **action masking** restricting agents to state-action pairs actually supported by observed data.
5. **Controlled comparison of Q-Learning vs. Monte Carlo Control** under identical states, rewards, transition models, and evaluation protocol.
6. **Comprehensive, multi-seed evaluation** — average return, mastery success rate, behavioral agreement, convergence behavior, computational time, hyperparameter sensitivity, and a state-representation ablation study.

---

## Dataset

This project uses **EdNet-KT3**, part of the [EdNet](https://github.com/riiid/ednet) large-scale educational dataset (131M+ interactions from 784,000+ students, collected from the Santa AI-based self-learning platform). KT3 was selected because — unlike question-only subsets — it also records lecture consumption and explanation-viewing activity, making it well suited to modeling broader student learning behavior.

For computational feasibility, this project uses **100 student interaction files** (98 retained after filtering students with fewer than 5 RL transitions).


---

## Repository Structure

```
.
├── Q-Learning_and_Monte_Carlo_Control.ipynb   # Full pipeline: preprocessing -> RL training -> evaluation
├── Adaptive_Personalized_Learning_Recommendation.pdf   # Full written report
├── README.md
└── outputs/                                   # Generated at runtime (CSVs, figures, processed data)
```

The notebook is organized into the following stages:

| Section | Description |
|---|---|
| Imports & configuration | Library imports, input/output paths, student subset size |
| Metadata loading | Load and validate `questions.csv` and `lectures.csv` |
| Student interaction loading | Load and standardize raw per-student interaction logs |
| Question attempt extraction | Reconstruct bundle-level question attempts and correctness |
| Skill mastery & recent performance | Leakage-safe mastery and rolling recent-accuracy features |
| Lecture sessions | Reconstruct lecture viewing sessions and durations |
| Explanation/revision sessions | Reconstruct explanation-viewing sessions |
| Behavioral feature combination | Merge all student-level behavioral features |
| RL state & transition construction | Discretize state components, build `(s, a, r, s')` transitions |
| Train/val/test split | Student-level split (68/15/15) to prevent information leakage |
| RL agent configuration | `StudentEnv` simulator, Q-Learning, Monte Carlo Control |
| Training & convergence | Train both agents, plot training convergence curves |
| Evaluation | Average return, behavioral agreement, mastery completion proxy, random baseline |
| Multi-seed evaluation | Retrain and evaluate both algorithms across 5 random seeds |
| Hyperparameter analysis | Grid over `α`, `γ`, and episode count |
| Ablation study | Remove each state component individually and re-evaluate |
| Figures | Ablation bar chart and multi-seed comparison charts |

---

## Methodology Summary

### RL Formulation
The problem is modeled as a Markov Decision Process `M = (S, A, P, R, γ)`:

- **State (`S`)**: built from information available *before* the current question bundle — recent accuracy, skill mastery, learning history (weak-skill indicator), and learning/activity progression. Continuous features are discretized into 3 levels (low / medium / high).
- **Action space (`A`)**: `{QUIZ, PRACTICE, REVISION}` — three directly observable, question-based actions with real associated outcomes. (A fourth conceptual action, `LESSON`, is defined but not used, since no observed data source maps to it.)
- **Reward (`R`)**: derived from the observed bundle-level learning outcome following the selected activity.
- **Transition model (`P`)**: an *empirical* transition model built by resampling observed `(state, action) → (reward, next_state, done)` outcomes from the training split, with action masking to exclude unsupported state-action pairs.

### Algorithms Compared
- **Q-Learning** — off-policy, temporal-difference bootstrapping:
  `Q(s,a) ← Q(s,a) + α[r + γ·maxₐ Q(s′,a) − Q(s,a)]`
- **Monte Carlo Control** — first-visit, complete-episode return averaging (no fixed learning rate; step size decays as `1/N(s,a)`).
- **Random policy** — uninformed baseline for comparison.

### Evaluation Metrics
- Test average return
- Mastery success rate (completion-rate proxy)
- Logged-action / empirical-best behavioral agreement
- Training convergence
- Computational (training) time
- Multi-seed mean ± standard deviation (5 seeds)

---

## Setup

### Requirements

```bash
pip install numpy pandas matplotlib scikit-learn tqdm
```

The notebook was developed and run in a **Kaggle notebook environment**, using:
- `INPUT_DIR = /kaggle/input` — location of the raw EdNet-KT3 CSV files
- `OUTPUT_DIR = /kaggle/working/cse753_processed` — location for processed data, result tables, and figures

### Running the Pipeline

1. Download the EdNet-KT3 dataset (`KT3/u*.csv`, `questions.csv`, `lectures.csv`) from the [official EdNet repository](https://github.com/riiid/ednet).
2. Update `INPUT_DIR` / `OUTPUT_DIR` in the configuration cell.
3. (Optional) Adjust `MAX_STUDENTS` (default `100`) to change the number of students processed.
4. Run the notebook top to bottom. Each stage saves its intermediate outputs to `OUTPUT_DIR` — preprocessing outputs, RL transitions, trained Q-tables, evaluation tables (CSV), and figures (PNG).

---

## Key Results

### Single-Seed Headline Comparison (random state = 42)

| Metric | Q-Learning | Monte Carlo Control |
|---|---|---|
| Test Average Return | **30.04** | 28.52 |
| Logged-Action Agreement | **87.88%** | 80.85% |
| Empirical-Best Agreement | **92.67%** | 85.49% |
| Mastery Completion Proxy | **49.4%** | 46.0% |
| Training Time (s) | 0.50 | **0.37** |

Both algorithms outperform the **Random baseline** (42.4% mastery completion proxy).

### ⚠️ Multi-Seed Evaluation (5 seeds, mean ± SD) — the more reliable comparison

| Algorithm | Test Average Return | Mastery Success Rate |
|---|---|---|
| Q-Learning | 31.21 ± 0.74 | **42.1% ± 4.1%** |
| Monte Carlo Control | **31.31 ± 0.78** | 41.4% ± 2.8% |

**The single-seed advantage for Q-Learning does not hold up under multi-seed evaluation.** Averaged over 5 seeds, the two algorithms are **statistically indistinguishable on return**, with Monte Carlo Control's mean return marginally *higher*. Q-Learning retains only a small, noisy edge in mastery success rate. This is a central finding of the study — see Section 3.8 of the report for full discussion.

### Hyperparameter Sensitivity
Neither algorithm's reported "main" configuration (`α = 0.10, γ = 0.95`, 3000 episodes) was its best-performing configuration in the tested grid — e.g., Monte Carlo Control with `γ = 0.95` and 5000 episodes reached 31.90, exceeding every tested Q-Learning configuration but one.

### Ablation Study
Removing 3 of the 4 proposed state components (skill mastery, learning history, learning progression) **increased** Q-Learning's test return relative to the full state — only removing recent performance hurt performance. This suggests the full 4-dimensional state may over-fragment the ~600 observed states relative to the size of the offline dataset (25,736 transitions from 98 students).

---

## Limitations

- **Offline evaluation only** — policies were not deployed to real students; results are based on historical data and an empirical transition model.
- **Limited student sample** — 100 students (98 after filtering) for computational feasibility.
- **Restricted action space** — only directly observable question-based actions (`QUIZ`, `PRACTICE`, `REVISION`); lecture/explanation activities were not artificially assigned outcomes.
- **Simplified, discretized state representation** — the ablation study suggests the 4-component state design may not be optimal for a dataset of this size.
- **Counterfactual limitation** — historical data only reveals outcomes for actions actually taken, a fundamental constraint of offline RL evaluation.
- Results are **seed-sensitive**; single-seed, single-configuration comparisons in this domain should be interpreted cautiously (see Sections 3.6 and 3.8 of the report).

See Section 4 of the report for the full discussion and future directions (deep RL methods, richer student representations, expanded action spaces, specialized offline RL algorithms, and real-world A/B evaluation).

---

## Report

The full written report — including methodology, all tables/figures, and detailed discussion — is included in this repository:

📄 [`Adaptive_Personalized_Learning_Recommendation.pdf`](https://github.com/Shamazda/Adaptive-Personalized-Learning-Recommendation-Using-Reinforcement-Learning/blob/main/Adaptive%20Personalized%20Learning%20Recommendation.pdf)

---

## References

1. Y. Choi et al., "EdNet: A Large-Scale Hierarchical Dataset in Education," *AIED*, 2020.
2. X. Tang, Y. Chen, X. Li, J. Liu, and Z. Ying, "A Reinforcement Learning Approach to Personalized Learning Recommendation Systems," *British Journal of Mathematical and Statistical Psychology*, 72(1):108–135, 2019.
3. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
4. C. Piech et al., "Deep Knowledge Tracing," *NeurIPS*, 2015.
5. C. Tan, R. Han, R. Ye, and K. Chen, "Adaptive Learning Recommendation Strategy Based on Deep Q-Learning," *Applied Psychological Measurement*, 44(4):251–266, 2020.
6. Y. Chen, X. Li, J. Liu, and Z. Ying, "Recommendation System for Adaptive Learning," *Applied Psychological Measurement*, 42(1):24–41, 2018.
7. A. Riedmann, P. Schaper, and B. Lugrin, "Reinforcement Learning in Education: A Systematic Literature Review," *International Journal of Artificial Intelligence in Education*, 2025.

---

## Keywords

Personalized Learning · Reinforcement Learning · Q-Learning · Monte Carlo Control · Adaptive Learning · Educational Recommendation · EdNet-KT3
