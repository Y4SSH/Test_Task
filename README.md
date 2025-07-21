# AgentSynth: Reinforcement Learning-Based Synthetic Data Generator

**AgentSynth** is an intelligent data simulation system that utilizes reinforcement learning (RL) to generate realistic synthetic datasets. It is designed for scenarios where real data is limited, sensitive, or unavailable. The system uses a custom Gym environment and Q-learning agent to learn from small seed data and produce structured outputs that preserve statistical and contextual properties.

---

## Features

* Reinforcement learning using Q-learning
* Custom Gymnasium environment for synthetic data generation
* Context-aware generation (e.g., time-based effects like evening rush)
* Export of high-quality synthetic datasets in CSV format
* Visual validation with statistical comparisons and plots

---

## Project Structure

```
.
├── agent_synth.py                  # Main Python script
├── synthetic_ecommerce_data.csv    # Output dataset
├── data_comparison.png             # Visual comparison of distributions
├── training_progress_<timestamp>.png # Reward curve during training
├── README.md                       # This file
```

---

## Installation

To run this project, ensure the following packages are installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn gymnasium
```

---

## How to Use

Run the main script directly:

```bash
python agent_synth.py
```

This will:

1. Generate or load a small e-commerce dataset
2. Initialize and train the RL agent over 30 episodes
3. Generate 500 synthetic samples using the best-learned policy
4. Validate and visualize the generated data
5. Export the final dataset as a CSV

---

## Output Artifacts

* **synthetic\_ecommerce\_data.csv**: Generated dataset of synthetic transactions
* **data\_comparison.png**: Overlay plots comparing original and synthetic data distributions
* **training\_progress\_YYYYMMDD\_HHMMSS.png**: Reward progression during agent training

---

## Customization

* The `context` parameter can be changed to simulate different scenarios (e.g., `"evening_rush"`)
* The `feedback` flag allows simulating adaptive learning by modifying exploration rate based on qualitative feedback

---

## Learnings

This project demonstrates the practical use of reinforcement learning in:

* Designing and training agents beyond traditional control problems
* Reward shaping for structured data generation
* Creating explainable, context-aware synthetic datasets

---

## Author

Yash Lokare

---

## License

This project is intended for academic and educational use only.
