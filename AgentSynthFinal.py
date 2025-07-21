import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import gymnasium as gym
from gymnasium import spaces
from sklearn.metrics import mean_squared_error
import warnings
import os
warnings.filterwarnings('ignore')

class SyntheticDataEnvironment(gym.Env):
    def __init__(self, original_data: pd.DataFrame, context=None):
        super().__init__()
        self.original_data = original_data
        self.feature_stats = self._calculate_feature_stats()
        self.action_space = spaces.Box(low=0.5, high=2.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.current_step = 0
        self.max_steps = 100
        self.context = context

    def _calculate_feature_stats(self) -> Dict:
        stats = {
            'price_mean': self.original_data['price'].mean(),
            'price_std': self.original_data['price'].std(),
            'quantity_mean': self.original_data['quantity'].mean(),
            'quantity_std': self.original_data['quantity'].std(),
            'category_dist': self.original_data['category'].value_counts(normalize=True).to_dict(),
            'user_type_dist': self.original_data['user_type'].value_counts(normalize=True).to_dict(),
            'total_revenue_mean': (self.original_data['price'] * self.original_data['quantity']).mean(),
        }
        return stats

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_observation(), {}

    def _get_observation(self):
        obs = np.array([
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            self.current_step / self.max_steps,
            0.5
        ], dtype=np.float32)
        return obs

    def step(self, action):
        price_factor, quantity_factor, category_factor, seasonal_factor = action
        synthetic_data = self._generate_data(price_factor, quantity_factor, category_factor, seasonal_factor)
        reward = self._calculate_reward(synthetic_data)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        return self._get_observation(), reward, terminated, truncated, {'synthetic_data': synthetic_data}

    def _generate_data(self, price_factor, quantity_factor, category_factor, seasonal_factor):
        n_samples = len(self.original_data)
        synthetic_data = []
        categories = list(self.feature_stats['category_dist'].keys())
        category_probs = list(self.feature_stats['category_dist'].values())
        user_types = list(self.feature_stats['user_type_dist'].keys())
        user_type_probs = list(self.feature_stats['user_type_dist'].values())
        adjusted_probs = np.array(category_probs) * category_factor
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        for i in range(n_samples):
            category = np.random.choice(categories, p=adjusted_probs)
            user_type = np.random.choice(user_types, p=user_type_probs)
            base_price = np.random.normal(self.feature_stats['price_mean'], self.feature_stats['price_std'])
            price = max(1.0, base_price * price_factor * (0.8 + 0.4 * random.random()))
            base_quantity = np.random.normal(self.feature_stats['quantity_mean'], self.feature_stats['quantity_std'])
            quantity = max(1, int(base_quantity * quantity_factor))
            day_of_year = random.randint(1, 365)
            seasonal_multiplier = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365) * seasonal_factor
            price *= seasonal_multiplier
            base_date = datetime(2024, 1, 1)
            timestamp = base_date + timedelta(days=random.randint(0, 365))
            if self.context == "evening_rush":
                hour = np.random.choice(range(17, 21))
            else:
                hour = np.random.choice(range(0, 24))
            timestamp = timestamp.replace(hour=hour)
            user_id = random.randint(1000, 9999)
            synthetic_data.append({
                'user_id': user_id,
                'user_type': user_type,
                'category': category,
                'price': round(price, 2),
                'quantity': quantity,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'total_revenue': round(price * quantity, 2)
            })
        return pd.DataFrame(synthetic_data)

    def _calculate_reward(self, synthetic_data):
        reward = 0.0
        try:
            price_mse = mean_squared_error([self.feature_stats['price_mean']], [synthetic_data['price'].mean()])
            price_reward = 1.0 / (1.0 + price_mse)
            reward += price_reward * 0.2
            quantity_mse = mean_squared_error([self.feature_stats['quantity_mean']], [synthetic_data['quantity'].mean()])
            quantity_reward = 1.0 / (1.0 + quantity_mse)
            reward += quantity_reward * 0.15
            synthetic_cat_dist = synthetic_data['category'].value_counts(normalize=True)
            category_similarity = 0.0
            for cat in self.feature_stats['category_dist']:
                if cat in synthetic_cat_dist:
                    category_similarity += min(self.feature_stats['category_dist'][cat], synthetic_cat_dist[cat])
            reward += category_similarity * 0.2
            synthetic_user_type_dist = synthetic_data['user_type'].value_counts(normalize=True)
            user_type_similarity = 0.0
            for ut in self.feature_stats['user_type_dist']:
                if ut in synthetic_user_type_dist:
                    user_type_similarity += min(self.feature_stats['user_type_dist'][ut], synthetic_user_type_dist[ut])
            reward += user_type_similarity * 0.2
            original_revenue_mean = self.feature_stats['total_revenue_mean']
            synthetic_revenue_mean = synthetic_data['total_revenue'].mean()
            revenue_reward = 1.0 / (1.0 + abs(original_revenue_mean - synthetic_revenue_mean) / original_revenue_mean)
            reward += revenue_reward * 0.25
        except Exception:
            reward = -1.0
        return reward

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, epsilon=0.1, decay_rate=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.q_table = {}
        self.best_action = np.array([1.0, 1.0, 1.0, 1.0])
        self.best_reward = -float('inf')

    def discretize_state(self, state):
        return tuple(np.round(state * 10).astype(int))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = np.random.uniform(0.5, 2.0, self.action_dim)
        else:
            action = self.best_action + np.random.normal(0, 0.1, self.action_dim)
            action = np.clip(action, 0.5, 2.0)
        return action.astype(np.float32)

    def update(self, state, action, reward, next_state):
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_action = action.copy()
        self.epsilon *= self.decay_rate

class AgentSynth:
    def __init__(self):
        self.env = None
        self.agent = None
        self.training_history = []

    def load_initial_data(self, file_path=None):
        if file_path and file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            data = self._create_sample_data()
        return data

    def _create_sample_data(self):
        np.random.seed(42)
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
        user_types = ['regular', 'premium', 'guest']
        data = []
        for i in range(200):
            category = np.random.choice(categories, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            user_type = np.random.choice(user_types, p=[0.6, 0.3, 0.1])
            if category == 'Electronics':
                price = np.random.normal(150, 50)
                quantity = np.random.poisson(1) + 1
            elif category == 'Clothing':
                price = np.random.normal(40, 15)
                quantity = np.random.poisson(2) + 1
            elif category == 'Books':
                price = np.random.normal(20, 5)
                quantity = np.random.poisson(1) + 1
            elif category == 'Home':
                price = np.random.normal(80, 30)
                quantity = np.random.poisson(1) + 1
            else:
                price = np.random.normal(60, 25)
                quantity = np.random.poisson(1) + 1
            price = max(5, price)
            base_date = datetime(2024, 1, 1)
            timestamp = base_date + timedelta(days=np.random.randint(0, 90))
            hour = np.random.choice(range(0, 24))
            timestamp = timestamp.replace(hour=hour)
            data.append({
                'user_id': 1000 + i,
                'user_type': user_type,
                'category': category,
                'price': round(price, 2),
                'quantity': quantity,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'total_revenue': round(price * quantity, 2)
            })
        return pd.DataFrame(data)

    def initialize_agent(self, initial_data, context=None):
        self.env = SyntheticDataEnvironment(initial_data, context=context)
        self.agent = QLearningAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0]
        )

    def train(self, episodes=50):
        print(f"Training AgentSynth for {episodes} episodes...")
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            for step in range(self.env.max_steps):
                action = self.agent.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if terminated or truncated:
                    break
            self.training_history.append(total_reward)
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Epsilon: {self.agent.epsilon:.4f}")

    def generate_synthetic_data(self, n_samples=1000):
        print(f"Generating {n_samples} synthetic samples...")
        action = self.agent.best_action
        state, _ = self.env.reset()
        _, _, _, _, info = self.env.step(action)
        base_synthetic = info['synthetic_data']
        all_synthetic_data = []
        for i in range(n_samples // len(base_synthetic) + 1):
            varied_action = action + np.random.normal(0, 0.05, len(action))
            varied_action = np.clip(varied_action, 0.5, 2.0)
            _, _, _, _, info = self.env.step(varied_action)
            varied_data = info['synthetic_data']
            all_synthetic_data.append(varied_data)
        combined_data = pd.concat(all_synthetic_data, ignore_index=True)
        final_data = combined_data.sample(n=min(n_samples, len(combined_data)))
        return final_data.reset_index(drop=True)

    def validate_synthetic_data(self, original_data, synthetic_data):
        print("\n=== Data Quality Validation ===")
        comparison = {
            'Metric': [],
            'Original': [],
            'Synthetic': [],
            'Difference': []
        }
        comparison['Metric'].extend(['Price Mean', 'Price Std'])
        comparison['Original'].extend([original_data['price'].mean(), original_data['price'].std()])
        comparison['Synthetic'].extend([synthetic_data['price'].mean(), synthetic_data['price'].std()])
        comparison['Difference'].extend([
            abs(original_data['price'].mean() - synthetic_data['price'].mean()),
            abs(original_data['price'].std() - synthetic_data['price'].std())
        ])
        comparison['Metric'].extend(['Quantity Mean', 'Quantity Std'])
        comparison['Original'].extend([original_data['quantity'].mean(), original_data['quantity'].std()])
        comparison['Synthetic'].extend([synthetic_data['quantity'].mean(), synthetic_data['quantity'].std()])
        comparison['Difference'].extend([
            abs(original_data['quantity'].mean() - synthetic_data['quantity'].mean()),
            abs(original_data['quantity'].std() - synthetic_data['quantity'].std())
        ])
        comparison['Metric'].extend(['Revenue Mean', 'Revenue Std'])
        comparison['Original'].extend([original_data['total_revenue'].mean(), original_data['total_revenue'].std()])
        comparison['Synthetic'].extend([synthetic_data['total_revenue'].mean(), synthetic_data['total_revenue'].std()])
        comparison['Difference'].extend([
            abs(original_data['total_revenue'].mean() - synthetic_data['total_revenue'].mean()),
            abs(original_data['total_revenue'].std() - synthetic_data['total_revenue'].std())
        ])
        comparison['Metric'].extend(['User Type Distribution'])
        orig_ut = original_data['user_type'].value_counts(normalize=True).to_dict()
        synth_ut = synthetic_data['user_type'].value_counts(normalize=True).to_dict()
        ut_diff = sum(abs(orig_ut.get(ut, 0) - synth_ut.get(ut, 0)) for ut in set(orig_ut) | set(synth_ut))
        comparison['Original'].append(orig_ut)
        comparison['Synthetic'].append(synth_ut)
        comparison['Difference'].append(ut_diff)
        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.round(2))
        return comparison_df

    def visualize_data_comparison(self, original_data, synthetic_data):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Original vs Synthetic Data Comparison', fontsize=16)
        axes[0, 0].hist(original_data['price'], alpha=0.7, label='Original', bins=20)
        axes[0, 0].hist(synthetic_data['price'], alpha=0.7, label='Synthetic', bins=20)
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 1].hist(original_data['quantity'], alpha=0.7, label='Original', bins=20)
        axes[0, 1].hist(synthetic_data['quantity'], alpha=0.7, label='Synthetic', bins=20)
        axes[0, 1].set_title('Quantity Distribution')
        axes[0, 1].set_xlabel('Quantity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        orig_cat = original_data['category'].value_counts()
        synth_cat = synthetic_data['category'].value_counts()
        x_pos = np.arange(len(orig_cat))
        width = 0.35
        axes[0, 2].bar(x_pos - width/2, orig_cat.values, width, alpha=0.7, label='Original')
        axes[0, 2].bar(x_pos + width/2, synth_cat.values, width, alpha=0.7, label='Synthetic')
        axes[0, 2].set_title('Category Distribution')
        axes[0, 2].set_xlabel('Category')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(orig_cat.index, rotation=45)
        axes[0, 2].legend()
        axes[1, 0].hist(original_data['total_revenue'], alpha=0.7, label='Original', bins=20)
        axes[1, 0].hist(synthetic_data['total_revenue'], alpha=0.7, label='Synthetic', bins=20)
        axes[1, 0].set_title('Revenue Distribution')
        axes[1, 0].set_xlabel('Total Revenue')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        orig_ut = original_data['user_type'].value_counts()
        synth_ut = synthetic_data['user_type'].value_counts()
        x_pos_ut = np.arange(len(orig_ut))
        axes[1, 1].bar(x_pos_ut - width/2, orig_ut.values, width, alpha=0.7, label='Original')
        axes[1, 1].bar(x_pos_ut + width/2, synth_ut.values, width, alpha=0.7, label='Synthetic')
        axes[1, 1].set_title('User Type Distribution')
        axes[1, 1].set_xlabel('User Type')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(x_pos_ut)
        axes[1, 1].set_xticklabels(orig_ut.index, rotation=45)
        axes[1, 1].legend()
        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.savefig('data_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        # Always create a new training progress plot with a unique filename
        progress_filename = f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history)
        plt.title('Training Progress - Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.savefig(progress_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training progress plot saved as: {progress_filename}")

    def export_data(self, data, filename):
        csv_filename = f"{filename}.csv"
        data.to_csv(csv_filename, index=False)
        print(f"Data exported to {csv_filename}")
        return csv_filename, None

def main():
    print("=== AgentSynth: RL-based Synthetic Data Generator ===\n")
    agent_synth = AgentSynth()
    print("1. Loading initial dataset...")
    initial_data = agent_synth.load_initial_data()
    print(f"Initial dataset shape: {initial_data.shape}")
    print(f"Initial data preview:\n{initial_data.head()}\n")
    print("2. Initializing RL agent and environment...")
    agent_synth.initialize_agent(initial_data, context="evening_rush")
    print("3. Training the agent...")
    agent_synth.train(episodes=30)
    print("\n4. Generating synthetic data...")
    synthetic_data = agent_synth.generate_synthetic_data(n_samples=500)
    print(f"Generated synthetic dataset shape: {synthetic_data.shape}")
    print(f"Synthetic data preview:\n{synthetic_data.head()}\n")
    print("5. Validating synthetic data quality...")
    comparison_results = agent_synth.validate_synthetic_data(initial_data, synthetic_data)
    print("\n6. Creating visualizations...")
    agent_synth.visualize_data_comparison(initial_data, synthetic_data)
    print("\n7. Exporting synthetic data...")
    csv_file, _ = agent_synth.export_data(synthetic_data, "synthetic_ecommerce_data")
    feedback = "good"
    if feedback == "good":
        agent_synth.agent.epsilon *= 0.9
    else:
        agent_synth.agent.epsilon = min(1.0, agent_synth.agent.epsilon * 1.1)
    print(f"\n=== AgentSynth Execution Complete ===")
    print(f"Generated {len(synthetic_data)} synthetic samples")
    print(f"Data exported to: {csv_file}")
    print(f"Visualizations saved: data_comparison.png and a new training progress PNG each run")

if __name__ == "__main__":
    main()