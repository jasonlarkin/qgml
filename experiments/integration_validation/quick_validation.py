"""
Quick QGML Integration Experiments - Fast version for rapid testing.

This module implements streamlined experiments that run in under 1 minute:
1. Reduced sample sizes and epochs
2. Fewer hyperparameter configurations
3. Essential comparisons only
4. Quick validation of integration and performance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import warnings
import time
warnings.filterwarnings('ignore')

# Import QGML models
from qgml.learning.specialized.genomics import ChromosomalInstabilityTrainer
from qgml.learning.supervised_trainer import SupervisedMatrixTrainer
from qgml.learning.specialized.regression import QGMLRegressionTrainer


class QuickQGMLExperimentSuite:
"""Fast QGML experimental validation suite - runs in <1 minute."""

# QUICK EXPERIMENTAL CONFIGURATION
QUICK_N_FEATURES = 8 # Reduced feature dimension
QUICK_N_SAMPLES = 100 # Small sample size
QUICK_EPOCHS = 50 # Fast training
QUICK_HILBERT_DIM = 4 # Small Hilbert space

def __init__(self, output_dir: str = "quick_experiments_output"):
"""Initialize quick experiment suite."""
self.output_dir = Path(output_dir)
self.output_dir.mkdir(exist_ok=True)

self.results = {}
self.models = {}
self.start_time = time.time()

print(f" Quick QGML Experiment Suite initialized")
print(f"Output directory: {self.output_dir}")
print(f"Quick config: Features={self.QUICK_N_FEATURES}, Samples={self.QUICK_N_SAMPLES}, Epochs={self.QUICK_EPOCHS}")

def generate_quick_genomic_data(
self,
n_samples: int = None,
complexity_level: str = 'simple',
seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
"""Generate quick synthetic genomic data for fast testing."""
if n_samples is None:
n_samples = self.QUICK_N_SAMPLES

np.random.seed(seed)
torch.manual_seed(seed)

print(f"Generating quick genomic data: {n_samples} samples, {self.QUICK_N_FEATURES} features")

# Simple linear relationships for speed
X_base = np.random.randn(n_samples, self.QUICK_N_FEATURES)

# Create genomic-like features (simplified)
X_genomic = np.abs(X_base) + 0.5 # Positive values like genomic data

# Generate LST values with balanced classes
n_low_lst = int(n_samples * 0.4) # 40% low LST
n_high_lst = n_samples - n_low_lst

# Simple LST generation
lst_low = np.random.uniform(0, 12, n_low_lst)
lst_high = np.random.uniform(12, 40, n_high_lst)

lst_values = np.concatenate([lst_low, lst_high])
indices = np.random.permutation(n_samples)
lst_values = lst_values[indices]
X_genomic = X_genomic[indices]

# Add simple signal
signal = np.mean(X_genomic[:, :3], axis=1) # Use first 3 features
lst_values += 0.3 * signal / np.std(signal)

# Create binary targets
y_binary = (lst_values > 12).astype(float)

# Convert to tensors
X_tensor = torch.tensor(X_genomic, dtype=torch.float32)
y_lst_tensor = torch.tensor(lst_values, dtype=torch.float32)
y_binary_tensor = torch.tensor(y_binary, dtype=torch.float32)

print(f" LST range: [{lst_values.min():.1f}, {lst_values.max():.1f}]")
print(f" High LST ratio: {np.mean(y_binary):.3f}")

return X_tensor, y_lst_tensor, y_binary_tensor

def quick_experiment_1_hyperparameter_test(self):
"""Quick hyperparameter test - 3 configs only."""
print("\n Quick Experiment 1: Hyperparameter Test")
print("=" * 50)

start_time = time.time()

# Generate data
X, y_lst, y_binary = self.generate_quick_genomic_data(seed=42)

# Split data
n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_lst_train, y_lst_test = y_lst[:n_train], y_lst[n_train:]
y_bin_train, y_bin_test = y_binary[:n_train], y_binary[n_train:]

# Quick hyperparameter configs (only 3)
quick_configs = [
{'N': self.QUICK_HILBERT_DIM, 'lr': 0.01, 'comm_penalty': 0.01},
{'N': self.QUICK_HILBERT_DIM, 'lr': 0.005, 'comm_penalty': 0.05},
{'N': 8, 'lr': 0.01, 'comm_penalty': 0.02},
]

results = {}
best_r2 = -np.inf
best_config = None

for i, config in enumerate(quick_configs):
print(f"\nTesting config {i+1}/{len(quick_configs)}: {config}")

# Create and train model
trainer = ChromosomalInstabilityTrainer(
N=config['N'],
D=self.QUICK_N_FEATURES,
lst_threshold=12.0,
use_mixed_loss=True,
learning_rate=config['lr'],
commutation_penalty=config['comm_penalty'],
device='cpu'
)

# Quick training
trainer.fit_chromosomal_instability(
X_train, y_lst_train,
n_epochs=self.QUICK_EPOCHS,
batch_size=16,
validation_split=0.2,
verbose=False
)

# Evaluate
metrics = trainer.evaluate_chromosomal_instability(X_test, y_lst_test, y_bin_test)
results[f"config_{i}"] = {'config': config, 'metrics': metrics}

r2_score = metrics['lst_r2']
print(f" R²: {r2_score:.4f}, MAE: {metrics['lst_mae']:.3f}, Acc: {metrics['accuracy']:.3f}")

if r2_score > best_r2:
best_r2 = r2_score
best_config = config

elapsed = time.time() - start_time
print(f"\n Best config: {best_config}")
print(f"Experiment 1 completed in {elapsed:.1f}s")

self.results['quick_experiment_1'] = {'results': results, 'best_config': best_config}
return results, best_config

def quick_experiment_2_model_comparison(self, best_config: Dict):
"""Quick model comparison - essential models only."""
print("\n Quick Experiment 2: Model Comparison")
print("=" * 50)

start_time = time.time()

# Generate data
X, y_lst, y_binary = self.generate_quick_genomic_data(seed=123)

# Split data
n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_lst_train, y_lst_test = y_lst[:n_train], y_lst[n_train:]
y_bin_train, y_bin_test = y_binary[:n_train], y_binary[n_train:]

# Essential models only
models = {
'chromosomal': ChromosomalInstabilityTrainer(
N=best_config['N'], D=self.QUICK_N_FEATURES, use_mixed_loss=True,
learning_rate=best_config['lr'], commutation_penalty=best_config['comm_penalty']
),
'supervised': SupervisedMatrixTrainer(
N=best_config['N'], D=self.QUICK_N_FEATURES, task_type='regression',
learning_rate=best_config['lr'], commutation_penalty=best_config['comm_penalty']
),
'qgml_original': QGMLRegressionTrainer(
N=best_config['N'], D=self.QUICK_N_FEATURES,
learning_rate=best_config['lr']
)
}

results = {}

for model_name, model in models.items():
print(f"\nTraining {model_name}...")

try:
if model_name == 'chromosomal':
model.fit_chromosomal_instability(
X_train, y_lst_train,
n_epochs=self.QUICK_EPOCHS,
batch_size=16,
validation_split=0.2,
verbose=False
)
metrics = model.evaluate_chromosomal_instability(X_test, y_lst_test, y_bin_test)

elif model_name == 'supervised':
model.fit(
X_train, y_lst_train,
n_epochs=self.QUICK_EPOCHS,
batch_size=16,
X_val=X_test, y_val=y_lst_test,
verbose=False
)
metrics = model.evaluate(X_test, y_lst_test)
predictions = model.predict_batch(X_test)
pred_binary = (predictions > 12).float()
metrics['accuracy'] = (pred_binary == y_bin_test).float().mean().item()

elif model_name == 'qgml_original':
model.fit(
X_train, y_lst_train,
epochs=self.QUICK_EPOCHS,
batch_size=16,
validation_data=(X_test, y_lst_test),
verbose=False
)
predictions = model.predict(X_test)
metrics = {
'lst_mae': mean_absolute_error(y_lst_test.numpy(), predictions.numpy()),
'lst_r2': r2_score(y_lst_test.numpy(), predictions.numpy()),
'accuracy': ((predictions > 12).float() == y_bin_test).float().mean().item()
}

results[model_name] = {'metrics': metrics}

print(f" R²: {metrics.get('lst_r2', metrics.get('r2_score', 0)):.4f}")
print(f" MAE: {metrics.get('lst_mae', metrics.get('mae', 0)):.3f}")
print(f" Accuracy: {metrics.get('accuracy', 0):.3f}")

except Exception as e:
print(f" Failed: {e}")
results[model_name] = {'error': str(e)}

elapsed = time.time() - start_time
print(f"Experiment 2 completed in {elapsed:.1f}s")

self.results['quick_experiment_2'] = results
self.models = models
return results

def quick_experiment_3_classical_comparison(self):
"""Quick classical ML comparison."""
print("\n Quick Experiment 3: Classical Comparison")
print("=" * 50)

start_time = time.time()

# Generate data
X, y_lst, y_binary = self.generate_quick_genomic_data(seed=123)

# Convert to numpy
X_np = X.numpy()
y_lst_np = y_lst.numpy()
y_binary_np = y_binary.numpy()

# Split data
n_train = int(0.8 * len(X_np))
X_train, X_test = X_np[:n_train], X_np[n_train:]
y_lst_train, y_lst_test = y_lst_np[:n_train], y_lst_np[n_train:]
y_bin_train, y_bin_test = y_binary_np[:n_train], y_binary_np[n_train:]

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Quick classical models
classical_models = {
'random_forest': RandomForestRegressor(n_estimators=20, random_state=42), # Reduced trees
'linear_regression': LinearRegression()
}

classical_results = {}

for model_name, model in classical_models.items():
print(f"Training {model_name}...")

try:
model.fit(X_train_scaled, y_lst_train)
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 12).astype(float)

metrics = {
'lst_mae': mean_absolute_error(y_lst_test, y_pred),
'lst_r2': r2_score(y_lst_test, y_pred),
'accuracy': np.mean(y_pred_binary == y_bin_test)
}

classical_results[model_name] = {'metrics': metrics}

print(f" R²: {metrics['lst_r2']:.4f}")
print(f" MAE: {metrics['lst_mae']:.3f}")
print(f" Accuracy: {metrics['accuracy']:.3f}")

except Exception as e:
print(f" Failed: {e}")
classical_results[model_name] = {'error': str(e)}

elapsed = time.time() - start_time
print(f"Experiment 3 completed in {elapsed:.1f}s")

self.results['quick_experiment_3'] = classical_results
return classical_results

def create_quick_comparison_plot(self):
"""Create quick comparison plot."""
if not self.results:
print("No results to plot")
return

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Quick QGML Integration Analysis', fontsize=14, fontweight='bold')

# Plot 1: Model R² comparison
if 'quick_experiment_2' in self.results and 'quick_experiment_3' in self.results:
qgml_results = self.results['quick_experiment_2']
classical_results = self.results['quick_experiment_3']

all_models = {}

# Add QGML models
for model_name, result in qgml_results.items():
if 'metrics' in result:
r2 = result['metrics'].get('lst_r2', result['metrics'].get('r2_score', 0))
all_models[f"QGML_{model_name}"] = r2

# Add classical models
for model_name, result in classical_results.items():
if 'metrics' in result:
all_models[f"Classical_{model_name}"] = result['metrics']['lst_r2']

models = list(all_models.keys())
scores = list(all_models.values())
colors = ['blue' if 'QGML' in m else 'green' for m in models]

bars = ax1.bar(range(len(models)), scores, color=colors, alpha=0.7)
ax1.set_xlabel('Model')
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, scores):
height = bar.get_height()
ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
f'{score:.3f}', ha='center', va='bottom')

# Plot 2: Hyperparameter results
if 'quick_experiment_1' in self.results:
exp1_results = self.results['quick_experiment_1']['results']
configs = list(exp1_results.keys())
r2_scores = [exp1_results[c]['metrics']['lst_r2'] for c in configs]
mae_scores = [exp1_results[c]['metrics']['lst_mae'] for c in configs]

x = np.arange(len(configs))
width = 0.35

ax2.bar(x - width/2, r2_scores, width, label='R² Score', alpha=0.7)
ax2_twin = ax2.twinx()
ax2_twin.bar(x + width/2, mae_scores, width, label='MAE', color='orange', alpha=0.7)

ax2.set_xlabel('Configuration')
ax2.set_ylabel('R² Score')
ax2_twin.set_ylabel('MAE')
ax2.set_title('Hyperparameter Optimization')
ax2.set_xticks(x)
ax2.set_xticklabels([f'C{i+1}' for i in range(len(configs))])
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(self.output_dir / 'quick_qcml_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Quick analysis plot saved to {self.output_dir}")

def generate_quick_report(self):
"""Generate quick summary report."""
total_time = time.time() - self.start_time

report = {
'execution_time': f"{total_time:.1f} seconds",
'experiments_completed': len(self.results),
'quick_config': {
'features': self.QUICK_N_FEATURES,
'samples': self.QUICK_N_SAMPLES,
'epochs': self.QUICK_EPOCHS,
'hilbert_dim': self.QUICK_HILBERT_DIM
},
'results': self.results
}

# Save report
with open(self.output_dir / 'quick_report.json', 'w') as f:
json.dump(report, f, indent=2, default=str)

# Print summary
print("\n Quick QGML Integration Summary")
print("=" * 40)
print(f"⏱️ Total execution time: {total_time:.1f} seconds")
print(f" Experiments completed: {len(self.results)}")

if 'quick_experiment_1' in self.results:
best_config = self.results['quick_experiment_1']['best_config']
print(f" Best hyperparameters: {best_config}")

if 'quick_experiment_2' in self.results:
qgml_results = self.results['quick_experiment_2']
best_qgml_r2 = max([r['metrics'].get('lst_r2', r['metrics'].get('r2_score', -np.inf))
for r in qgml_results.values() if 'metrics' in r])
print(f" Best QGML R²: {best_qgml_r2:.4f}")

if 'quick_experiment_3' in self.results:
classical_results = self.results['quick_experiment_3']
best_classical_r2 = max([r['metrics']['lst_r2']
for r in classical_results.values() if 'metrics' in r])
print(f" Best Classical R²: {best_classical_r2:.4f}")

print(f" Results saved to: {self.output_dir}")

return report


def main():
"""Run quick QGML integration experiments."""
print(" Quick QGML Integration Experiments")
print("=" * 50)

start_time = time.time()

# Initialize suite
suite = QuickQGMLExperimentSuite()

# Run quick experiments
print("\n Running quick experimental protocol...")

# Quick Experiment 1: Hyperparameter test
results_1, best_config = suite.quick_experiment_1_hyperparameter_test()

# Quick Experiment 2: Model comparison
results_2 = suite.quick_experiment_2_model_comparison(best_config)

# Quick Experiment 3: Classical comparison
results_3 = suite.quick_experiment_3_classical_comparison()

# Generate plots and report
suite.create_quick_comparison_plot()
final_report = suite.generate_quick_report()

total_time = time.time() - start_time
print(f"\n All quick experiments completed in {total_time:.1f} seconds!")

return suite, final_report


if __name__ == "__main__":
suite, report = main()
