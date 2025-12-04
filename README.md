# Dynamic Hawkes Process (DHP)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 1.15](https://img.shields.io/badge/tensorflow-1.15-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **Dynamic Hawkes Processes** for modeling time-evolving community states behind diffusion processes.

> **Paper**: [Dynamic Hawkes Processes for Discovering Time-evolving Communities' States behind Diffusion Processes](https://dl.acm.org/doi/10.1145/3447548.3467248)  
> **Authors**: Maya Okawa, Tomoharu Iwata, Yusuke Tanaka, Hiroyuki Toda, Takeshi Kurashima, Hisashi Kashima  
> **Conference**: KDD 2021

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

Dynamic Hawkes Process (DHP) is a novel framework for modeling diffusion processes across multiple communities (e.g., disease spread across countries, information diffusion across social networks). Unlike traditional Hawkes processes that assume static influence patterns, DHP learns **time-evolving community states** that drive the diffusion dynamics.

### Key Innovation

DHP introduces a **latent dynamics function** `f_m(t)` for each community that captures hidden state changes over time. This allows the model to learn how factors like:
- People's awareness of diseases
- Public interest in topics
- Community engagement levels

evolve and influence diffusion patterns.

### Architecture

```
λ_m(t) = μ_m + Σ_j g_{m,m_j}(F_m(t) - F_m(t_j)) × f_m(t)
         ↑       ↑                                    ↑
    Background  Triggering kernel with              Latent
      rate      time-transformed intervals          dynamics
```

Where:
- `F_m(t)` is modeled by a **monotonic mixture neural network**
- `f_m(t) = dF_m(t)/dt` represents the time-evolving state
- Time transformation `Δ̃ = F_m(t) - F_m(t_j)` adjusts diffusion speed

## ✨ Key Features

- **🧠 Automatic Discovery**: Learns hidden community dynamics without domain expertise
- **📈 Tractable Learning**: Closed-form likelihood computation using analytical integrals
- **🎯 Flexible Architecture**: Supports exponential, power-law, and Raleigh kernels
- **⚡ Efficient Training**: Vectorized TensorFlow operations with GPU support
- **📊 Comprehensive Evaluation**: NLL and MAPE metrics with detailed diagnostics
- **🔧 Easy Configuration**: Dataset-specific configs matching paper results

## 🚀 Installation

### Prerequisites

- Python 3.9+
- CUDA 10.0+ (optional, for GPU support)

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/karthikeyamadhavan123/Dynamic-Hawkes.git
cd dynamic-hawkes-process

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/karthikeyamadhavan123/Dynamic-Hawkes.git
cd dynamic-hawkes-process

# Create conda environment
conda env create -f environment.yml
conda activate dhp
```

### Requirements

```txt
tensorflow-gpu==1.15.0  # or tensorflow==1.15.0 for CPU
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
```

## 🎬 Quick Start

### 1. Prepare Your Data

```python
# Your events should be in the format: (time, community_id, source_community_id)
events = [
    (0.1, 0, 0),   # Event at time 0.1 in community 0, triggered by community 0
    (0.2, 1, 0),   # Event at time 0.2 in community 1, triggered by community 0
    (0.3, 0, 1),   # Event at time 0.3 in community 0, triggered by community 1
    # ... more events
]
```

### 2. Train the Model

```python
from config import get_config
from training.train_dhp import DHPTrainer
from data.data_loader import load_dataset

# Load data
data = load_dataset('reddit')  # or 'news', 'protest', 'crime'

# Get optimized configuration
config = get_config('reddit')

# Initialize trainer
trainer = DHPTrainer(config)

# Train model
results, train_losses, val_losses = trainer.train(
    train_events=data['train'],
    val_events=data['val'],
    test_events=data['test'],
    num_communities=data['num_communities']
)

# Print results
print(f"Test NLL: {results['test_nll_per_event']:.4f}")
print(f"Test MAPE: {results['test_mape']:.4f}")

trainer.close()
```

### 3. Visualize Results

```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('NLL')
plt.legend()
plt.title('Training History')

# Plot latent dynamics
plt.subplot(1, 2, 2)
times = results['latent_dynamics'][0]['times']
f_vals = results['latent_dynamics'][0]['f']
plt.plot(times, f_vals)
plt.xlabel('Time')
plt.ylabel('f(t)')
plt.title('Learned Community Dynamics')

plt.tight_layout()
plt.savefig('results.png')
```

## 📁 Project Structure

```
dynamic-hawkes-process/
├── config/
│   └── config.py              # Dataset configurations
├── data/
│   ├── data_loader.py         # Data loading utilities
│   └── preprocessing.py       # Data preprocessing
├── models/
│   ├── dhp_model.py          # DHP model implementation
│   └── monotonic_net.py      # Monotonic neural network
├── training/
│   └── train_dhp.py          # Training logic
├── utils/
│   ├── metrics.py            # Evaluation metrics (NLL, MAPE)
│   ├── helpers.py            # Helper functions
│   └── visualization.py      # Plotting utilities
├── saved_models/             # Trained model checkpoints
├── results/                  # Experiment results
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment
└── README.md                # This file
```

## 📊 Dataset Preparation

### Supported Datasets

DHP has been evaluated on four real-world datasets:

| Dataset | Domain | Time Period | Events | Communities |
|---------|--------|-------------|--------|-------------|
| **Reddit** | Social Media | Mar-Aug 2020 | 23,059 | 25 subreddits |
| **News** | Information Diffusion | Jan-Mar 2020 | 19,541 | 40 news sites |
| **Protest** | Social Events | Mar-Nov 2020 | 22,313 | 35 countries |
| **Crime** | Urban Safety | Mar-Dec 2020 | 29,318 | 13 areas |

### Data Format

Events should be provided as a list of tuples:

```python
events = [
    (time, target_community, source_community),
    ...
]
```

Where:
- `time` (float): Event timestamp, normalized to [0, T]
- `target_community` (int): Community where event occurred (0-indexed)
- `source_community` (int): Community that influenced this event (0-indexed)

### Custom Dataset

To use your own dataset:

```python
from data.data_loader import DataLoader

# Load your raw data
raw_events = load_your_data()  # Your loading function

# Initialize data loader
loader = DataLoader(
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2
)

# Process events
data = loader.prepare_events(
    events=raw_events,
    num_communities=num_communities
)

# data contains: 'train', 'val', 'test', 'num_communities'
```

## ⚙️ Configuration

### Pre-configured Settings

The `config.py` file contains optimized hyperparameters from the paper:

```python
from config import get_config, print_config_summary

# Get configuration for a specific dataset
config = get_config('reddit')

# Print configuration details
print_config_summary(config, 'reddit')
```

### Key Hyperparameters

| Parameter | Reddit | News | Protest | Crime |
|-----------|--------|------|---------|-------|
| **Learning Rate** | 0.002 | 0.002 | 0.002 | 0.002 |
| **Hidden Units** | 8 | 8 | 8 | 8 |
| **Num Mixtures** | 3 | 3 | 3 | 5 |
| **Num Layers** | 2 | 1 | 2 | 3 |
| **Kernel Type** | power_law | power_law | power_law | power_law |
| **Max Epochs** | 100 | 100 | 30 | 30 |

### Custom Configuration

```python
config = {
    # Model Architecture
    'num_mixtures': 3,        # Number of mixture components (1-5)
    'hidden_units': 8,        # Hidden units per layer
    'num_layers': 2,          # Number of neural network layers (1-5)
    
    # Optimization
    'learning_rate': 0.002,   # ADAM learning rate
    'beta1': 0.9,             # ADAM β₁
    'beta2': 0.999,           # ADAM β₂
    
    # Kernel
    'kernel_type': 'power_law',  # 'exponential', 'power_law', 'raleigh'
    
    # Training
    'epochs': 100,            # Maximum epochs
    'patience': 20,           # Early stopping patience
    'batch_size': 128,        # Batch size
    
    # Data Splits
    'validation_split': 0.1,  # 10% validation
    'test_split': 0.2         # 20% test
}
```

## 🎓 Training

### Basic Training

```python
from config import get_config
from training.train_dhp import DHPTrainer

# Load configuration
config = get_config('reddit')

# Initialize trainer
trainer = DHPTrainer(config)

# Train model
results, train_losses, val_losses = trainer.train(
    train_events=train_events,
    val_events=val_events,
    test_events=test_events,
    num_communities=num_communities
)
```

### Training Output

```
Setting up DHP model with 25 communities...
DHP model setup complete!

Epoch  | Train NLL    | Val NLL      | Time     | Patience
---------------------------------------------------------------
     0 |      -5.234 |      -5.189 |   0:02   |        0
    10 |      -5.892 |      -5.847 |   0:02   |        0
    20 |      -6.234 |      -6.198 |   0:02   |        0
    30 |      -6.398 |      -6.367 |   0:02   |        0
    40 |      -6.445 |      -6.421 |   0:02   |        0

Early stopping at epoch 47
Best model from epoch 27 loaded for evaluation

Evaluating model on test set...
Test Results:
  Total NLL: -1486.23
  NLL per event: -6.447
  MAPE: 0.305
```

### Advanced: Grid Search

```python
from config import get_grid_search_config

# Get grid search configuration
grid_config = get_grid_search_config('reddit')

best_config = None
best_nll = float('inf')

# Grid search over hyperparameters
for num_mixtures in grid_config['num_mixtures']:
    for num_layers in grid_config['num_layers']:
        for kernel_type in grid_config['kernel_type']:
            config = {
                'num_mixtures': num_mixtures,
                'num_layers': num_layers,
                'kernel_type': kernel_type,
                # ... other params
            }
            
            trainer = DHPTrainer(config)
            results, _, _ = trainer.train(train, val, test, num_communities)
            trainer.close()
            
            if results['test_nll_per_event'] < best_nll:
                best_nll = results['test_nll_per_event']
                best_config = config

print(f"Best configuration: {best_config}")
print(f"Best NLL: {best_nll:.4f}")
```

## 📈 Evaluation

### Metrics

DHP is evaluated using two metrics:

1. **NLL (Negative Log-Likelihood per event)**: Measures how well the model fits the data
   - Lower is better
   - Reported as negative value (e.g., -6.447)

2. **MAPE (Mean Absolute Percentage Error)**: Measures prediction accuracy
   - Lower is better
   - Range: [0, ∞), where 0 is perfect

### Evaluation Code

```python
# Evaluate on test set
results = trainer.evaluate(
    test_events=test_events,
    history_events=train_events,
    validation_events=val_events,
    T=observation_period
)

print(f"NLL per event: {results['test_nll_per_event']:.4f}")
print(f"MAPE: {results['test_mape']:.4f}")

# Access detailed results
predictions = results['predictions']          # Predicted event counts
latent_dynamics = results['latent_dynamics']  # Learned dynamics
influence_matrix = results['influence_matrix'] # Community interactions
```

### Debugging Predictions

If you get unexpected MAPE values (e.g., MAPE = 1):

```python
from utils.debug import debug_mape_issue, validate_inputs

# Validate inputs
inputs_valid = validate_inputs(
    predictions=predictions,
    test_events=test_events,
    future_intervals=intervals
)

# Debug prediction quality
if inputs_valid:
    debug_mape_issue(
        predictions=predictions,
        test_events=test_events,
        future_intervals=intervals,
        model_name="DHP"
    )
```

## 📊 Results

### Paper Results (Table 2)

| Dataset | DHP NLL | DHP MAPE | Hawkes NLL | Hawkes MAPE |
|---------|---------|----------|------------|-------------|
| **Reddit** | **-6.447** | **0.305** | -5.696 | 0.458 |
| **News** | **-6.301** | **0.442** | -6.167 | 0.471 |
| **Protest** | **-6.914** | **0.318** | -6.260 | 0.415 |
| **Crime** | **-6.983** | **0.117** | -6.799 | 0.179 |

**Bold**: Best performance

### Reproducing Paper Results

```python
from config import get_config

# Use paper-optimized configuration
config = get_config('reddit')

# Train with early stopping
trainer = DHPTrainer(config)
results, _, _ = trainer.train(train, val, test, num_communities)

# Compare with expected results
expected_nll = config['expected_nll']
expected_mape = config['expected_mape']

print(f"Achieved NLL: {results['test_nll_per_event']:.4f}")
print(f"Expected NLL: {expected_nll:.4f}")
print(f"Difference: {abs(results['test_nll_per_event'] - expected_nll):.4f}")

print(f"\nAchieved MAPE: {results['test_mape']:.4f}")
print(f"Expected MAPE: {expected_mape:.4f}")
```

### Visualization

```python
from utils.visualization import plot_results

# Plot training curves
plot_results(
    train_losses=train_losses,
    val_losses=val_losses,
    latent_dynamics=results['latent_dynamics'],
    influence_matrix=results['influence_matrix'],
    save_path='results/dhp_results.png'
)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. MAPE = 1 or Very High

**Symptoms**: MAPE is 1.0 or close to 1.0

**Causes**:
- Test events not in evaluation intervals
- Wrong time scale/normalization
- Model not trained properly

**Solution**:
```python
from utils.debug import debug_mape_issue, fix_interval_creation

# Fix intervals to match test events
intervals = fix_interval_creation(test_events, num_intervals=5)

# Debug the issue
debug_mape_issue(predictions, test_events, intervals, "DHP")
```

#### 2. NLL Not Matching Paper

**Symptoms**: NLL is much worse than paper results

**Causes**:
- Wrong hyperparameters
- Data preprocessing issues
- Not enough training epochs

**Solution**:
```python
# Use exact paper configuration
config = get_config('reddit')  # Use dataset-specific config

# Ensure data is preprocessed correctly
# Events should be sorted by time
events = sorted(events, key=lambda x: x[0])

# Train longer if needed
config['epochs'] = 200
config['patience'] = 30
```

#### 3. Training is Slow

**Symptoms**: Training takes very long per epoch

**Causes**:
- Large number of events
- CPU-only execution
- Inefficient batch processing

**Solution**:
```python
# Use GPU
# Install: pip install tensorflow-gpu==1.15.0

# Reduce batch operations
config['batch_size'] = 64  # Default is 128

# Use fewer mixture components
config['num_mixtures'] = 1  # Start simple
```

#### 4. Out of Memory

**Symptoms**: CUDA OOM error or system memory error

**Solution**:
```python
# Reduce model size
config['hidden_units'] = 4  # Default is 8
config['num_mixtures'] = 2  # Default is 3

# Enable memory growth
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(gpu_options=gpu_options)
```

#### 5. NaN Loss

**Symptoms**: Loss becomes NaN during training

**Causes**:
- Learning rate too high
- Numerical instability in kernel

**Solution**:
```python
# Reduce learning rate
config['learning_rate'] = 0.0005  # Default is 0.002

# Use more stable kernel
config['kernel_type'] = 'exponential'  # Instead of power_law

# Add gradient clipping
# (Modify train_dhp.py to add clip_by_norm)
```

### Getting Help

If you encounter issues:

1. **Check the debug output**: Use the debug functions provided
2. **Verify data format**: Ensure events are `(time, community_id, source_id)`
3. **Compare configurations**: Use `print_config_summary(config, dataset)`
4. **Open an issue**: Provide debug output and configuration

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{okawa2021dynamic,
  title={Dynamic Hawkes Processes for Discovering Time-evolving Communities' States behind Diffusion Processes},
  author={Okawa, Maya and Iwata, Tomoharu and Tanaka, Yusuke and Toda, Hiroyuki and Kurashima, Takeshi and Kashima, Hisashi},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1292--1302},
  year={2021}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper authors for the DHP framework
- TensorFlow team for the deep learning framework
- KDD 2021 reviewers for valuable feedback

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: karthikeyadonnipad2005@gmail.com

## 🔗 Links

- [Paper (ACM DL)](https://dl.acm.org/doi/10.1145/3447548.3467248)
- [Paper (arXiv)](https://arxiv.org/abs/2105.xxxxx)
- [Project Page](https://your-project-page.com)
- [Documentation](https://your-docs-page.com)

---

**Last Updated**: December 2025  
**Version**: 1.0.0
