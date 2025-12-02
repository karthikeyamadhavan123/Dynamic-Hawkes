import json
import os
from typing import Dict, Any


def get_config(dataset: str = 'default') -> Dict[str, Any]:
    """
    Get configuration for DHP model based on dataset
    
    Configurations are based on the paper:
    "Dynamic Hawkes Processes for Discovering Time-evolving Communities' States 
    behind Diffusion Processes" (KDD 2021)
    
    Key hyperparameters from Section 5.3:
    - Learning rate: 0.002 (ADAM optimizer with β₁=0.9, β₂=0.999)
    - Hidden units: 8 per layer
    - Number of mixtures: {1, 2, 3, 4, 5} - tuned via grid search
    - Number of layers: {1, 2, 3, 4, 5} - tuned via grid search
    - Kernel function: exponential, power-law, or Raleigh
    - Batch size: 128
    - Early stopping: max 100 epochs for Reddit/News, 30 for Protest/Crime
    """
    configs = {
        'default': {
            # Model architecture
            'num_mixtures': 3,
            'hidden_units': 8,  # Paper uses 8 (Section 5.3)
            'num_layers': 2,
            
            # Optimization
            'learning_rate': 0.002,  # Paper uses 0.002 (Section 5.3)
            'beta1': 0.9,  # ADAM β₁
            'beta2': 0.999,  # ADAM β₂
            
            # Kernel
            'kernel_type': 'power_law',  # Best performing in paper
            
            # Training
            'epochs': 100,
            'patience': 20,
            'batch_size': 128,  # Paper uses 128 (Section 5.3)
            
            # Data splits
            'validation_split': 0.1,  # 70% train, 10% val, 20% test
            'test_split': 0.2
        },
        
        'reddit': {
            # Model architecture (Section 5.3 and Appendix D.2)
            'num_mixtures': 3,  # Best for Reddit
            'hidden_units': 8,
            'num_layers': 2,  # Best for Reddit (Appendix D.2)
            
            # Optimization
            'learning_rate': 0.002,
            'beta1': 0.9,
            'beta2': 0.999,
            
            # Kernel
            'kernel_type': 'power_law',  # Best performing (Figure 3b)
            
            # Training
            'epochs': 100,  # Max 100 for Reddit (Section 5.3)
            'patience': 20,
            'batch_size': 128,
            
            # Data splits
            'validation_split': 0.1,
            'test_split': 0.2,
            
            # Expected results (Table 2)
            'expected_nll': -6.447,
            'expected_mape': 0.305
        },
        
        'news': {
            # Model architecture (Appendix D.2)
            'num_mixtures': 3,  # Best for News
            'hidden_units': 8,
            'num_layers': 1,  # Best for News (Appendix D.2)
            
            # Optimization
            'learning_rate': 0.002,
            'beta1': 0.9,
            'beta2': 0.999,
            
            # Kernel
            'kernel_type': 'power_law',  # Best performing (Figure 3b)
            
            # Training
            'epochs': 100,  # Max 100 for News (Section 5.3)
            'patience': 20,
            'batch_size': 128,
            
            # Data splits
            'validation_split': 0.1,
            'test_split': 0.2,
            
            # Expected results (Table 2)
            'expected_nll': -6.301,
            'expected_mape': 0.442
        },
        
        'protest': {
            # Model architecture (Appendix D.2)
            'num_mixtures': 3,  # Best for Protest
            'hidden_units': 8,
            'num_layers': 2,  # Best for Protest (Appendix D.2)
            
            # Optimization
            'learning_rate': 0.002,
            'beta1': 0.9,
            'beta2': 0.999,
            
            # Kernel
            'kernel_type': 'power_law',  # Best performing (Figure 3b)
            
            # Training
            'epochs': 30,  # Max 30 for Protest (Section 5.3)
            'patience': 10,  # Earlier stopping for Protest
            'batch_size': 128,  # Keeping consistent with other datasets
            
            # Data splits
            'validation_split': 0.1,
            'test_split': 0.2,
            
            # Expected results (Table 2)
            'expected_nll': -6.914,
            'expected_mape': 0.318
        },
        
        'crime': {
            # Model architecture (Appendix D.2)
            'num_mixtures': 5,  # Best for Crime (Appendix D.2)
            'hidden_units': 8,
            'num_layers': 3,  # Best for Crime (Appendix D.2)
            
            # Optimization
            'learning_rate': 0.002,
            'beta1': 0.9,
            'beta2': 0.999,
            
            # Kernel
            'kernel_type': 'power_law',  # Best performing (Figure 3b)
            
            # Training
            'epochs': 30,  # Max 30 for Crime (Section 5.3)
            'patience': 10,  # Earlier stopping for Crime
            'batch_size': 128,
            
            # Data splits
            'validation_split': 0.1,
            'test_split': 0.2,
            
            # Expected results (Table 2)
            'expected_nll': -6.983,
            'expected_mape': 0.117
        },
        
        'dblp': {
            # Generic configuration for DBLP (not in paper)
            'num_mixtures': 3,
            'hidden_units': 8,
            'num_layers': 2,
            
            # Optimization
            'learning_rate': 0.002,
            'beta1': 0.9,
            'beta2': 0.999,
            
            # Kernel
            'kernel_type': 'power_law',
            
            # Training
            'epochs': 30,
            'patience': 20,
            'batch_size': 128,
            
            # Data splits
            'validation_split': 0.1,
            'test_split': 0.2
        }
    }

    return configs.get(dataset, configs['default'])


def get_grid_search_config(dataset: str = 'default') -> Dict[str, Any]:
    """
    Get grid search configuration for hyperparameter tuning
    
    Based on Section 5.3: "The hyperparameters of each model are optimized via grid search"
    """
    grid_configs = {
        'default': {
            'num_mixtures': [1, 2, 3, 4, 5],
            'num_layers': [1, 2, 3, 4, 5],
            'kernel_type': ['exponential', 'power_law', 'raleigh'],
            'hidden_units': [8],  # Fixed in paper
            'learning_rate': [0.002],  # Fixed in paper
        },
        'reddit': {
            'num_mixtures': [1, 2, 3, 4, 5],
            'num_layers': [1, 2, 3, 4, 5],
            'kernel_type': ['exponential', 'power_law', 'raleigh'],
            'hidden_units': [8],
            'learning_rate': [0.002],
        },
        'news': {
            'num_mixtures': [1, 2, 3, 4, 5],
            'num_layers': [1, 2, 3, 4, 5],
            'kernel_type': ['exponential', 'power_law', 'raleigh'],
            'hidden_units': [8],
            'learning_rate': [0.002],
        },
        'protest': {
            'num_mixtures': [1, 2, 3, 4, 5],
            'num_layers': [1, 2, 3, 4, 5],
            'kernel_type': ['exponential', 'power_law', 'raleigh'],
            'hidden_units': [8],
            'learning_rate': [0.002],
        },
        'crime': {
            'num_mixtures': [1, 2, 3, 4, 5],
            'num_layers': [1, 2, 3, 4, 5],
            'kernel_type': ['exponential', 'power_law', 'raleigh'],
            'hidden_units': [8],
            'learning_rate': [0.002],
        }
    }
    
    return grid_configs.get(dataset, grid_configs['default'])


def get_kernel_params(kernel_type: str) -> Dict[str, Any]:
    """
    Get kernel-specific parameters
    
    Based on Table 4 (Appendix B) and common practices
    """
    kernel_params = {
        'exponential': {
            'name': 'exp',
            'formula': 'α * exp(-β * Δt)',
            'has_alpha': True,
            'has_beta': True,
            'extra_params': {}
        },
        'power_law': {
            'name': 'power_law',
            'formula': 'αβ / (α + βt)^(p+1)',
            'has_alpha': True,
            'has_beta': True,
            'extra_params': {
                'p': 2  # Fixed in paper (Appendix B)
            }
        },
        'raleigh': {
            'name': 'raleigh',
            'formula': 'αt * exp(-βt²)',
            'has_alpha': True,
            'has_beta': True,
            'extra_params': {}
        }
    }
    
    return kernel_params.get(kernel_type, kernel_params['exponential'])


def print_config_summary(config: Dict[str, Any], dataset: str):
    """Print a summary of the configuration"""
    print(f"\n{'='*60}")
    print(f"DHP Configuration for {dataset.upper()} Dataset")
    print(f"{'='*60}")
    print(f"\nModel Architecture:")
    print(f"  Number of mixtures: {config['num_mixtures']}")
    print(f"  Hidden units per layer: {config['hidden_units']}")
    print(f"  Number of layers: {config['num_layers']}")
    print(f"  Kernel type: {config['kernel_type']}")
    
    print(f"\nOptimization:")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  ADAM β₁: {config.get('beta1', 0.9)}")
    print(f"  ADAM β₂: {config.get('beta2', 0.999)}")
    print(f"  Batch size: {config['batch_size']}")
    
    print(f"\nTraining:")
    print(f"  Max epochs: {config['epochs']}")
    print(f"  Early stopping patience: {config['patience']}")
    
    print(f"\nData Splits:")
    print(f"  Validation: {config['validation_split']*100:.0f}%")
    print(f"  Test: {config['test_split']*100:.0f}%")
    print(f"  Train: {(1-config['validation_split']-config['test_split'])*100:.0f}%")
    
    if 'expected_nll' in config:
        print(f"\nExpected Results (from paper):")
        print(f"  NLL: {config['expected_nll']:.4f}")
        print(f"  MAPE: {config['expected_mape']:.4f}")
    
    print(f"{'='*60}\n")


def save_config(config: Dict[str, Any], file_path: str):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {file_path}")


def load_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from: {file_path}")
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
    required_keys = [
        'num_mixtures', 'hidden_units', 'num_layers', 'learning_rate',
        'kernel_type', 'epochs', 'patience', 'batch_size'
    ]
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required configuration key: {key}")
            return False
    
    # Validate ranges
    if config['num_mixtures'] < 1:
        print("Error: num_mixtures must be >= 1")
        return False
    
    if config['hidden_units'] < 1:
        print("Error: hidden_units must be >= 1")
        return False
    
    if config['num_layers'] < 1:
        print("Error: num_layers must be >= 1")
        return False
    
    if config['learning_rate'] <= 0:
        print("Error: learning_rate must be > 0")
        return False
    
    if config['kernel_type'] not in ['exponential', 'power_law', 'raleigh', 'exp', 'pwl', 'ray']:
        print(f"Warning: Unknown kernel_type: {config['kernel_type']}")
    
    if config['epochs'] < 1:
        print("Error: epochs must be >= 1")
        return False
    
    if config['patience'] < 1:
        print("Error: patience must be >= 1")
        return False
    
    if config['batch_size'] < 1:
        print("Error: batch_size must be >= 1")
        return False
    
    return True


# Example usage
if __name__ == "__main__":
    # Get configuration for Reddit dataset
    config = get_config('reddit')
    print_config_summary(config, 'reddit')
    
    # Validate configuration
    if validate_config(config):
        print("✓ Configuration is valid")
    
    # Save configuration
    save_config(config, 'configs/reddit_config.json')
    
    # Show grid search options
    print("\nGrid Search Configuration:")
    grid_config = get_grid_search_config('reddit')
    for param, values in grid_config.items():
        print(f"  {param}: {values}")