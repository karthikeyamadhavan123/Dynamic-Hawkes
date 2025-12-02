import numpy as np
import json
import os
from typing import List, Dict, Any
from utils.metrics import calculate_mape, calculate_nll, calculate_accuracy

class Evaluator:
    """
    Comprehensive evaluation for DHP and baseline models
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model_name: str, predictions: np.ndarray, 
                      true_events: List, intervals: List[Tuple[float, float]]) -> Dict[str, float]:
        """Evaluate a single model"""
        mape = calculate_mape(predictions, true_events, intervals)
        nll = calculate_nll(predictions, true_events, intervals)
        accuracy = calculate_accuracy(predictions, true_events, intervals)
        
        results = {
            'mape': mape,
            'nll': nll,
            'accuracy': accuracy
        }
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare multiple models"""
        comparison = {}
        
        for model_name, results in model_results.items():
            comparison[model_name] = {
                'mape': results.get('mape', float('inf')),
                'nll': results.get('nll', float('inf')),
                'accuracy': results.get('accuracy', 0.0)
            }
        
        # Find best model for each metric
        best_mape = min(comparison.items(), key=lambda x: x[1]['mape'])
        best_nll = min(comparison.items(), key=lambda x: x[1]['nll'])
        best_accuracy = max(comparison.items(), key=lambda x: x[1]['accuracy'])
        
        comparison['best_models'] = {
            'mape': best_mape[0],
            'nll': best_nll[0],
            'accuracy': best_accuracy[0]
        }
        
        return comparison
    
    def save_results(self, file_path: str):
        """Save evaluation results to file"""
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, file_path: str):
        """Load evaluation results from file"""
        with open(file_path, 'r') as f:
            self.results = json.load(f)
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            print(f"  MAPE:    {results.get('mape', 'N/A'):.4f}")
            print(f"  NLL:     {results.get('nll', 'N/A'):.4f}")
            print(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}")