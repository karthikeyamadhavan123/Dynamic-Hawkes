import numpy as np
from typing import List, Tuple
import math

def validate_inputs(predictions: np.ndarray, true_events: List[Tuple], 
                   intervals: List[Tuple[float, float]]) -> bool:
    """Validate inputs for metric calculations"""
    if predictions is None or len(predictions) == 0:
        print("ERROR: Predictions are empty")
        return False
    
    if len(true_events) == 0:
        print("ERROR: No true events provided")
        return False
    
    if len(intervals) == 0:
        print("ERROR: No intervals provided")
        return False
    
    if predictions.shape[0] != len(intervals):
        print(f"ERROR: Predictions shape {predictions.shape} doesn't match intervals count {len(intervals)}")
        return False
    
    return True

def debug_prediction_quality(predictions: np.ndarray, true_events: List[Tuple],
                           intervals: List[Tuple[float, float]]):
    """Debug function to check prediction quality"""
    print("\n=== Prediction Quality Debug ===")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Total predicted events: {np.sum(predictions):.2f}")
    print(f"Number of true events: {len(true_events)}")
    print(f"Number of intervals: {len(intervals)}")
    
    # Check for NaN or Inf
    if np.any(np.isnan(predictions)):
        print("WARNING: Predictions contain NaN values")
    if np.any(np.isinf(predictions)):
        print("WARNING: Predictions contain Inf values")
    
    # Check prediction ranges
    print(f"Prediction range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
    
    # Count true events per interval
    true_counts_per_interval = []
    for i, (start, end) in enumerate(intervals):
        interval_events = [e for e in true_events if start <= e[0] <= end]
        true_counts_per_interval.append(len(interval_events))
    
    print(f"True events per interval: {true_counts_per_interval}")
    print(f"Total true events in intervals: {sum(true_counts_per_interval)}")
    
    # Compare with predictions
    pred_counts_per_interval = np.sum(predictions, axis=1)
    print(f"Predicted events per interval: {pred_counts_per_interval}")
    print("=== End Debug ===\n")

def calculate_mape(predictions: np.ndarray, true_events: List[Tuple], 
                  intervals: List[Tuple[float, float]]) -> float:
    """
    Calculate Mean Absolute Percentage Error (Equation 14 from DHP paper)
    MAPE = (1/M) * Σ_m |(Σ_s predicted - Σ_s true) / Σ_s true|
    
    Fixed to handle cases where total_true = 0
    """
    if not validate_inputs(predictions, true_events, intervals):
        return float('inf')
    
    M = predictions.shape[1]  # Number of communities
    total_mape = 0.0
    valid_communities = 0
    
    print(f"\nMAPE Calculation Debug:")
    print(f"Number of communities: {M}")
    print(f"Number of intervals: {len(intervals)}")
    print(f"Total true events: {len(true_events)}")
    
    for m in range(M):
        total_predicted = np.sum(predictions[:, m])
        
        # Count true events in each interval for community m
        true_counts = []
        for start, end in intervals:
            count = sum(1 for event in true_events 
                       if start <= event[0] <= end and event[1] == m)
            true_counts.append(count)
        
        total_true = np.sum(true_counts)
        
        print(f"Community {m}: predicted={total_predicted:.4f}, true={total_true}")
        
        if total_true > 0:
            # Normal case: use MAPE formula
            mape_m = abs(total_predicted - total_true) / total_true
            total_mape += mape_m
            valid_communities += 1
            print(f"  MAPE_{m} = {mape_m:.4f}")
        elif total_predicted == 0:
            # Perfect prediction: both are zero
            mape_m = 0.0
            total_mape += mape_m
            valid_communities += 1
            print(f"  MAPE_{m} = 0.0 (perfect)")
        else:
            # No true events but predicted some - use normalized absolute error
            # Normalize by average events per community to avoid infinity
            avg_events_per_community = len(true_events) / max(1, M)
            if avg_events_per_community > 0:
                mape_m = abs(total_predicted) / avg_events_per_community
            else:
                mape_m = abs(total_predicted)  # Fallback to absolute error
            total_mape += mape_m
            valid_communities += 1
            print(f"  MAPE_{m} = {mape_m:.4f} (normalized)")
    
    final_mape = total_mape / valid_communities if valid_communities > 0 else float('inf')
    print(f"Final MAPE: {final_mape:.4f} (averaged over {valid_communities} communities)")
    
    return final_mape

def calculate_nll(predictions: np.ndarray, true_events: List[Tuple],
                 intervals: List[Tuple[float, float]]) -> float:
    """
    Calculate Negative Log Likelihood
    Using Poisson assumption for event counts in intervals
    """
    if not validate_inputs(predictions, true_events, intervals):
        return float('inf')
    
    total_nll = 0.0
    count = 0
    
    for i, (start, end) in enumerate(intervals):
        for m in range(predictions.shape[1]):
            lambda_pred = max(predictions[i, m], 1e-10)  # Avoid log(0)
            
            # Count true events in this interval and community
            true_count = sum(1 for event in true_events 
                           if start <= event[0] <= end and event[1] == m)
            
            if true_count >= 0:
                # Poisson log likelihood with safe computation
                try:
                    # log(P(true_count | lambda_pred)) = true_count * log(lambda_pred) - lambda_pred - log(true_count!)
                    log_factorial = math.lgamma(true_count + 1) if true_count > 0 else 0.0
                    log_likelihood = true_count * math.log(lambda_pred) - lambda_pred - log_factorial
                    nll = -log_likelihood
                    total_nll += nll
                    count += 1
                except (ValueError, OverflowError):
                    # Handle numerical issues
                    continue
    
    return total_nll / count if count > 0 else float('inf')

def calculate_accuracy(predictions: np.ndarray, true_events: List[Tuple],
                      intervals: List[Tuple[float, float]], threshold: float = 0.5) -> float:
    """
    Calculate prediction accuracy for event occurrence
    """
    if not validate_inputs(predictions, true_events, intervals):
        return 0.0
    
    correct = 0
    total = 0
    
    for i, (start, end) in enumerate(intervals):
        for m in range(predictions.shape[1]):
            predicted_count = predictions[i, m]
            true_count = sum(1 for event in true_events 
                           if start <= event[0] <= end and event[1] == m)
            
            predicted_occurrence = 1 if predicted_count > threshold else 0
            true_occurrence = 1 if true_count > 0 else 0
            
            if predicted_occurrence == true_occurrence:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0

def calculate_rmse(predictions: np.ndarray, true_events: List[Tuple],
                  intervals: List[Tuple[float, float]]) -> float:
    """
    Calculate Root Mean Square Error as an alternative metric
    """
    if not validate_inputs(predictions, true_events, intervals):
        return float('inf')
    
    total_se = 0.0
    count = 0
    
    for i, (start, end) in enumerate(intervals):
        for m in range(predictions.shape[1]):
            predicted_count = predictions[i, m]
            true_count = sum(1 for event in true_events 
                           if start <= event[0] <= end and event[1] == m)
            
            error = predicted_count - true_count
            total_se += error * error
            count += 1
    
    return math.sqrt(total_se / count) if count > 0 else float('inf')