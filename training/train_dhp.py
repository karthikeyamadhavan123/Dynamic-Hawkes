import tensorflow as tf

# Force TensorFlow 1.x compatibility
tf = tf.compat.v1
tf.disable_v2_behavior()

import numpy as np
import os
import time
import json
from typing import List, Tuple, Dict, Any

from models.dhp_model import DynamicHawkesProcess
from utils.metrics import calculate_mape, calculate_nll, validate_inputs, debug_prediction_quality
from utils.helpers import create_directory, format_time

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class DHPTrainer:
    """
    Complete trainer for Dynamic Hawkes Process model
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.sess = None
        self.saver = None
        
    def setup_model(self, num_communities: int):
        """Initialize DHP model and TensorFlow session"""
        print(f"Setting up DHP model with {num_communities} communities...")
        
        # Clear any existing graph
        tf.reset_default_graph()
        
        # Check what parameters the model actually accepts
        model_params = {
            'num_communities': num_communities,
            'num_mixtures': self.config.get('num_mixtures', 3),
            'hidden_units': self.config.get('hidden_units', 8),  # Paper uses 8
            'learning_rate': self.config.get('learning_rate', 0.002),  # Paper uses 0.002
            'kernel_type': self.config.get('kernel_type', 'power_law')  # Paper uses power-law
        }
        
        # Add num_layers only if it exists in the config and model supports it
        if 'num_layers' in self.config:
            try:
                self.model = DynamicHawkesProcess(**model_params, num_layers=self.config['num_layers'])
            except TypeError:
                # Model doesn't support num_layers, proceed without it
                print(f"Note: Model doesn't support num_layers parameter, using default")
                self.model = DynamicHawkesProcess(**model_params)
        else:
            self.model = DynamicHawkesProcess(**model_params)
        
        # Create session with optimized config
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=session_config)
        
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        
        # Create saver
        self.saver = tf.train.Saver(max_to_keep=3)
        
        print("DHP model setup complete!")
    
    def prepare_batch_data(self, events: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert events to batch format
        
        CRITICAL: Events must be sorted by time for correct computation
        """
        if len(events) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Sort events by time (CRITICAL for correct likelihood computation)
        events_sorted = sorted(events, key=lambda x: x[0])
        
        event_times = np.array([e[0] for e in events_sorted], dtype=np.float32)
        community_ids = np.array([e[1] for e in events_sorted], dtype=np.int32)
        source_communities = np.array([e[2] for e in events_sorted], dtype=np.int32)
        
        return event_times, community_ids, source_communities
    
    def train(self, train_events: List[Tuple], val_events: List[Tuple], 
              test_events: List[Tuple], num_communities: int) -> Tuple[Dict, List, List]:
        """
        Main training loop for DHP model
        
        Key optimizations based on paper:
        1. Use ADAM optimizer with lr=0.002, beta1=0.9, beta2=0.999
        2. Use early stopping based on validation log-likelihood
        3. Compute NLL per event (not total) for fair comparison
        """
        print("Starting DHP training...")
        
        self.setup_model(num_communities)
        
        # Prepare data (MUST be sorted)
        train_times, train_comms, train_sources = self.prepare_batch_data(train_events)
        val_times, val_comms, val_sources = self.prepare_batch_data(val_events)
        
        if len(train_times) == 0:
            raise ValueError("No training events provided")
        
        # Observation period - should be the maximum time in training data
        T_train = float(np.max(train_times))
        T_val = float(np.max(val_times)) if len(val_times) > 0 else T_train
        
        print(f"Training period: [0, {T_train:.2f}]")
        print(f"Validation period: [0, {T_val:.2f}]")
        print(f"Number of training events: {len(train_times)}")
        print(f"Number of validation events: {len(val_times)}")
        
        # Training history
        train_losses = []
        val_losses = []
        
        best_val_nll = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        max_patience = self.config.get('patience', 20)
        max_epochs = self.config.get('epochs', 100)
        
        print(f"\n{'Epoch':^6} | {'Train NLL':^12} | {'Val NLL':^12} | {'Time':^8} | {'Patience':^8}")
        print("-" * 65)
        
        for epoch in range(max_epochs):
            start_time = time.time()
            
            # Training step
            train_feed_dict = {
                self.model.event_times: train_times,
                self.model.community_ids: train_comms,
                self.model.source_communities: train_sources,
                self.model.T: T_train
            }
            
            # Run training operation
            try:
                _, train_loss_val, train_ll_val = self.sess.run(
                    [self.model.train_op, self.model.loss, self.model.log_likelihood],
                    feed_dict=train_feed_dict
                )
                
                # Convert to NLL per event for comparison with paper
                train_nll_per_event = -train_ll_val / len(train_times)
                
            except Exception as e:
                print(f"Error during training step: {e}")
                train_nll_per_event = float('inf')
            
            # Validation step
            if len(val_times) > 0:
                val_feed_dict = {
                    self.model.event_times: val_times,
                    self.model.community_ids: val_comms,
                    self.model.source_communities: val_sources,
                    self.model.T: T_val
                }
                
                try:
                    val_ll_val = self.sess.run(self.model.log_likelihood, feed_dict=val_feed_dict)
                    # NLL per event
                    val_nll_per_event = -val_ll_val / len(val_times)
                except Exception as e:
                    print(f"Error during validation: {e}")
                    val_nll_per_event = float('inf')
            else:
                val_nll_per_event = train_nll_per_event
            
            # Store history
            train_losses.append(float(train_nll_per_event))
            val_losses.append(float(val_nll_per_event))
            
            epoch_time = time.time() - start_time
            
            # Early stopping check (based on validation NLL per event)
            if val_nll_per_event < best_val_nll:
                best_val_nll = val_nll_per_event
                patience_counter = 0
                best_epoch = epoch
                # Save best model
                self.save_model('best_dhp_model')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == 0:
                print(f"{epoch:6d} | {train_nll_per_event:12.4f} | {val_nll_per_event:12.4f} | "
                      f"{format_time(epoch_time):>8} | {patience_counter:8d}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation NLL: {best_val_nll:.4f} at epoch {best_epoch}")
                break
        
        # Load best model for final evaluation
        try:
            self.load_model('best_dhp_model')
            print(f"\nBest model from epoch {best_epoch} loaded for evaluation")
        except Exception as e:
            print(f"Could not load best model: {e}")
            print("Using final model for evaluation")
        
        # Final evaluation on test set
        # Use the full observation period for test evaluation
        test_results = self.evaluate(test_events=test_events, history_events=train_events,validation_events=val_events,T=T_train)
        
        return test_results, train_losses, val_losses
    
    def evaluate(self, test_events: List[Tuple], history_events: List[Tuple], 
             validation_events: List[Tuple], T: float) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the trained model
        
        CRITICAL: NLL should be computed per event for comparison with paper
        """
        print("\nEvaluating model on test set...")
        
        # Prepare data (sorted)
        test_times, test_comms, test_sources = self.prepare_batch_data(test_events)
        hist_times, hist_comms, hist_sources = self.prepare_batch_data(history_events)
        val_times, val_comms, val_sources = self.prepare_batch_data(validation_events)
        
        if len(test_times) == 0:
            print("No test events to evaluate")
            return {
                'test_nll': float('inf'),
                'test_nll_per_event': float('inf'),
                'test_mape': float('inf'),
                'predictions': [],
                'future_intervals': [],
                'latent_dynamics': {},
                'influence_matrix': []
            }
        
        # Use test period end time
        T_test = float(np.max(test_times))
        
        # Calculate test log-likelihood
        try:
            test_ll_val = self.sess.run(self.model.log_likelihood, feed_dict={
                self.model.event_times: test_times,
                self.model.community_ids: test_comms,
                self.model.source_communities: test_sources,
                self.model.T: T_test
            })
            
            # Total NLL
            test_nll = -test_ll_val
            # NLL per event (this is what the paper reports)
            test_nll_per_event = test_nll / len(test_times)
            
        except Exception as e:
            print(f"Error calculating test NLL: {e}")
            test_nll = float('inf')
            test_nll_per_event = float('inf')
        
        # Create future intervals for prediction
        future_intervals = self._create_evaluation_intervals(test_times)
        
        # Get predictions for MAPE calculation
        predictions_val = []
        try:
            # Create future intervals from test data if not provided
            if len(future_intervals) == 0 or all(t[1] - t[0] <= 0 for t in future_intervals):
                print("Creating evaluation intervals from test data...")
                future_intervals = self._create_evaluation_intervals(test_times, num_intervals=5)
            
            print(f"\nPrediction Details:")
            print(f"  History events: {len(hist_times)}")
            print(f"  Test events: {len(test_times)}")
            print(f"  Prediction intervals: {len(future_intervals)}")
            print(f"  First interval: {future_intervals[0]}")
            print(f"  Last interval: {future_intervals[-1]}")
            print(f"  Test time range: [{np.min(test_times):.4f}, {np.max(test_times):.4f}]")
            
            # Use ALL events (train + val) as history for prediction
            all_history_times = np.concatenate([hist_times, val_times]) if len(val_times) > 0 else hist_times
            all_history_comms = np.concatenate([hist_comms, val_comms]) if len(val_comms) > 0 else hist_comms
            all_history_sources = np.concatenate([hist_sources, val_sources]) if len(val_sources) > 0 else hist_sources
            
            # Sort history by time
            sort_idx = np.argsort(all_history_times)
            all_history_times = all_history_times[sort_idx]
            all_history_comms = all_history_comms[sort_idx]
            all_history_sources = all_history_sources[sort_idx]
            
            print(f"  Total history events: {len(all_history_times)}")
            
            # Predict using the model
            predictions_val = self.sess.run(
                self.model.predict_event_counts(
                    tf.constant(all_history_times, dtype=tf.float32),
                    tf.constant(all_history_comms, dtype=tf.int32),
                    tf.constant(all_history_sources, dtype=tf.int32),
                    tf.constant(future_intervals, dtype=tf.float32)
                )
            )
            
            print(f"  Prediction shape: {predictions_val.shape}")
            print(f"  Total predicted events: {np.sum(predictions_val):.2f}")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            predictions_val = np.zeros((len(future_intervals), self.model.num_communities))
        
        # Calculate MAPE with debugging - REMOVED REDUNDANT IMPORT
        try:
            # Validate inputs
            inputs_valid = validate_inputs(predictions_val, test_events, future_intervals)
            
            if inputs_valid:
                # Debug prediction quality
                debug_prediction_quality(predictions_val, test_events, future_intervals)
                
                # Calculate MAPE
                mape = calculate_mape(predictions_val, test_events, future_intervals)
            else:
                print("Input validation failed, cannot calculate MAPE")
                mape = float('inf')
                
        except Exception as e:
            print(f"Error calculating MAPE: {e}")
            import traceback
            traceback.print_exc()
            mape = float('inf')
        
        # Get latent dynamics for analysis
        latent_dynamics = {}
        try:
            sample_times = np.linspace(0, T_test, 10)
            for comm_id in range(min(3, self.model.num_communities)):
                F_vals, f_vals = self.sess.run(
                    self.model.get_latent_dynamics(
                        comm_id, 
                        tf.constant(sample_times, dtype=tf.float32)
                    )
                )
                latent_dynamics[comm_id] = {
                    'F': convert_numpy_types(F_vals),
                    'f': convert_numpy_types(f_vals),
                    'times': convert_numpy_types(sample_times)
                }
        except Exception as e:
            print(f"Error getting latent dynamics: {e}")
        
        # Get influence matrix
        influence_matrix = []
        try:
            influence_matrix = self.sess.run(self.model.influence_matrix)
            influence_matrix = convert_numpy_types(influence_matrix)
        except Exception as e:
            print(f"Error getting influence matrix: {e}")
        
        results = {
            'test_nll': convert_numpy_types(test_nll),
            'test_nll_per_event': convert_numpy_types(test_nll_per_event),
            'test_mape': convert_numpy_types(mape),
            'predictions': convert_numpy_types(predictions_val),
            'future_intervals': convert_numpy_types(future_intervals),
            'latent_dynamics': convert_numpy_types(latent_dynamics),
            'influence_matrix': convert_numpy_types(influence_matrix)
        }
        
        print(f"\nTest Results:")
        print(f"  Total NLL: {test_nll:.4f}")
        print(f"  NLL per event: {test_nll_per_event:.4f}")
        print(f"  MAPE: {mape:.4f}")
        
        return results
    
    def _create_evaluation_intervals(self, test_times: np.ndarray, num_intervals: int = 5) -> List[Tuple[float, float]]:
        """Create evaluation intervals for MAPE calculation"""
        if len(test_times) == 0:
            return [(0.8, 0.84), (0.84, 0.88), (0.88, 0.92), (0.92, 0.96), (0.96, 1.0)]
        
        min_time = float(np.min(test_times))
        max_time = float(np.max(test_times))
        interval_length = (max_time - min_time) / num_intervals
        
        intervals = []
        for i in range(num_intervals):
            start = min_time + i * interval_length
            end = min_time + (i + 1) * interval_length
            intervals.append((float(start), float(end)))
        
        return intervals
    
    def save_model(self, model_name: str):
        """Save model to disk"""
        try:
            create_directory('saved_models')
            save_path = os.path.join('saved_models', model_name)
            self.saver.save(self.sess, save_path)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, model_name: str):
        """Load model from disk"""
        try:
            load_path = os.path.join('saved_models', model_name)
            self.saver.restore(self.sess, load_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def close(self):
        """Clean up resources"""
        if self.sess:
            self.sess.close()
            print("Training session closed")