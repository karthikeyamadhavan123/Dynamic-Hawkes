# test.py
import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()

import numpy as np
from models.dhp_model import DynamicHawkesProcess

print("Testing Dynamic Hawkes Process model...")

# Create a simple test
with tf.Session() as sess:
    # Create model with very small parameters
    model = DynamicHawkesProcess(
        num_communities=2,  # Small number for testing
        num_mixtures=2, 
        hidden_units=8,     # Small network
        learning_rate=0.001
    )
    
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    print("✓ Model created successfully")
    print("✓ Variables initialized")
    
    # Test that required attributes exist
    required_attrs = ['train_op', 'loss', 'log_likelihood']
    for attr in required_attrs:
        if hasattr(model, attr):
            print(f"✓ {attr} exists")
        else:
            print(f"✗ {attr} missing")
    
    # Test with very simple dummy data
    dummy_times = np.array([0.1, 0.5], dtype=np.float32)
    dummy_communities = np.array([0, 1], dtype=np.int32)
    dummy_sources = np.array([1, 0], dtype=np.int32)
    T = 1.0
    
    try:
        # Run a training step
        _, loss_val, ll_val = sess.run(
            [model.train_op, model.loss, model.log_likelihood],
            feed_dict={
                model.event_times: dummy_times,
                model.community_ids: dummy_communities,
                model.source_communities: dummy_sources,
                model.T: T
            }
        )
        print(f"✓ Training step successful")
        print(f"  Loss: {loss_val:.4f}, Log Likelihood: {ll_val:.4f}")
        
        # Test prediction
        future_intervals = np.array([[0.8, 1.0], [1.0, 1.2]], dtype=np.float32)
        predictions = sess.run(
            model.predict_event_counts(
                tf.constant(dummy_times),
                tf.constant(dummy_communities),
                tf.constant(dummy_sources),
                tf.constant(future_intervals)
            )
        )
        print(f"✓ Prediction successful")
        print(f"  Predictions shape: {predictions.shape}")
        
    except Exception as e:
        print(f"✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("Test completed!")