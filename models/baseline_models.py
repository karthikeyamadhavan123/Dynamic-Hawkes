# models/baseline_models.py
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional

class HPP:
    """
    Homogeneous Poisson Process Baseline
    Constant intensity for each community
    """
    
    def __init__(self, num_communities: int, learning_rate: float = 0.01):
        self.num_communities = num_communities
        self.learning_rate = learning_rate
        
        # Model parameters - constant rate for each community
        self.rates = tf.get_variable(
            "hpp_rates",
            shape=[num_communities],
            initializer=tf.ones_initializer(),
            constraint=lambda x: tf.nn.relu(x) + 1e-8
        )
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # Placeholders
        self.event_times = tf.placeholder(tf.float32, [None], name="event_times")
        self.community_ids = tf.placeholder(tf.int32, [None], name="community_ids")
        self.T = tf.placeholder(tf.float32, name="T")
    
    def intensity(self, community_ids):
        """Constant intensity for each community"""
        return tf.gather(self.rates, community_ids)
    
    def log_likelihood(self, event_times, community_ids, T):
        """
        Log likelihood for Homogeneous Poisson Process
        L = Σ log(λ_{c_i}) - Σ λ_c * T
        """
        # Term 1: Sum of log intensities at event times
        intensities = self.intensity(community_ids)
        term1 = tf.reduce_sum(tf.log(intensities + 1e-8))
        
        # Term 2: Integral of intensity over [0, T]
        # For HPP, each community contributes λ_c * T
        term2 = tf.reduce_sum(self.rates) * T
        
        return term1 - term2
    
    def train(self, event_times, community_ids, T):
        """Training operation to maximize log likelihood"""
        log_likelihood = self.log_likelihood(event_times, community_ids, T)
        loss = -log_likelihood  # Negative log likelihood
        
        train_op = self.optimizer.minimize(loss)
        return train_op, loss, log_likelihood
    
    def predict_event_counts(self, future_intervals):
        """
        Predict number of events in future intervals
        For HPP: λ_c * (t_end - t_start) for each community
        """
        predictions = []
        
        for t_start, t_end in future_intervals:
            interval_length = t_end - t_start
            counts = self.rates * interval_length
            predictions.append(counts)
        
        return tf.stack(predictions)

class RPP:
    """
    Reinforced Poisson Process Baseline
    Intensity depends on cumulative count and aging effect
    """
    
    def __init__(self, num_communities: int, learning_rate: float = 0.01):
        self.num_communities = num_communities
        self.learning_rate = learning_rate
        
        # Parameters for relaxation function (Equation 17 from DHP paper)
        self.alpha = tf.get_variable(
            "rpp_alpha",
            shape=[num_communities],
            initializer=tf.ones_initializer()
        )
        self.beta = tf.get_variable(
            "rpp_beta", 
            shape=[num_communities],
            initializer=tf.ones_initializer(),
            constraint=lambda x: tf.nn.relu(x) + 1e-8
        )
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # Placeholders
        self.event_times = tf.placeholder(tf.float32, [None], name="event_times")
        self.community_ids = tf.placeholder(tf.int32, [None], name="community_ids")
        self.T = tf.placeholder(tf.float32, name="T")
    
    def relaxation_function(self, t, community_ids):
        """
        Relaxation function γ_m(t) (Equation 17)
        γ_m(t) = exp(-(log t - α_m)² / (2β_m²)) / (√(2π) β_m t)
        """
        alpha_vals = tf.gather(self.alpha, community_ids)
        beta_vals = tf.gather(self.beta, community_ids)
        
        log_t = tf.log(t + 1e-8)
        exponent = -tf.square(log_t - alpha_vals) / (2 * tf.square(beta_vals))
        
        numerator = tf.exp(exponent)
        denominator = tf.sqrt(2 * np.pi) * beta_vals * t
        
        return numerator / (denominator + 1e-8)
    
    def compute_event_counts(self, event_times, community_ids, current_time):
        """Compute cumulative event counts for each community up to current_time"""
        batch_size = tf.shape(event_times)[0]
        counts = tf.TensorArray(tf.float32, size=self.num_communities)
        
        for m in range(self.num_communities):
            # Count events for community m that occurred before current_time
            mask = tf.logical_and(
                community_ids == m,
                event_times < current_time
            )
            count = tf.reduce_sum(tf.cast(mask, tf.float32))
            counts = counts.write(m, count)
        
        return counts.stack()
    
    def intensity(self, t, event_times, community_ids):
        """
        RPP intensity (Equation 16)
        λ_m(t) = γ_m(t) * N_m(t)
        where N_m(t) is cumulative count of events for community m
        """
        # Get event counts for each community
        counts = self.compute_event_counts(event_times, community_ids, t)
        
        # Get relaxation function values
        gamma = self.relaxation_function(t, tf.range(self.num_communities))
        
        return gamma * counts
    
    def log_likelihood(self, event_times, community_ids, T):
        """Compute log likelihood for RPP"""
        # This is computationally intensive - we'll use a simplified version
        # In practice, you'd need to compute intensity at each event time
        
        intensities = tf.TensorArray(tf.float32, size=tf.shape(event_times)[0])
        
        for i in range(tf.shape(event_times)[0]):
            t_i = event_times[i]
            m_i = community_ids[i]
            
            # Intensity at event time
            intensity = self.intensity(t_i, event_times, community_ids)[m_i]
            intensities = intensities.write(i, intensity)
        
        intensities_stack = intensities.stack()
        
        # Term 1: Sum of log intensities
        term1 = tf.reduce_sum(tf.log(intensities_stack + 1e-8))
        
        # Term 2: Integral of intensity (simplified)
        term2 = 0.0
        for m in range(self.num_communities):
            # Approximate integral - in practice this would be more complex
            term2 += tf.gather(self.compute_event_counts(event_times, community_ids, T), [m])[0]
        
        return term1 - term2
    
    def train(self, event_times, community_ids, T):
        """Training operation"""
        log_likelihood = self.log_likelihood(event_times, community_ids, T)
        loss = -log_likelihood
        
        train_op = self.optimizer.minimize(loss)
        return train_op, loss, log_likelihood

class SelfCorrecting:
    """
    Self-correcting Point Process Baseline
    Intensity increases linearly and is corrected by past events
    """
    
    def __init__(self, num_communities: int, learning_rate: float = 0.01):
        self.num_communities = num_communities
        self.learning_rate = learning_rate
        
        # Parameters (Equation 18 from DHP paper)
        self.alpha = tf.get_variable(
            "sc_alpha",
            shape=[num_communities],
            initializer=tf.zeros_initializer()
        )
        self.beta = tf.get_variable(
            "sc_beta",
            shape=[num_communities],
            initializer=tf.ones_initializer(),
            constraint=lambda x: tf.nn.relu(x) + 1e-8
        )
        self.rho = tf.get_variable(
            "sc_rho", 
            shape=[num_communities],
            initializer=tf.ones_initializer(),
            constraint=lambda x: tf.nn.relu(x) + 1e-8
        )
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # Placeholders
        self.event_times = tf.placeholder(tf.float32, [None], name="event_times")
        self.community_ids = tf.placeholder(tf.int32, [None], name="community_ids")
        self.T = tf.placeholder(tf.float32, name="T")
    
    def compute_event_counts(self, event_times, community_ids, current_time):
        """Compute cumulative event counts for each community"""
        counts = tf.TensorArray(tf.float32, size=self.num_communities)
        
        for m in range(self.num_communities):
            mask = tf.logical_and(
                community_ids == m,
                event_times < current_time
            )
            count = tf.reduce_sum(tf.cast(mask, tf.float32))
            counts = counts.write(m, count)
        
        return counts.stack()
    
    def intensity(self, t, event_times, community_ids):
        """
        Self-correcting intensity (Equation 18)
        λ_m(t) = exp(α_m + β_m * t - ρ_m * N_m(t))
        """
        counts = self.compute_event_counts(event_times, community_ids, t)
        
        alpha_vals = tf.gather(self.alpha, tf.range(self.num_communities))
        beta_vals = tf.gather(self.beta, tf.range(self.num_communities))
        rho_vals = tf.gather(self.rho, tf.range(self.num_communities))
        
        exponent = alpha_vals + beta_vals * t - rho_vals * counts
        return tf.exp(exponent)
    
    def log_likelihood(self, event_times, community_ids, T):
        """Compute log likelihood for Self-correcting process"""
        intensities = tf.TensorArray(tf.float32, size=tf.shape(event_times)[0])
        
        for i in range(tf.shape(event_times)[0]):
            t_i = event_times[i]
            m_i = community_ids[i]
            
            intensity = self.intensity(t_i, event_times, community_ids)[m_i]
            intensities = intensities.write(i, intensity)
        
        intensities_stack = intensities.stack()
        
        # Term 1: Sum of log intensities
        term1 = tf.reduce_sum(tf.log(intensities_stack + 1e-8))
        
        # Term 2: Integral of intensity (simplified)
        term2 = 0.0
        # This integral is complex - using approximation
        for m in range(self.num_communities):
            alpha_m = tf.gather(self.alpha, [m])[0]
            beta_m = tf.gather(self.beta, [m])[0]
            rho_m = tf.gather(self.rho, [m])[0]
            count_m = tf.gather(self.compute_event_counts(event_times, community_ids, T), [m])[0]
            
            # Approximate integral
            integral_approx = (tf.exp(alpha_m + beta_m * T - rho_m * count_m) - tf.exp(alpha_m)) / beta_m
            term2 += integral_approx
        
        return term1 - term2
    
    def train(self, event_times, community_ids, T):
        """Training operation"""
        log_likelihood = self.log_likelihood(event_times, community_ids, T)
        loss = -log_likelihood
        
        train_op = self.optimizer.minimize(loss)
        return train_op, loss, log_likelihood

class HawkesProcess:
    """
    Standard Hawkes Process Baseline
    Exponential kernel with static parameters
    """
    
    def __init__(self, num_communities: int, learning_rate: float = 0.001):
        self.num_communities = num_communities
        self.learning_rate = learning_rate
        
        # Model parameters
        self.background_rates = tf.get_variable(
            "hawkes_background",
            shape=[num_communities],
            initializer=tf.ones_initializer(),
            constraint=lambda x: tf.nn.relu(x) + 1e-8
        )
        
        self.influence_matrix = tf.get_variable(
            "hawkes_influence",
            shape=[num_communities, num_communities],
            initializer=tf.random_normal_initializer(mean=0.1, stddev=0.01),
            constraint=lambda x: tf.nn.relu(x)
        )
        
        self.decay_rates = tf.get_variable(
            "hawkes_decay",
            shape=[num_communities, num_communities],
            initializer=tf.ones_initializer(),
            constraint=lambda x: tf.nn.relu(x) + 1e-8
        )
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # Placeholders
        self.event_times = tf.placeholder(tf.float32, [None], name="event_times")
        self.community_ids = tf.placeholder(tf.int32, [None], name="community_ids")
        self.source_communities = tf.placeholder(tf.int32, [None], name="source_communities")
        self.T = tf.placeholder(tf.float32, name="T")
    
    def intensity(self, event_times, community_ids, source_communities):
        """
        Standard Hawkes process intensity
        λ_m(t) = μ_m + Σ_{j:t_j < t} α_{m,m_j} exp(-β_{m,m_j} (t - t_j))
        """
        batch_size = tf.shape(event_times)[0]
        intensities = tf.TensorArray(tf.float32, size=batch_size)
        
        for i in range(batch_size):
            t_i = event_times[i]
            m_i = community_ids[i]
            
            # Base intensity
            base = tf.gather(self.background_rates, [m_i])[0]
            
            # Excitation from historical events
            mask = tf.logical_and(
                event_times < t_i,
                tf.not_equal(community_ids, m_i)  # Only events from other communities
            )
            hist_indices = tf.where(mask)
            
            if tf.shape(hist_indices)[0] > 0:
                hist_times = tf.gather(event_times, hist_indices)
                hist_sources = tf.gather(source_communities, hist_indices)
                
                # Time differences
                time_diffs = t_i - hist_times
                
                # Get kernel parameters
                alpha_vals = tf.gather_nd(
                    self.influence_matrix,
                    tf.stack([
                        tf.tile([m_i], [tf.shape(hist_sources)[0]]),
                        tf.reshape(hist_sources, [-1])
                    ], axis=1)
                )
                beta_vals = tf.gather_nd(
                    self.decay_rates,
                    tf.stack([
                        tf.tile([m_i], [tf.shape(hist_sources)[0]]),
                        tf.reshape(hist_sources, [-1])
                    ], axis=1)
                )
                
                # Exponential kernel
                kernel_vals = alpha_vals * tf.exp(-beta_vals * time_diffs)
                excitation = tf.reduce_sum(kernel_vals)
            else:
                excitation = 0.0
            
            intensity = base + excitation
            intensities = intensities.write(i, intensity)
        
        return intensities.stack()
    
    def log_likelihood(self, event_times, community_ids, source_communities, T):
        """
        Log likelihood for Hawkes process
        """
        # Term 1: Sum of log intensities at event times
        intensities = self.intensity(event_times, community_ids, source_communities)
        term1 = tf.reduce_sum(tf.log(intensities + 1e-8))
        
        # Term 2: Integral of intensity over [0, T]
        term2 = 0.0
        
        # Background rate integral
        background_integral = tf.reduce_sum(self.background_rates) * T
        
        # Excitation integral
        excitation_integral = 0.0
        for m in range(self.num_communities):
            for m_j in range(self.num_communities):
                if m != m_j:
                    alpha = self.influence_matrix[m, m_j]
                    beta = self.decay_rates[m, m_j]
                    
                    # For each event from community m_j
                    mask = community_ids == m_j
                    event_times_mj = tf.boolean_mask(event_times, mask)
                    
                    if tf.shape(event_times_mj)[0] > 0:
                        # Integral of excitation from events of community m_j to community m
                        time_to_end = T - event_times_mj
                        integral_vals = (alpha / beta) * (1 - tf.exp(-beta * time_to_end))
                        excitation_integral += tf.reduce_sum(integral_vals)
        
        term2 = background_integral + excitation_integral
        
        return term1 - term2
    
    def train(self, event_times, community_ids, source_communities, T):
        """Training operation"""
        log_likelihood = self.log_likelihood(event_times, community_ids, source_communities, T)
        loss = -log_likelihood
        
        train_op = self.optimizer.minimize(loss)
        return train_op, loss, log_likelihood
    
    def predict_event_counts(self, history_times, history_communities, history_sources, future_intervals):
        """
        Predict number of events in future intervals
        """
        predictions = []
        
        for t_start, t_end in future_intervals:
            interval_length = t_end - t_start
            counts_per_community = []
            
            for m in range(self.num_communities):
                # Background events
                background_count = self.background_rates[m] * interval_length
                
                # Excitation from historical events
                excitation_count = 0.0
                
                for j in range(tf.shape(history_times)[0]):
                    t_j = history_times[j]
                    m_j = history_communities[j]
                    
                    if m_j != m:
                        alpha = self.influence_matrix[m, m_j]
                        beta = self.decay_rates[m, m_j]
                        
                        # Expected number of events caused by historical event
                        integral_start = (alpha / beta) * (1 - tf.exp(-beta * (t_start - t_j)))
                        integral_end = (alpha / beta) * (1 - tf.exp(-beta * (t_end - t_j)))
                        count_val = integral_end - integral_start
                        
                        excitation_count += count_val
                
                total_count = background_count + excitation_count
                counts_per_community.append(total_count)
            
            predictions.append(tf.stack(counts_per_community))
        
        return tf.stack(predictions)

class RMTPP:
    """
    Recurrent Marked Temporal Point Process Baseline
    Uses RNN to model event sequences
    """
    
    def __init__(self, num_communities: int, hidden_units: int = 64, 
                 learning_rate: float = 0.001, sequence_length: int = 50):
        self.num_communities = num_communities
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        
        # RNN cell
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
        
        # Output layers
        self.intensity_layer = tf.layers.Dense(1, activation=tf.exp, name="intensity_output")
        self.mark_layer = tf.layers.Dense(num_communities, activation=tf.nn.softmax, name="mark_output")
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # Placeholders
        self.event_sequences = tf.placeholder(tf.float32, [None, sequence_length, 3], name="event_sequences")
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name="sequence_lengths")
    
    def build_model(self, event_sequences, sequence_lengths):
        """
        Build RNN model for marked temporal point process
        Input: [batch_size, sequence_length, 3] - (time_delta, community_id, source_community)
        """
        # RNN over event sequences
        outputs, states = tf.nn.dynamic_rnn(
            self.lstm_cell,
            event_sequences,
            sequence_length=sequence_lengths,
            dtype=tf.float32
        )
        
        # Intensity prediction
        intensities = self.intensity_layer(outputs)  # [batch_size, sequence_length, 1]
        
        # Mark prediction (community prediction)
        mark_probs = self.mark_layer(outputs)  # [batch_size, sequence_length, num_communities]
        
        return intensities, mark_probs, states
    
    def log_likelihood(self, event_sequences, sequence_lengths):
        """
        Log likelihood for RMTPP
        This is a simplified version - actual RMTPP uses a more complex formulation
        """
        intensities, mark_probs, _ = self.build_model(event_sequences, sequence_lengths)
        
        batch_size = tf.shape(event_sequences)[0]
        seq_len = tf.shape(event_sequences)[1]
        
        total_log_likelihood = 0.0
        
        for i in range(batch_size):
            actual_length = sequence_lengths[i]
            
            for j in range(actual_length):
                # Get true next event time and mark
                if j < actual_length - 1:
                    true_time_delta = event_sequences[i, j+1, 0]
                    true_community = tf.cast(event_sequences[i, j+1, 1], tf.int32)
                    
                    # Predicted intensity and mark probability
                    pred_intensity = intensities[i, j, 0]
                    pred_mark_prob = mark_probs[i, j, true_community]
                    
                    # Log likelihood components
                    time_ll = tf.log(pred_intensity + 1e-8) - pred_intensity * true_time_delta
                    mark_ll = tf.log(pred_mark_prob + 1e-8)
                    
                    total_log_likelihood += time_ll + mark_ll
        
        return total_log_likelihood
    
    def train(self, event_sequences, sequence_lengths):
        """Training operation"""
        log_likelihood = self.log_likelihood(event_sequences, sequence_lengths)
        loss = -log_likelihood
        
        train_op = self.optimizer.minimize(loss)
        return train_op, loss, log_likelihood
    
    def predict_next_event(self, event_sequences, sequence_lengths):
        """Predict next event time and community"""
        intensities, mark_probs, states = self.build_model(event_sequences, sequence_lengths)
        
        # For prediction, we use the last output of each sequence
        batch_size = tf.shape(event_sequences)[0]
        
        last_intensities = tf.TensorArray(tf.float32, size=batch_size)
        last_mark_probs = tf.TensorArray(tf.float32, size=batch_size)
        
        for i in range(batch_size):
            seq_len = sequence_lengths[i]
            last_intensity = intensities[i, seq_len-1, 0]
            last_mark_prob = mark_probs[i, seq_len-1, :]
            
            last_intensities = last_intensities.write(i, last_intensity)
            last_mark_probs = last_mark_probs.write(i, last_mark_prob)
        
        return last_intensities.stack(), last_mark_probs.stack()

class BaselineModelFactory:
    """
    Factory class to create baseline models
    """
    
    @staticmethod
    def create_model(model_type: str, num_communities: int, **kwargs):
        """
        Create a baseline model instance
        
        Args:
            model_type: Type of baseline model ('hpp', 'rpp', 'self_correcting', 'hawkes', 'rmtpp')
            num_communities: Number of communities/dimensions
            **kwargs: Additional model-specific parameters
        """
        if model_type == 'hpp':
            learning_rate = kwargs.get('learning_rate', 0.01)
            return HPP(num_communities, learning_rate)
        
        elif model_type == 'rpp':
            learning_rate = kwargs.get('learning_rate', 0.01)
            return RPP(num_communities, learning_rate)
        
        elif model_type == 'self_correcting':
            learning_rate = kwargs.get('learning_rate', 0.01)
            return SelfCorrecting(num_communities, learning_rate)
        
        elif model_type == 'hawkes':
            learning_rate = kwargs.get('learning_rate', 0.001)
            return HawkesProcess(num_communities, learning_rate)
        
        elif model_type == 'rmtpp':
            learning_rate = kwargs.get('learning_rate', 0.001)
            hidden_units = kwargs.get('hidden_units', 64)
            sequence_length = kwargs.get('sequence_length', 50)
            return RMTPP(num_communities, hidden_units, learning_rate, sequence_length)
        
        else:
            raise ValueError(f"Unknown baseline model type: {model_type}")

# Convenience function to get all baseline models
def get_all_baselines(num_communities: int, **kwargs) -> dict:
    """
    Get all baseline models for comparison
    
    Returns:
        Dictionary of baseline model instances
    """
    model_types = ['hpp', 'rpp', 'self_correcting', 'hawkes', 'rmtpp']
    baselines = {}
    
    for model_type in model_types:
        baselines[model_type] = BaselineModelFactory.create_model(
            model_type, num_communities, **kwargs
        )
    
    return baselines

# Example usage
if __name__ == "__main__":
    # Test the baseline models
    num_communities = 10
    
    # Create all baseline models
    baselines = get_all_baselines(num_communities)
    
    print("Available baseline models:")
    for model_name, model in baselines.items():
        print(f"  - {model_name}: {type(model).__name__}")
    
    # Test HPP specifically
    hpp_model = baselines['hpp']
    print(f"\nHPP model parameters: {[var.name for var in tf.trainable_variables() if 'hpp' in var.name]}")