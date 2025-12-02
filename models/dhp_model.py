import tensorflow as tf

# Force TensorFlow 1.x compatibility
tf = tf.compat.v1
tf.disable_v2_behavior()

import numpy as np
from .monotonic_net import MonotonicMixtureNetwork

class DynamicHawkesProcess:
    """
    Dynamic Hawkes Process implementation following the KDD 2021 paper.
    Uses vectorized operations for efficiency.
    """
    
    def __init__(self, num_communities, num_mixtures=3, hidden_units=8, 
                 num_layers=2, learning_rate=0.002, kernel_type='power_law',
                 beta1=0.9, beta2=0.999):
        self.num_communities = num_communities
        self.num_mixtures = num_mixtures
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.kernel_type = kernel_type
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Power-law exponent (fixed at 2 per paper)
        if self.kernel_type == 'power_law' or self.kernel_type == 'pwl':
            self.p = 2.0
        
        # Placeholders
        self.event_times = tf.placeholder(tf.float32, [None], name="event_times")
        self.community_ids = tf.placeholder(tf.int32, [None], name="community_ids") 
        self.source_communities = tf.placeholder(tf.int32, [None], name="source_communities")
        self.T = tf.placeholder(tf.float32, name="T")
        
        self.build_model()
        self.build_training_ops()
        
    def build_model(self):
        """Build DHP model"""
        with tf.variable_scope("dhp_model"):
            # Background rates μ_m
            self._background_rates = tf.get_variable(
                "background_rates",
                shape=[self.num_communities],
                initializer=tf.constant_initializer(0.1)
            )
            self._background_rates_constrained = tf.nn.softplus(self._background_rates)
            
            # Influence matrix α_{m,m'}
            self._influence_matrix = tf.get_variable(
                "influence_matrix",
                shape=[self.num_communities, self.num_communities],
                initializer=tf.random_normal_initializer(mean=0.1, stddev=0.01)
            )
            self._influence_matrix_constrained = tf.nn.softplus(self._influence_matrix)
            
            # Decay rates β_{m,m'}
            self._decay_rates = tf.get_variable(
                "decay_rates",
                shape=[self.num_communities, self.num_communities],
                initializer=tf.constant_initializer(1.0)
            )
            self._decay_rates_constrained = tf.nn.softplus(self._decay_rates) + 1e-6
            
            # Monotonic networks for each community
            self.monotonic_nets = []
            for m in range(self.num_communities):
                net = MonotonicMixtureNetwork(
                    num_mixtures=self.num_mixtures,
                    hidden_units=self.hidden_units,
                    num_layers=self.num_layers,
                    name=f"monotonic_net_{m}"
                )
                self.monotonic_nets.append(net)
                
    def build_training_ops(self):
        """Build training operations"""
        with tf.variable_scope("training_ops"):
            self.log_likelihood = self.compute_log_likelihood()
            self.loss = -self.log_likelihood
            
            # ADAM optimizer with paper's hyperparameters
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=self.beta1,
                beta2=self.beta2
            )
            self.train_op = self.optimizer.minimize(self.loss)
    
    def compute_log_likelihood(self):
        """
        Compute log-likelihood: L = Σᵢ log λ_mᵢ(tᵢ) - Σₘ ∫₀ᵀ λₘ(t)dt
        
        Using vectorized operations for efficiency.
        """
        num_events = tf.shape(self.event_times)[0]
        
        # === First term: Σᵢ log λ_mᵢ(tᵢ) ===
        
        # Get F and f values for all events and all communities
        all_F_values = []
        all_f_values = []
        
        for m in range(self.num_communities):
            F_m = self.monotonic_nets[m].F(self.event_times)  # [num_events]
            f_m = self.monotonic_nets[m].f(self.event_times)  # [num_events]
            all_F_values.append(F_m)
            all_f_values.append(f_m)
        
        F_matrix = tf.stack(all_F_values, axis=1)  # [num_events, num_communities]
        f_matrix = tf.stack(all_f_values, axis=1)  # [num_events, num_communities]
        
        # Get F_m and f_m for each event's target community
        indices = tf.stack([tf.range(num_events), self.community_ids], axis=1)
        F_target = tf.gather_nd(F_matrix, indices)  # [num_events]
        f_target = tf.gather_nd(f_matrix, indices)  # [num_events]
        
        # Compute pairwise time differences: Δ̃ᵢⱼ = F_m(tᵢ) - F_m(tⱼ)
        F_target_expanded = tf.expand_dims(F_target, 1)  # [num_events, 1]
        F_target_tiled = tf.expand_dims(F_target, 0)     # [1, num_events]
        
        delta_tilde = F_target_expanded - F_target_tiled  # [num_events, num_events]
        
        # Create mask for j < i (causal mask)
        mask = tf.linalg.band_part(tf.ones([num_events, num_events]), -1, 0)
        mask = mask - tf.eye(num_events)  # Exclude diagonal
        
        # Get influence parameters α[m_i, m_j]
        community_pairs = tf.stack([
            tf.tile(tf.expand_dims(self.community_ids, 1), [1, num_events]),  # m_i
            tf.tile(tf.expand_dims(self.source_communities, 0), [num_events, 1])  # m_j
        ], axis=2)  # [num_events, num_events, 2]
        
        alpha_matrix = tf.gather_nd(self._influence_matrix_constrained, community_pairs)
        beta_matrix = tf.gather_nd(self._decay_rates_constrained, community_pairs)
        
        # Compute kernel values g(Δ̃ᵢⱼ)
        kernel_values = self.compute_kernel_vectorized(delta_tilde, alpha_matrix, beta_matrix)
        
        # Apply causal mask
        kernel_values = kernel_values * mask
        
        # Sum over j to get triggering term for each event i
        triggering_sum = tf.reduce_sum(kernel_values, axis=1)  # [num_events]
        
        # Get background rates for each event
        background = tf.gather(self._background_rates_constrained, self.community_ids)
        
        # Compute intensity: λ_m(tᵢ) = μₘ + f_m(tᵢ) * Σⱼ g(Δ̃ᵢⱼ)
        intensity = background + f_target * triggering_sum
        intensity = tf.maximum(intensity, 1e-10)
        
        log_intensity_sum = tf.reduce_sum(tf.log(intensity))
        
        # === Second term: Σₘ ∫₀ᵀ λₘ(t)dt ===
        
        integral_term = 0.0
        
        for m in range(self.num_communities):
            # Background integral: μₘ * T
            background_integral = self._background_rates_constrained[m] * self.T
            
            # Get F(T) and F(tⱼ) for all events
            F_T = self.monotonic_nets[m].F(tf.reshape(self.T, [1]))[0]
            F_events = all_F_values[m]  # [num_events]
            
            # Compute Δ = F(T) - F(tⱼ) for all events
            delta_T = F_T - F_events  # [num_events]
            
            # Get α[m, m_j] and β[m, m_j] for all events
            alpha_vec = tf.gather(self._influence_matrix_constrained[m], self.source_communities)
            beta_vec = tf.gather(self._decay_rates_constrained[m], self.source_communities)
            
            # Compute G(F(T) - F(tⱼ)) for all events
            G_T_vec = self.compute_kernel_integral_vectorized(delta_T, alpha_vec, beta_vec)
            
            # Since F(0) = 0, G(0 - F(tⱼ)) contribution
            delta_0 = -F_events
            G_0_vec = self.compute_kernel_integral_vectorized(delta_0, alpha_vec, beta_vec)
            
            triggering_integral = tf.reduce_sum(G_T_vec - G_0_vec)
            
            integral_term += (background_integral + triggering_integral)
        
        return log_intensity_sum - integral_term
    
    def compute_kernel_vectorized(self, delta, alpha, beta):
        """Vectorized kernel computation"""
        delta = tf.maximum(delta, 0.0)
        
        if self.kernel_type == 'exponential' or self.kernel_type == 'exp':
            return alpha * tf.exp(-beta * delta)
        
        elif self.kernel_type == 'power_law' or self.kernel_type == 'pwl':
            numerator = alpha * beta
            denominator = tf.pow(alpha + beta * delta, self.p + 1)
            return numerator / (denominator + 1e-10)
        
        elif self.kernel_type == 'raleigh' or self.kernel_type == 'ray':
            return alpha * delta * tf.exp(-beta * delta * delta)
        
        else:
            return alpha * tf.exp(-beta * delta)
    
    def compute_kernel_integral_vectorized(self, delta, alpha, beta):
        """Vectorized kernel integral computation"""
        delta = tf.maximum(delta, 0.0)
        
        if self.kernel_type == 'exponential' or self.kernel_type == 'exp':
            return -alpha / (beta + 1e-10) * tf.exp(-beta * delta)
        
        elif self.kernel_type == 'power_law' or self.kernel_type == 'pwl':
            return -alpha / self.p * tf.pow(alpha + beta * delta + 1e-10, -self.p)
        
        elif self.kernel_type == 'raleigh' or self.kernel_type == 'ray':
            return -alpha / (2.0 * beta + 1e-10) * tf.exp(-beta * delta * delta)
        
        else:
            return -alpha / (beta + 1e-10) * tf.exp(-beta * delta)
    
    def train(self, event_times, community_ids, source_communities, T):
        """Training operation"""
        return self.train_op, self.loss, self.log_likelihood
    
    def predict_event_counts(self, history_times, history_communities, history_sources, future_intervals):
        """Predict number of events in future intervals"""
        num_intervals = tf.shape(future_intervals)[0]
        num_history = tf.shape(history_times)[0]
        
        predictions_list = []
        
        for m in range(self.num_communities):
            # Get F values for interval boundaries
            interval_times = tf.reshape(future_intervals, [-1])  # Flatten start and end times
            F_intervals = self.monotonic_nets[m].F(interval_times)
            F_intervals = tf.reshape(F_intervals, [-1, 2])  # [num_intervals, 2]
            
            F_start = F_intervals[:, 0]
            F_end = F_intervals[:, 1]
            
            # Background contribution
            interval_lengths = future_intervals[:, 1] - future_intervals[:, 0]
            background_count = self._background_rates_constrained[m] * interval_lengths
            
            # Use tf.cond for conditional logic instead of Python if
            def compute_with_history():
                F_history = self.monotonic_nets[m].F(history_times)  # [num_history]
                
                # Get α[m, m_j] and β[m, m_j]
                alpha_vec = tf.gather(self._influence_matrix_constrained[m], history_sources)
                beta_vec = tf.gather(self._decay_rates_constrained[m], history_sources)
                
                # Compute for each interval using vectorized operations
                F_end_expanded = tf.expand_dims(F_end, 1)  # [num_intervals, 1]
                F_start_expanded = tf.expand_dims(F_start, 1)  # [num_intervals, 1]
                F_history_expanded = tf.expand_dims(F_history, 0)  # [1, num_history]
                
                delta_end = F_end_expanded - F_history_expanded  # [num_intervals, num_history]
                delta_start = F_start_expanded - F_history_expanded  # [num_intervals, num_history]
                
                # Tile alpha and beta to match dimensions
                alpha_tiled = tf.tile(tf.expand_dims(alpha_vec, 0), [num_intervals, 1])  # [num_intervals, num_history]
                beta_tiled = tf.tile(tf.expand_dims(beta_vec, 0), [num_intervals, 1])  # [num_intervals, num_history]
                
                G_end = self.compute_kernel_integral_vectorized(delta_end, alpha_tiled, beta_tiled)
                G_start = self.compute_kernel_integral_vectorized(delta_start, alpha_tiled, beta_tiled)
                
                triggering = tf.reduce_sum(G_end - G_start, axis=1)  # [num_intervals]
                return triggering
            
            def compute_without_history():
                return tf.zeros([num_intervals])
            
            # Use tf.cond to handle the conditional
            triggering_counts = tf.cond(
                tf.greater(num_history, 0),
                compute_with_history,
                compute_without_history
            )
            
            total_counts = background_count + triggering_counts
            predictions_list.append(total_counts)
        
        return tf.stack(predictions_list, axis=1)
    
    def get_latent_dynamics(self, community_id, times):
        """Get latent dynamics F(t) and f(t)"""
        F_vals = self.monotonic_nets[community_id].F(times)
        f_vals = self.monotonic_nets[community_id].f(times)
        return F_vals, f_vals
    
    def get_influence_network(self, time):
        """Get time-dependent influence network"""
        time_tensor = tf.reshape(time, [1])
        influences = []
        
        for m in range(self.num_communities):
            f_m = self.monotonic_nets[m].f(time_tensor)[0]
            row = []
            for m2 in range(self.num_communities):
                influence = self._influence_matrix_constrained[m, m2] * f_m
                row.append(influence)
            influences.append(tf.stack(row))
        
        return tf.stack(influences)

    @property
    def influence_matrix(self):
        return self._influence_matrix_constrained
    
    @property
    def background_rates(self):
        return self._background_rates_constrained
    
    @property
    def decay_rates(self):
        return self._decay_rates_constrained