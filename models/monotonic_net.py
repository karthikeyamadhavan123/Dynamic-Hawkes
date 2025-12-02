import tensorflow as tf

# Force TensorFlow 1.x compatibility
tf = tf.compat.v1
tf.disable_v2_behavior()

import numpy as np

class MonotonicMixtureNetwork:
    """
    Monotonic Mixture Neural Network for modeling F_m(t) and f_m(t)
    
    Following Equations 8-10 from the paper:
    - F_m(t) = Σ_c π_c Φ^c_m(t) + b_0 * t  (Equation 8)
    - f_m(t) = Σ_c π_c φ^c_m(t) + b_0      (Equation 10)
    
    where Φ^c_m is a monotonic neural network and φ^c_m is its derivative.
    
    Architecture details from Section 5.3 and Appendix D.2:
    - Number of layers: {1, 2, 3, 4, 5} (tuned via grid search)
    - Hidden units per layer: 8
    - Activation: tanh for hidden layers, softplus for output
    - Weight constraints: non-negative for monotonicity
    """
    
    def __init__(self, num_mixtures=3, hidden_units=8, num_layers=2, name="monotonic_net"):
        self.num_mixtures = num_mixtures
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.name = name
        
        self.build_network()
    
    def build_network(self):
        """Build the monotonic mixture network"""
        with tf.variable_scope(self.name):
            # Mixture weights π_c (must be non-negative, sum to valid values)
            # Using softmax ensures they sum to 1 and are non-negative
            self.mixture_weights_raw = tf.get_variable(
                "mixture_weights_raw",
                shape=[self.num_mixtures],
                initializer=tf.ones_initializer()
            )
            self.mixture_weights = tf.nn.softmax(self.mixture_weights_raw)
            
            # Bias term b_0 (must be non-negative for monotonicity)
            self.bias_raw = tf.get_variable(
                "bias_raw",
                shape=[1],
                initializer=tf.constant_initializer(0.1)
            )
            self.bias = tf.nn.softplus(self.bias_raw)
            
            # Build mixture components (each is a separate monotonic NN)
            self.component_weights = []
            self.component_biases = []
            
            for c in range(self.num_mixtures):
                component_w = []
                component_b = []
                
                with tf.variable_scope(f"component_{c}"):
                    # Build L layers for component c
                    for layer in range(self.num_layers):
                        if layer == 0:
                            # Input layer: 1 input (time t)
                            input_dim = 1
                            output_dim = self.hidden_units
                        elif layer == self.num_layers - 1:
                            # Output layer: hidden_units -> 1
                            input_dim = self.hidden_units
                            output_dim = 1
                        else:
                            # Hidden layers
                            input_dim = self.hidden_units
                            output_dim = self.hidden_units
                        
                        # Weights (must be non-negative for monotonicity)
                        w_raw = tf.get_variable(
                            f"w{layer}_raw",
                            shape=[input_dim, output_dim],
                            initializer=tf.random_normal_initializer(mean=0.1, stddev=0.01)
                        )
                        # Use softplus to ensure non-negativity (smoother than ReLU)
                        w = tf.nn.softplus(w_raw)
                        
                        # Biases (can be any value)
                        b = tf.get_variable(
                            f"b{layer}",
                            shape=[output_dim],
                            initializer=tf.zeros_initializer()
                        )
                        
                        component_w.append(w)
                        component_b.append(b)
                
                self.component_weights.append(component_w)
                self.component_biases.append(component_b)
    
    def _forward_component(self, t, component_idx):
        """
        Forward pass through one monotonic neural network component Φ^c_m(t)
        
        Following Equation 9:
        h^(l) = σ(W^(l) h^(l-1) + b^(l))
        
        where σ is a monotonic activation function.
        """
        t_reshaped = tf.reshape(t, [-1, 1])
        h = t_reshaped
        
        weights = self.component_weights[component_idx]
        biases = self.component_biases[component_idx]
        
        for layer in range(self.num_layers):
            h = tf.matmul(h, weights[layer]) + biases[layer]
            
            # Activation function (Section 4.1 and references [6, 33])
            if layer < self.num_layers - 1:
                # Hidden layers: tanh (monotonic and bounded)
                h = tf.nn.tanh(h)
            else:
                # Output layer: softplus (ensures positive output)
                h = tf.nn.softplus(h)
        
        return h
    
    def _gradient_component(self, t, component_idx):
        """
        Compute gradient φ^c_m(t) = dΦ^c_m(t)/dt using automatic differentiation
        
        This is more accurate than manual chain rule implementation.
        """
        t_tensor = tf.reshape(t, [-1, 1])
        
        # Use TensorFlow's automatic differentiation
        output = self._forward_component(t_tensor, component_idx)
        
        # Compute gradient with respect to input
        grad = tf.gradients(output, t_tensor)[0]
        
        if grad is None:
            # Fallback to manual computation if auto-diff fails
            return self._manual_gradient_component(t, component_idx)
        
        return grad
    
    def _manual_gradient_component(self, t, component_idx):
        """
        Manual gradient computation using chain rule (fallback)
        
        For a monotonic network: φ(t) = ∂Φ/∂t
        Using chain rule: ∂Φ/∂t = (∂Φ/∂h^(L-1)) * (∂h^(L-1)/∂h^(L-2)) * ... * (∂h^(1)/∂t)
        """
        t_reshaped = tf.reshape(t, [-1, 1])
        
        weights = self.component_weights[component_idx]
        biases = self.component_biases[component_idx]
        
        # Forward pass to store intermediate activations
        h_list = [t_reshaped]
        
        for layer in range(self.num_layers):
            h_prev = h_list[-1]
            linear = tf.matmul(h_prev, weights[layer]) + biases[layer]
            
            if layer < self.num_layers - 1:
                h = tf.nn.tanh(linear)
            else:
                h = tf.nn.softplus(linear)
            
            h_list.append(h)
        
        # Backward pass to compute gradient
        grad = tf.ones_like(h_list[-1])
        
        for layer in range(self.num_layers - 1, -1, -1):
            h_prev = h_list[layer]
            linear = tf.matmul(h_prev, weights[layer]) + biases[layer]
            
            # Compute activation derivative
            if layer < self.num_layers - 1:
                # tanh derivative: 1 - tanh^2(x)
                activation_deriv = 1.0 - tf.square(tf.nn.tanh(linear))
            else:
                # softplus derivative: sigmoid(x)
                activation_deriv = tf.nn.sigmoid(linear)
            
            # Chain rule
            grad = grad * activation_deriv
            grad = tf.matmul(grad, tf.transpose(weights[layer]))
        
        return grad
    
    def F(self, t):
        """
        Compute F_m(t) = Σ_c π_c Φ^c_m(t) + b_0 * t  (Equation 8)
        
        This is the integral of the latent dynamics function f_m(t).
        """
        with tf.variable_scope(self.name, reuse=True):
            t_reshaped = tf.reshape(t, [-1, 1])
            
            # Compute mixture: Σ_c π_c Φ^c_m(t)
            F_mixture = tf.zeros([tf.shape(t_reshaped)[0], 1], dtype=tf.float32)
            
            for c in range(self.num_mixtures):
                Phi_c = self._forward_component(t, c)
                F_mixture += self.mixture_weights[c] * Phi_c
            
            # Add linear bias term: b_0 * t
            F_total = F_mixture + self.bias * t_reshaped
            
            return tf.reshape(F_total, [-1])
    
    def f(self, t):
        """
        Compute f_m(t) = Σ_c π_c φ^c_m(t) + b_0  (Equation 10)
        
        This is the latent dynamics function (derivative of F_m(t)).
        Must be non-negative to preserve monotonicity of F_m(t).
        """
        with tf.variable_scope(self.name, reuse=True):
            t_reshaped = tf.reshape(t, [-1, 1])
            
            # Compute mixture: Σ_c π_c φ^c_m(t)
            f_mixture = tf.zeros([tf.shape(t_reshaped)[0], 1], dtype=tf.float32)
            
            for c in range(self.num_mixtures):
                # Get gradient of component c
                phi_c = self._manual_gradient_component(t, c)
                f_mixture += self.mixture_weights[c] * phi_c
            
            # Add bias (derivative of b_0*t is b_0)
            f_total = f_mixture + self.bias
            
            # Ensure non-negativity (critical for monotonicity)
            # The bias b_0 is already non-negative, and all weights are non-negative,
            # so theoretically f should be non-negative. But we add a small epsilon
            # and use softplus for numerical stability.
            f_total = tf.nn.softplus(f_total) + 1e-8
            
            return tf.reshape(f_total, [-1])
    
    def get_parameters(self):
        """Get network parameters for inspection/debugging"""
        params = {
            'mixture_weights': self.mixture_weights,
            'bias': self.bias,
            'num_components': self.num_mixtures,
            'num_layers': self.num_layers,
            'hidden_units': self.hidden_units
        }
        return params