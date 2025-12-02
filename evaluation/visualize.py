import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

class Visualization:
    """
    Visualization tools for DHP model results
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_training_history(self, history: Dict[str, List], save_path: str = None):
        """Plot training and validation loss history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot losses
        ax1.plot(history['train_losses'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_losses'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot log likelihoods
        ax2.plot(history['train_lls'], label='Train LL', linewidth=2)
        ax2.plot(history['val_lls'], label='Val LL', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Log Likelihood')
        ax2.set_title('Log Likelihood History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_latent_dynamics(self, latent_dynamics: Dict, save_path: str = None):
        """Plot latent dynamics functions"""
        num_communities = len(latent_dynamics)
        cols = min(3, num_communities)
        rows = (num_communities + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        
        for i, (comm_id, dynamics) in enumerate(latent_dynamics.items()):
            row = i // cols
            col = i % cols
            
            if rows > 1:
                ax = axes[row][col]
            else:
                ax = axes[col] if cols > 1 else axes
            
            times = dynamics['times']
            f_vals = dynamics['f']
            F_vals = dynamics['F']
            
            ax.plot(times, f_vals, 'b-', linewidth=2, label='f(t)')
            ax.set_xlabel('Time')
            ax.set_ylabel('f(t)')
            ax.set_title(f'Community {comm_id} - Latent Dynamics')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(num_communities, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row][col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_influence_network(self, influence_matrix: np.ndarray, save_path: str = None):
        """Plot influence network between communities"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(influence_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Source Community')
        ax.set_ylabel('Target Community')
        ax.set_title('Influence Network')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Influence Strength')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()