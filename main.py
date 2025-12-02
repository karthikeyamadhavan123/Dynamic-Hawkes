# main.py
import argparse
import json
import numpy as np
from data.data_processor import DHPDataProcessor
from training.train_dhp import DHPTrainer
from utils.config import get_config

def convert_to_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description='Dynamic Hawkes Process Implementation')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['reddit', 'news', 'protest', 'dblp'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset file')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration preset')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = get_config(args.config)
        
        # Process data
        processor = DHPDataProcessor()
        
        if args.dataset == 'reddit':
            data = processor.process_reddit_data(args.data_path)
        elif args.dataset == 'news':
            data = processor.process_news_data(args.data_path)
        elif args.dataset == 'protest':
            data = processor.process_protest_data(args.data_path)
        elif args.dataset == 'dblp':
            data = processor.process_dblp_data(args.data_path)
        
        # Split data
        train_events, val_events, test_events = processor.temporal_split(data['events'])
        
        print(f"Dataset: {args.dataset}")
        print(f"Total events: {len(data['events'])}")
        print(f"Train/Val/Test: {len(train_events)}/{len(val_events)}/{len(test_events)}")
        print(f"Number of communities: {data['num_communities']}")
        
        # Train model
        trainer = DHPTrainer(config)
        results, train_losses, val_losses = trainer.train(
            train_events, val_events, test_events, data['num_communities']
        )
        
        # Convert all NumPy types to native Python types for JSON serialization
        output = {
            'dataset': args.dataset,
            'config': convert_to_serializable(config),
            'results': convert_to_serializable(results),
            'train_losses': convert_to_serializable(train_losses),
            'val_losses': convert_to_serializable(val_losses)
        }
        
        # Save results
        with open(f'results_{args.dataset}.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        trainer.close()
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()