import os
import json
import argparse
from data.data_processor import DHPDataProcessor
from training.train_dhp import DHPTrainer
from training.train_baselines import BaselineTrainer
from evaluation.evaluate import Evaluator
from utils.config import get_config

def run_dhp_experiment(dataset_name, data_path, config_name='default'):
    """Run DHP experiment on a specific dataset"""
    print(f"Running DHP experiment on {dataset_name} dataset...")
    
    # Load and process data
    processor = DHPDataProcessor()
    
    if dataset_name == 'reddit':
        data = processor.process_reddit_data(data_path)
    elif dataset_name == 'news':
        data = processor.process_news_data(data_path)
    elif dataset_name == 'protest':
        data = processor.process_protest_data(data_path)
    elif dataset_name == 'dblp':
        data = processor.process_dblp_data(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split data
    train_events, val_events, test_events = processor.temporal_split(data['events'])
    
    # Get configuration
    config = get_config(dataset_name)
    
    # Train DHP model
    trainer = DHPTrainer(config)
    results, history = trainer.train(train_events, val_events, test_events, data['num_communities'])
    
    # Save results
    experiment_results = {
        'dataset': dataset_name,
        'config': config,
        'results': results,
        'history': history
    }
    
    output_dir = 'experiment_results'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/dhp_{dataset_name}_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    trainer.close()
    
    return experiment_results

def run_baseline_experiment(dataset_name, data_path, baseline_type):
    """Run baseline experiment"""
    print(f"Running {baseline_type} baseline on {dataset_name} dataset...")
    
    # Similar setup as DHP experiment
    processor = DHPDataProcessor()
    
    if dataset_name == 'reddit':
        data = processor.process_reddit_data(data_path)
    elif dataset_name == 'news':
        data = processor.process_news_data(data_path)
    elif dataset_name == 'protest':
        data = processor.process_protest_data(data_path)
    elif dataset_name == 'dblp':
        data = processor.process_dblp_data(data_path)
    
    train_events, val_events, test_events = processor.temporal_split(data['events'])
    
    config = get_config('default')
    trainer = BaselineTrainer(baseline_type, config)
    results = trainer.train(train_events, val_events, test_events, data['num_communities'])
    
    trainer.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run DHP experiments')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['reddit', 'news', 'protest', 'dblp', 'all'],
                       help='Dataset to run experiments on')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--model', type=str, default='dhp',
                       choices=['dhp', 'all_baselines', 'hpp', 'rpp', 'self_correcting', 'hawkes', 'rmtpp'],
                       help='Model to run')
    
    args = parser.parse_args()
    
    datasets = ['reddit', 'news', 'protest', 'dblp'] if args.dataset == 'all' else [args.dataset]
    
    all_results = {}
    
    for dataset in datasets:
        data_path = f'{args.data_dir}/{dataset}_processed.pkl'
        
        if args.model in ['dhp', 'all_baselines']:
            # Run DHP
            dhp_results = run_dhp_experiment(dataset, data_path)
            all_results[f'dhp_{dataset}'] = dhp_results
        
        if args.model in ['all_baselines', 'hpp', 'rpp', 'self_correcting', 'hawkes', 'rmtpp']:
            # Run baselines
            baselines = ['hpp', 'rpp', 'self_correcting', 'hawkes', 'rmtpp'] if args.model == 'all_baselines' else [args.model]
            
            for baseline in baselines:
                baseline_results = run_baseline_experiment(dataset, data_path, baseline)
                all_results[f'{baseline}_{dataset}'] = baseline_results
    
    # Compare results
    evaluator = Evaluator()
    # Add evaluation logic here
    
    print("All experiments completed!")

if __name__ == "__main__":
    main()