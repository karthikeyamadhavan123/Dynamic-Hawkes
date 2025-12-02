import os
import argparse
from data.data_processor import DHPDataProcessor

def preprocess_dataset(dataset_name, input_path, output_path):
    """Preprocess a specific dataset"""
    processor = DHPDataProcessor()
    
    if dataset_name == 'reddit':
        data = processor.process_reddit_data(input_path)
    elif dataset_name == 'news':
        data = processor.process_news_data(input_path)
    elif dataset_name == 'protest':
        data = processor.process_protest_data(input_path)
    elif dataset_name == 'dblp':
        data = processor.process_dblp_data(input_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Save processed data
    processor.save_processed_data(data, output_path)
    print(f"Processed {dataset_name} data saved to {output_path}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets for DHP')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['reddit', 'news', 'protest', 'dblp', 'all'],
                       help='Dataset to preprocess')
    parser.add_argument('--input_dir', type=str, default='data/raw',
                       help='Input directory for raw data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets = ['reddit', 'news', 'protest', 'dblp'] if args.dataset == 'all' else [args.dataset]
    
    for dataset in datasets:
        input_path = os.path.join(args.input_dir, f"{dataset}_data.csv")
        output_path = os.path.join(args.output_dir, f"{dataset}_processed.pkl")
        
        print(f"Preprocessing {dataset} data...")
        preprocess_dataset(dataset, input_path, output_path)

if __name__ == "__main__":
    main()