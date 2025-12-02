# scripts/run_all_datasets.py
import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    packages = [
        'praw', 'requests', 'pandas', 'numpy', 
        'tensorflow==1.15.0', 'matplotlib', 'seaborn'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def run_data_download():
    """Run all data download scripts"""
    scripts = [
        'scripts/download_reddit_data.py',
        'scripts/download_gdelt_data.py', 
        'scripts/process_acled_data.py',
        'scripts/download_chicago_data.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"\n📥 Running {script}...")
            try:
                subprocess.check_call([sys.executable, script])
            except subprocess.CalledProcessError:
                print(f"⚠️  {script} had issues (may need manual setup)")
        else:
            print(f"❌ {script} not found")

def run_dhp_experiments():
    """Run DHP on all downloaded datasets"""
    from data.data_processor import DHPDataProcessor
    from training.train_dhp import DHPTrainer
    from utils.config import get_config
    
    datasets = [
        ('reddit', 'data/raw/reddit_hyperlinks.csv'),
        ('news', 'data/raw/gdelt_covid_news.csv'),
        ('protest', 'data/processed/acled_protests_processed.csv'), 
        ('dblp', 'data/raw/dblp_data.csv')  # Use your existing DBLP data
    ]
    
    for dataset_name, data_path in datasets:
        if os.path.exists(data_path):
            print(f"\n🚀 Training DHP on {dataset_name}...")
            
            processor = DHPDataProcessor()
            
            if dataset_name == 'reddit':
                data = processor.process_reddit_data(data_path)
            elif dataset_name == 'news':
                data = processor.process_news_data(data_path)
            elif dataset_name == 'protest':
                data = processor.process_protest_data(data_path)
            elif dataset_name == 'dblp':
                data = processor.process_dblp_data(data_path)
            
            # Split and train
            train_events, val_events, test_events = processor.temporal_split(data['events'])
            config = get_config(dataset_name)
            trainer = DHPTrainer(config)
            results, history = trainer.train(train_events, val_events, test_events, data['num_communities'])
            
            print(f"✅ {dataset_name}: Test NLL = {results['test_nll']:.4f}")
            trainer.close()

if __name__ == "__main__":
    print("🚀 Starting Complete DHP Pipeline with Real APIs")
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Download data
    print("\n" + "="*50)
    print("DOWNLOADING REAL DATASETS")
    print("="*50)
    run_data_download()
    
    # Step 3: Run experiments
    print("\n" + "="*50)
    print("RUNNING DHP EXPERIMENTS")
    print("="*50)
    run_dhp_experiments()
    
    print("\n🎉 Pipeline completed!")