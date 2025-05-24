"""
Run systematic sentiment analysis experiments.

This script runs multiple experiments to compare different approaches
and find the best performing model configuration.
"""

import sys
import os

# Add src to path
sys.path.append('src')

# Import experiment runner
from experiments.experiment_runner import ExperimentRunner, create_experiment_configs

def main():
    """Run all experiments and compare results."""
    print("🚀 Starting Systematic Sentiment Analysis Experiments")
    print("="*60)
    
    # Initialize runner
    runner = ExperimentRunner("experiments/results")
    
    # Setup data
    print("📊 Setting up data...")
    runner.setup_data(
        "data/tweet_sentiment.train.jsonl",
        "data/tweet_sentiment.test.jsonl", 
        "embeddings/glove.6B.100d.txt"
    )
    
    # Create experiment configurations
    print("⚙️ Creating experiment configurations...")
    configs = create_experiment_configs()
    
    print(f"📋 Configured {len(configs)} experiments:")
    for config in configs:
        print(f"  - {config.experiment_id}: {config.description}")
        runner.add_experiment(config)
    
    # Run all experiments
    print(f"\n🏃 Running {len(configs)} experiments...")
    results = runner.run_all_experiments()
    
    # Print summary
    print(f"\n📈 Experiment Results:")
    runner.print_comparison_summary()
    
    # Get best experiment
    best = runner.get_best_experiment()
    if best:
        print(f"\n🏆 WINNER: {best.experiment_id}")
        print(f"   Test F1 Score: {best.test_f1:.4f}")
        print(f"   Configuration: {best.config.model_type} with {best.config.aggregation_method} aggregation")
    
    print(f"\n💾 Results saved to: experiments/results/")
    print(f"   - experiment_comparison.csv")
    print(f"   - experiment_comparison.json")
    print(f"   - Individual result files")

if __name__ == "__main__":
    main() 