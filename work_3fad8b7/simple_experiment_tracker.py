#!/usr/bin/env python3
"""
Simple Experiment Tracker
A lightweight alternative to MLflow for tracking experiments
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class SimpleExperimentTracker:
    """Simple experiment tracking without external dependencies"""
    
    def __init__(self, base_dir: str = "test_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create master index
        self.index_file = self.base_dir / "experiment_index.json"
        self.load_index()
    
    def load_index(self):
        """Load or create the experiment index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                'experiments': [],
                'last_updated': datetime.now().isoformat()
            }
    
    def save_index(self):
        """Save the experiment index"""
        self.index['last_updated'] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def start_experiment(self, 
                        experiment_name: str,
                        parameters: Dict[str, Any],
                        description: str = "") -> str:
        """Start a new experiment and return experiment ID"""
        
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create experiment directory
        experiment_dir = self.base_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment metadata
        experiment_data = {
            'experiment_id': experiment_id,
            'name': experiment_name,
            'description': description,
            'start_time': datetime.now().isoformat(),
            'parameters': parameters,
            'status': 'running',
            'results': {},
            'files': []
        }
        
        # Save experiment metadata
        with open(experiment_dir / 'experiment.json', 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        # Add to index
        self.index['experiments'].append({
            'id': experiment_id,
            'name': experiment_name,
            'start_time': experiment_data['start_time'],
            'status': 'running'
        })
        self.save_index()
        
        print(f"🚀 Started experiment: {experiment_id}")
        print(f"   Directory: {experiment_dir}")
        
        return experiment_id
    
    def log_parameter(self, experiment_id: str, key: str, value: Any):
        """Log a parameter value during the experiment"""
        experiment_file = self.base_dir / experiment_id / 'experiment.json'
        
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                experiment_data = json.load(f)
            
            experiment_data['parameters'][key] = value
            
            with open(experiment_file, 'w') as f:
                json.dump(experiment_data, f, indent=2)
    
    def log_result(self, experiment_id: str, key: str, value: Any):
        """Log a result value during the experiment"""
        experiment_file = self.base_dir / experiment_id / 'experiment.json'
        
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                experiment_data = json.load(f)
            
            experiment_data['results'][key] = value
            
            with open(experiment_file, 'w') as f:
                json.dump(experiment_data, f, indent=2)
    
    def log_file(self, experiment_id: str, file_path: str, description: str = ""):
        """Log a file generated during the experiment"""
        experiment_file = self.base_dir / experiment_id / 'experiment.json'
        
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                experiment_data = json.load(f)
            
            experiment_data['files'].append({
                'path': file_path,
                'description': description,
                'timestamp': datetime.now().isoformat()
            })
            
            with open(experiment_file, 'w') as f:
                json.dump(experiment_data, f, indent=2)
    
    def end_experiment(self, experiment_id: str, status: str = 'completed'):
        """End an experiment and update its status"""
        experiment_file = self.base_dir / experiment_id / 'experiment.json'
        
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                experiment_data = json.load(f)
            
            experiment_data['end_time'] = datetime.now().isoformat()
            experiment_data['status'] = status
            
            # Calculate duration
            start_time = datetime.fromisoformat(experiment_data['start_time'])
            end_time = datetime.fromisoformat(experiment_data['end_time'])
            duration = (end_time - start_time).total_seconds()
            experiment_data['duration_seconds'] = duration
            
            with open(experiment_file, 'w') as f:
                json.dump(experiment_data, f, indent=2)
        
        # Update index
        for exp in self.index['experiments']:
            if exp['id'] == experiment_id:
                exp['status'] = status
                exp['end_time'] = datetime.now().isoformat()
                break
        
        self.save_index()
        
        print(f"✅ Completed experiment: {experiment_id}")
        print(f"   Status: {status}")
        if 'duration_seconds' in experiment_data:
            print(f"   Duration: {duration:.2f} seconds")
    
    def list_experiments(self):
        """List all experiments"""
        print(f"\n📋 Experiment Index ({len(self.index['experiments'])} experiments)")
        print("=" * 80)
        
        for exp in self.index['experiments']:
            status_emoji = "🟢" if exp['status'] == 'completed' else "🟡" if exp['status'] == 'running' else "🔴"
            print(f"{status_emoji} {exp['id']}")
            print(f"   Name: {exp['name']}")
            print(f"   Status: {exp['status']}")
            print(f"   Start: {exp['start_time']}")
            if 'end_time' in exp:
                print(f"   End: {exp['end_time']}")
            print()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment data by ID"""
        experiment_file = self.base_dir / experiment_id / 'experiment.json'
        
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                return json.load(f)
        return None
    
    def compare_experiments(self, experiment_ids: list):
        """Compare multiple experiments"""
        experiments = []
        for exp_id in experiment_ids:
            exp_data = self.get_experiment(exp_id)
            if exp_data:
                experiments.append(exp_data)
        
        if len(experiments) < 2:
            print("Need at least 2 experiments to compare")
            return
        
        print(f"\n🔍 Comparing {len(experiments)} experiments")
        print("=" * 80)
        
        # Compare parameters
        print("Parameters:")
        for exp in experiments:
            print(f"  {exp['experiment_id']}: {exp['parameters']}")
        
        # Compare results
        print("\nResults:")
        for exp in experiments:
            print(f"  {exp['experiment_id']}: {exp['results']}")

# Example usage
if __name__ == "__main__":
    tracker = SimpleExperimentTracker()
    
    # Example experiment
    exp_id = tracker.start_experiment(
        name="test_experiment",
        parameters={
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        },
        description="Testing the experiment tracker"
    )
    
    # Log some results
    tracker.log_result(exp_id, 'final_loss', 0.123)
    tracker.log_result(exp_id, 'accuracy', 0.95)
    
    # End experiment
    tracker.end_experiment(exp_id)
    
    # List all experiments
    tracker.list_experiments()
