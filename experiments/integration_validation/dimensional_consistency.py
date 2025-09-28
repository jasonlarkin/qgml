"""
Quick Dimensional Consistency Test for QGML Integration.

This module specifically tests the dimensional consistency fixes across all experiments
to verify the bug is completely resolved before running long experiments.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import QGML models
from qgml.learning.specialized.genomics import ChromosomalInstabilityTrainer
from qgml.learning.supervised_trainer import SupervisedMatrixTrainer
from qgml.learning.specialized.regression import QGMLRegressionTrainer


class DimensionalConsistencyTester:
    """Quick tester for dimensional consistency across QGML models."""
    
    # TEST CONFIGURATION - Small for speed
    TEST_N_FEATURES = 6      # Fixed feature dimension
    TEST_N_SAMPLES = 50      # Small sample size
    TEST_EPOCHS = 10         # Minimal training
    TEST_HILBERT_DIM = 4     # Small Hilbert space
    
    def __init__(self):
        """Initialize tester."""
        self.start_time = time.time()
        self.results = {}
        
        print(f"Dimensional Consistency Test Initialized")
        print(f"Test config: Features={self.TEST_N_FEATURES}, Samples={self.TEST_N_SAMPLES}, Epochs={self.TEST_EPOCHS}")
    
    def generate_test_data(self, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate minimal test data with exact feature count."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data with EXACT feature count
        X = torch.randn(self.TEST_N_SAMPLES, self.TEST_N_FEATURES, dtype=torch.float32)
        
        # Simple LST generation
        lst_values = 10 + 15 * torch.randn(self.TEST_N_SAMPLES)
        lst_values = torch.clamp(lst_values, 0, 40)
        
        # Binary classification
        y_binary = (lst_values > 12).float()
        
        print(f"Generated test data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, lst_values, y_binary
    
    def test_model_creation_consistency(self):
        """Test that all models can be created with consistent dimensions."""
        print("\nTest 1: Model Creation Consistency")
        print("=" * 45)
        
        X, y_lst, y_binary = self.generate_test_data()
        
        # Test all model types with EXACT dimension matching
        models_to_test = {
            'chromosomal_mixed': lambda: ChromosomalInstabilityTrainer(
                N=self.TEST_HILBERT_DIM,
                D=self.TEST_N_FEATURES,  # Must match data exactly
                lst_threshold=12.0,
                use_mixed_loss=True,
                learning_rate=0.01,
                device='cpu'
            ),
            'chromosomal_povm': lambda: ChromosomalInstabilityTrainer(
                N=self.TEST_HILBERT_DIM,
                D=self.TEST_N_FEATURES,  # Must match data exactly
                use_mixed_loss=True,
                use_povm=True,
                learning_rate=0.01,
                device='cpu'
            ),
            'supervised_standard': lambda: SupervisedMatrixTrainer(
                N=self.TEST_HILBERT_DIM,
                D=self.TEST_N_FEATURES,  # Must match data exactly
                task_type='regression',
                learning_rate=0.01,
                device='cpu'
            ),
            'qgml_original': lambda: QGMLRegressionTrainer(
                N=self.TEST_HILBERT_DIM,
                D=self.TEST_N_FEATURES,  # Must match data exactly
                learning_rate=0.01,
                device='cpu'
            )
        }
        
        created_models = {}
        creation_results = {}
        
        for model_name, model_factory in models_to_test.items():
            try:
                print(f"Creating {model_name}...")
                model = model_factory()
                
                # Verify dimensions
                assert model.D == self.TEST_N_FEATURES, f"Model {model_name} has D={model.D}, expected {self.TEST_N_FEATURES}"
                assert model.D == X.shape[1], f"Model {model_name} dimension mismatch with data"
                
                created_models[model_name] = model
                creation_results[model_name] = "SUCCESS"
                print(f"  Model D={model.D}, Data features={X.shape[1]} - MATCH")
                
            except Exception as e:
                creation_results[model_name] = f"FAILED: {e}"
                print(f"  FAILED: {e}")
        
        self.results['model_creation'] = creation_results
        return created_models, X, y_lst, y_binary
    
    def test_quantum_state_computation(self, models: Dict, X: torch.Tensor):
        """Test that quantum state computation works without dimension errors."""
        print("\nTest 2: Quantum State Computation")
        print("=" * 45)
        
        state_computation_results = {}
        
        for model_name, model in models.items():
            try:
                print(f"Testing quantum states for {model_name}...")
                
                # Test on first few samples
                for i in range(min(3, len(X))):
                    x_sample = X[i]
                    
                    # Verify dimensions before computation
                    assert len(x_sample) == model.D, f"Sample {i} has {len(x_sample)} features, model expects {model.D}"
                    
                    # Compute quantum state
                    psi = model.compute_ground_state(x_sample)
                    
                    # Verify quantum state properties
                    assert psi.shape[0] == model.N, f"Quantum state has wrong dimension: {psi.shape[0]} vs {model.N}"
                    assert torch.isfinite(psi).all(), f"Quantum state contains non-finite values"
                    
                    print(f"  Sample {i}: |ψ⟩ shape={psi.shape}, norm={torch.norm(psi).item():.3f}")
                
                state_computation_results[model_name] = "SUCCESS"
                print(f"  All quantum state computations successful")
                
            except Exception as e:
                state_computation_results[model_name] = f"FAILED: {e}"
                print(f"  FAILED: {e}")
        
        self.results['quantum_states'] = state_computation_results
        return state_computation_results
    
    def test_training_consistency(self, models: Dict, X: torch.Tensor, y_lst: torch.Tensor, y_binary: torch.Tensor):
        """Test that training works without dimension errors."""
        print("\nTest 3: Training Consistency")
        print("=" * 45)
        
        # Split data
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_lst_train, y_lst_test = y_lst[:n_train], y_lst[n_train:]
        y_bin_train, y_bin_test = y_binary[:n_train], y_binary[n_train:]
        
        training_results = {}
        
        for model_name, model in models.items():
            try:
                print(f"Training {model_name}...")
                
                # Verify training data dimensions
                assert X_train.shape[1] == model.D, f"Training data has {X_train.shape[1]} features, model expects {model.D}"
                
                # Train model
                if 'chromosomal' in model_name:
                    model.fit_chromosomal_instability(
                        X_train, y_lst_train,
                        n_epochs=self.TEST_EPOCHS,
                        batch_size=16,
                        validation_split=0.2,
                        verbose=False
                    )
                elif model_name == 'supervised_standard':
                    model.fit(
                        X_train, y_lst_train,
                        n_epochs=self.TEST_EPOCHS,
                        batch_size=16,
                        X_val=X_test, y_val=y_lst_test,
                        verbose=False
                    )
                elif model_name == 'qgml_original':
                    model.fit(
                        X_train, y_lst_train,
                        epochs=self.TEST_EPOCHS,
                        batch_size=16,
                        verbose=False
                    )
                
                # Test prediction on test data
                assert X_test.shape[1] == model.D, f"Test data has {X_test.shape[1]} features, model expects {model.D}"
                
                # Make predictions
                if hasattr(model, 'evaluate_chromosomal_instability'):
                    metrics = model.evaluate_chromosomal_instability(X_test, y_lst_test, y_bin_test)
                elif hasattr(model, 'evaluate'):
                    metrics = model.evaluate(X_test, y_lst_test)
                else:
                    predictions = model.predict(X_test)
                    metrics = {'predictions_generated': True}
                
                training_results[model_name] = "SUCCESS"
                print(f"  Training and prediction successful")
                
            except Exception as e:
                training_results[model_name] = f"FAILED: {e}"
                print(f"  FAILED: {e}")
        
        self.results['training'] = training_results
        return training_results
    
    def test_cross_experiment_consistency(self):
        """Test that models work across different data generation scenarios."""
        print("\nTest 4: Cross-Experiment Consistency")
        print("=" * 45)
        
        cross_experiment_results = {}
        
        # Test different scenarios like the real experiments
        scenarios = [
            {'name': 'scenario_1', 'samples': 40, 'seed': 42},
            {'name': 'scenario_2', 'samples': 50, 'seed': 123}, 
            {'name': 'scenario_3', 'samples': 30, 'seed': 456}
        ]
        
        for scenario in scenarios:
            try:
                print(f"Testing {scenario['name']}...")
                
                # Generate data for this scenario
                X, y_lst, y_binary = self.generate_test_data(seed=scenario['seed'])
                X = X[:scenario['samples']]  # Truncate to scenario size
                y_lst = y_lst[:scenario['samples']]
                y_binary = y_binary[:scenario['samples']]
                
                # Create fresh model for this scenario
                model = ChromosomalInstabilityTrainer(
                    N=self.TEST_HILBERT_DIM,
                    D=self.TEST_N_FEATURES,  # MUST match data
                    learning_rate=0.01,
                    device='cpu'
                )
                
                # Verify dimensions
                assert X.shape[1] == model.D, f"Scenario {scenario['name']}: data {X.shape[1]} != model {model.D}"
                
                # Quick training
                n_train = int(0.8 * len(X))
                X_train = X[:n_train]
                y_train = y_lst[:n_train]
                
                model.fit_chromosomal_instability(
                    X_train, y_train,
                    n_epochs=5,  # Very quick
                    batch_size=8,
                    verbose=False
                )
                
                # Test prediction
                X_test = X[n_train:]
                for i in range(len(X_test)):
                    psi = model.compute_ground_state(X_test[i])
                    assert psi.shape[0] == model.N, f"Wrong quantum state dimension"
                
                cross_experiment_results[scenario['name']] = "SUCCESS"
                print(f"  {scenario['name']}: {len(X)} samples, {X.shape[1]} features - SUCCESS")
                
            except Exception as e:
                cross_experiment_results[scenario['name']] = f"FAILED: {e}"
                print(f"  {scenario['name']} FAILED: {e}")
        
        self.results['cross_experiment'] = cross_experiment_results
        return cross_experiment_results
    
    def generate_test_report(self):
        """Generate final test report."""
        total_time = time.time() - self.start_time
        
        print("\nDIMENSIONAL CONSISTENCY TEST REPORT")
        print("=" * 50)
        print(f"Total test time: {total_time:.1f} seconds")
        
        all_passed = True
        
        for test_name, test_results in self.results.items():
            print(f"\n{test_name.upper()}:")
            
            for item_name, result in test_results.items():
                if "SUCCESS" in result:
                    print(f"  {item_name}: PASSED")
                else:
                    print(f"  {item_name}: {result}")
                    all_passed = False
        
        print(f"\n{'='*50}")
        if all_passed:
            print("ALL TESTS PASSED - Dimensional consistency is FIXED!")
            print("Ready to run full advanced experiments")
        else:
            print("TESTS FAILED - Dimensional consistency issues remain")
            print("Need to fix remaining issues before running full experiments")
        
        return all_passed


def main():
    """Run dimensional consistency tests."""
    print("Dimensional Consistency Test Suite")
    print("=" * 40)
    
    tester = DimensionalConsistencyTester()
    
    # Test 1: Model creation
    models, X, y_lst, y_binary = tester.test_model_creation_consistency()
    
    # Test 2: Quantum state computation  
    tester.test_quantum_state_computation(models, X)
    
    # Test 3: Training consistency
    tester.test_training_consistency(models, X, y_lst, y_binary)
    
    # Test 4: Cross-experiment consistency
    tester.test_cross_experiment_consistency()
    
    # Generate report
    all_passed = tester.generate_test_report()
    
    return tester, all_passed


if __name__ == "__main__":
    tester, passed = main()
