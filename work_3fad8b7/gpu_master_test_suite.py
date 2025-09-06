#!/usr/bin/env python3
"""
GPU Master Test Suite
Orchestrates all GPU-focused optimizer comparison experiments.

This script runs comprehensive tests to validate our discoveries:
1. Quantum weight crossover point (≈1.15)
2. Dimensionality crossover point (D ≥ 20)
3. Convergence behavior with proper training
4. Learning rate sensitivity
5. Batch size effects
6. Matrix dimension scaling

Designed for GPU-accelerated environments (Colab, etc.)
"""

import torch
import time
from datetime import datetime
from pathlib import Path
import argparse

# Import our GPU test modules
from gpu_convergence_testing import GPUConvergenceTester
from gpu_dimensionality_crossover import GPUDimensionalityCrossoverTester
from gpu_batch_size_effects import GPUBatchSizeTester

class GPUMasterTestSuite:
    """Master test suite for all GPU experiments"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.start_time = time.time()
        
        print("🚀 GPU MASTER TEST SUITE")
        print("=" * 60)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 60)
        
        # Create main results directory
        self.results_dir = Path("test_results/gpu_master_suite")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize test modules
        self.convergence_tester = GPUConvergenceTester(device=device)
        self.dimensionality_tester = GPUDimensionalityCrossoverTester(device=device)
        self.batch_size_tester = GPUBatchSizeTester(device=device)
        
    def run_full_test_suite(self):
        """Run the complete test suite"""
        
        print("\n🎯 RUNNING FULL GPU TEST SUITE")
        print("=" * 60)
        
        # Test 1: Convergence Testing
        print("\n📊 TEST 1: Convergence Testing")
        print("-" * 40)
        self.convergence_tester.test_convergence_scenarios()
        
        # Test 2: Dimensionality Crossover
        print("\n📊 TEST 2: Dimensionality Crossover Validation")
        print("-" * 40)
        self.dimensionality_tester.test_dimensionality_crossover()
        self.dimensionality_tester.test_matrix_dimension_scaling()
        
        # Test 3: Batch Size Effects
        print("\n📊 TEST 3: Batch Size Effects")
        print("-" * 40)
        self.batch_size_tester.test_batch_size_scenarios()
        
        # Generate comprehensive summary
        self.generate_master_summary()
        
        print("\n🎉 FULL GPU TEST SUITE COMPLETED!")
        
    def run_convergence_tests_only(self):
        """Run only convergence testing"""
        
        print("\n📊 RUNNING CONVERGENCE TESTS ONLY")
        print("=" * 50)
        
        self.convergence_tester.test_convergence_scenarios()
        
        print("\n✅ Convergence tests completed!")
        
    def run_dimensionality_tests_only(self):
        """Run only dimensionality crossover tests"""
        
        print("\n📊 RUNNING DIMENSIONALITY CROSSOVER TESTS ONLY")
        print("=" * 50)
        
        self.dimensionality_tester.test_dimensionality_crossover()
        self.dimensionality_tester.test_matrix_dimension_scaling()
        
        print("\n✅ Dimensionality crossover tests completed!")
        
    def run_batch_size_tests_only(self):
        """Run only batch size effect tests"""
        
        print("\n📊 RUNNING BATCH SIZE EFFECT TESTS ONLY")
        print("=" * 50)
        
        self.batch_size_tester.test_batch_size_scenarios()
        
        print("\n✅ Batch size effect tests completed!")
        
    def run_quick_validation(self):
        """Run a quick validation of our key discoveries"""
        
        print("\n⚡ RUNNING QUICK VALIDATION")
        print("=" * 40)
        print("Testing our key discoveries with minimal parameters:")
        print("1. Quantum weight crossover (≈1.15)")
        print("2. Dimensionality crossover (D ≥ 20)")
        print("3. Basic convergence behavior")
        
        # Quick quantum weight test
        print("\n🔍 Quick Quantum Weight Crossover Test")
        self.convergence_tester.test_quantum_weight_crossover()
        
        # Quick dimensionality test
        print("\n🔍 Quick Dimensionality Crossover Test")
        self.dimensionality_tester.test_dimensionality_crossover()
        
        print("\n✅ Quick validation completed!")
        
    def generate_master_summary(self):
        """Generate a comprehensive summary of all test results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"master_suite_summary_{timestamp}.txt"
        
        total_time = time.time() - self.start_time
        
        with open(summary_file, 'w') as f:
            f.write("GPU MASTER TEST SUITE - COMPREHENSIVE SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Total Runtime: {total_time:.2f} seconds ({total_time/60:.1f} minutes)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("TESTS EXECUTED:\n")
            f.write("-" * 20 + "\n")
            f.write("1. ✅ Convergence Testing\n")
            f.write("   - Low-dimensional convergence (N=3, D=3)\n")
            f.write("   - High-dimensional convergence (N=16, D=40)\n")
            f.write("   - Learning rate sensitivity\n")
            f.write("   - Quantum weight crossover validation\n\n")
            
            f.write("2. ✅ Dimensionality Crossover Validation\n")
            f.write("   - Test scenarios: D=3, 10, 15, 18, 20, 25, 30, 40, 50\n")
            f.write("   - Matrix dimensions: N=3, 8, 12, 16, 20, 24, 32\n")
            f.write("   - Quantum weights: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0\n\n")
            
            f.write("3. ✅ Batch Size Effects\n")
            f.write("   - Low-dimensional batch sizes: 50, 100, 250, 500, 1000\n")
            f.write("   - High-dimensional batch sizes: 50, 100, 250, 500\n")
            f.write("   - Memory vs performance trade-offs\n\n")
            
            f.write("EXPECTED OUTCOMES:\n")
            f.write("-" * 20 + "\n")
            f.write("1. Quantum Weight Crossover: SGD wins at QW > 1.15\n")
            f.write("2. Dimensionality Crossover: ADAM always wins at D ≥ 20\n")
            f.write("3. Convergence: All tests should reach stable minima\n")
            f.write("4. Learning Rate: Optimal LRs identified for each scenario\n")
            f.write("5. Batch Size: Performance vs memory trade-offs quantified\n\n")
            
            f.write("NEXT STEPS:\n")
            f.write("-" * 15 + "\n")
            f.write("1. Analyze results for validation of our discoveries\n")
            f.write("2. Identify optimal hyperparameters for each problem type\n")
            f.write("3. Document final optimizer selection rules\n")
            f.write("4. Prepare research paper with validated findings\n")
        
        print(f"📝 Master summary saved to: {summary_file}")
        
        # Also save a simple status file
        status_file = self.results_dir / "test_suite_status.txt"
        with open(status_file, 'w') as f:
            f.write(f"Last run: {timestamp}\n")
            f.write(f"Status: COMPLETED\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Runtime: {total_time:.2f}s\n")
        
        print(f"📊 Status file saved to: {status_file}")

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="GPU Master Test Suite for Optimizer Comparison")
    parser.add_argument('--mode', choices=['full', 'convergence', 'dimensionality', 'batch_size', 'quick'], 
                       default='full', help='Test mode to run')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='auto',
                       help='Device to use (auto detects CUDA)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize test suite
    test_suite = GPUMasterTestSuite(device=device)
    
    # Run selected mode
    if args.mode == 'full':
        test_suite.run_full_test_suite()
    elif args.mode == 'convergence':
        test_suite.run_convergence_tests_only()
    elif args.mode == 'dimensionality':
        test_suite.run_dimensionality_tests_only()
    elif args.mode == 'batch_size':
        test_suite.run_batch_size_tests_only()
    elif args.mode == 'quick':
        test_suite.run_quick_validation()
    
    print(f"\n🎯 Test suite completed in mode: {args.mode}")
    print("Check the test_results/gpu_master_suite/ directory for results.")

if __name__ == "__main__":
    main()
