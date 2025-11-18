"""
Migration validation tests for QGML.

Tests to validate that the migration from QCML to QGML was successful.
"""

import pytest
import importlib.util
import sys
from pathlib import Path


class TestMigrationValidation:
    """Test that the migration from QCML to QGML was successful."""
    
    def test_basic_imports(self, setup_qgml_backend):
        """Test that all QGML modules can be imported."""
        modules_to_test = [
            'qgml',
            'qgml.core',
            'qgml.backends', 
            'qgml.geometry',
            'qgml.topology',
            'qgml.learning',
            'qgml.quantum_computing',
            'qgml.information',
            'qgml.utils'
        ]
        
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Failed to import {module_name}"
            except ImportError as e:
                pytest.fail(f"Could not import {module_name}: {e}")
    
    def test_core_files_exist(self):
        """Test that all expected core files exist."""
        qgml_root = Path(__file__).parent.parent
        
        expected_files = [
            'qgml/core/base_quantum_trainer.py',
            'qgml/learning/supervised_trainer.py',
            'qgml/learning/unsupervised_trainer.py',
            'qgml/geometry/quantum_geometry_trainer.py',
            'qgml/topology/topological_analyzer.py',
            'qgml/information/quantum_information.py',
            'qgml/backends/pytorch_backend.py',
            'qgml/backends/jax_backend.py',
            'qgml/learning/specialized/genomics.py',
            'qgml/learning/specialized/regression.py',
            'qgml/quantum_computing/circuit_implementations.py'
        ]
        
        for file_path in expected_files:
            full_path = qgml_root / file_path
            assert full_path.exists(), f"Expected file missing: {file_path}"
            assert full_path.stat().st_size > 0, f"File is empty: {file_path}"
    
    @pytest.mark.backend
    def test_backend_switching(self, setup_qgml_backend):
        """Test that backend switching functionality works."""
        import qgml
        
        # Test getting current backend
        current = qgml.get_backend()
        assert current is not None
        
        # Test switching to PyTorch
        qgml.set_backend("pytorch")
        pytorch_backend = qgml.get_backend()
        assert str(pytorch_backend).lower() == "pytorch" or "pytorch" in str(pytorch_backend).lower()
        
        # Test switching to JAX (if available)
        try:
            qgml.set_backend("jax")
            jax_backend = qgml.get_backend()
            assert str(jax_backend).lower() == "jax" or "jax" in str(jax_backend).lower()
        except Exception:
            # JAX backend might not be available, that's OK
            pass
        
        # Switch back to PyTorch
        qgml.set_backend("pytorch")
    
    def test_jax_pytorch_preservation(self):
        """Test that JAX/PyTorch comparison files were preserved."""
        qgml_root = Path(__file__).parent.parent
        
        jax_pytorch_files = [
            'experiments/backend_comparison/scaling_analysis.py',
            'experiments/backend_comparison/performance_comparison.py',
            'experiments/backend_comparison/gpu_benchmarks.py',
            'experiments/backend_comparison/optimizer_comparison.py'
        ]
        
        for file_path in jax_pytorch_files:
            full_path = qgml_root / file_path
            assert full_path.exists(), f"JAX/PyTorch file missing: {file_path}"
            # Should have substantial content
            assert full_path.stat().st_size > 1000, f"JAX/PyTorch file too small: {file_path}"
    
    def test_experiment_structure(self):
        """Test that experiment directories are properly organized."""
        qgml_root = Path(__file__).parent.parent
        
        expected_dirs = [
            'experiments/integration_validation',
            'experiments/backend_comparison',
            'experiments/benchmarks',
            'experiments/applications',
            'experiments/quantum_hardware'
        ]
        
        for dir_path in expected_dirs:
            full_path = qgml_root / dir_path
            assert full_path.exists(), f"Experiment directory missing: {dir_path}"
            assert full_path.is_dir(), f"Expected directory, found file: {dir_path}"
            
            # Should contain at least one Python file
            py_files = list(full_path.glob('*.py'))
            assert len(py_files) > 0, f"No Python files in experiment directory: {dir_path}"
    
    def test_setup_files(self):
        """Test that setup and configuration files exist."""
        qgml_root = Path(__file__).parent.parent
        
        setup_files = [
            'README.md',
            'setup.py', 
            'pyproject.toml',
            'requirements.txt'
        ]
        
        for file_name in setup_files:
            file_path = qgml_root / file_name
            assert file_path.exists(), f"Setup file missing: {file_name}"
            assert file_path.stat().st_size > 0, f"Setup file empty: {file_name}"
    
    def test_qgml_branding(self):
        """Test that QGML branding is properly updated."""
        qgml_root = Path(__file__).parent.parent
        readme_path = qgml_root / 'README.md'
        
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                content = f.read()
            
            # Should contain QGML branding
            assert 'QGML' in content or 'Quantum Geometric Machine Learning' in content
            
            # Should not contain old QCML branding
            assert 'Quantum Cognition Machine Learning' not in content
    
    def test_python_syntax(self):
        """Test that key files have valid Python syntax."""
        qgml_root = Path(__file__).parent.parent
        
        key_files = [
            'qgml/__init__.py',
            'qgml/core/__init__.py', 
            'qgml/backends/__init__.py'
        ]
        
        for file_path in key_files:
            full_path = qgml_root / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    code = f.read()
                
                try:
                    compile(code, str(full_path), 'exec')
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {file_path}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
