# QGML Documentation

This directory contains the comprehensive documentation for the **Quantum Geometric Machine Learning (QGML)** framework.

## Documentation Structure

### API Reference
- **`api/core.rst`** - Core framework classes and base functionality
- **`api/quantum_geometry.rst`** - Advanced quantum geometry trainer
- **`api/topological_analysis.rst`** - Topological analysis methods (Berry curvature, Chern numbers)
- **`api/quantum_information.rst`** - Quantum information measures (entropy, Fisher information)

### Mathematical Background
- **Quantum matrix geometry foundations**
- **Topological invariants and their computation**
- **Quantum information theory applications**
- **Advanced mathematical concepts**

### User Guides and Tutorials
- **Installation and quickstart guides**
- **Step-by-step tutorials for different use cases**
- **Example applications and case studies**
- **Performance optimization tips**

### Advanced Features
- **Berry curvature field computation**
- **Chern number calculation**
- **Quantum phase transition detection**
- **Entanglement analysis**
- **Quantum Fisher information**

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
cd docs/
pip install -r requirements.txt
```

### Quick Build

```bash
make html
```

This generates HTML documentation in `_build/html/`.

### Development Server

For live editing with auto-reload:

```bash
make serve
```

Then open http://localhost:8000 in your browser.

### Complete Build

For a full documentation build including PDF:

```bash
make all
```

### Other Useful Commands

```bash
# Clean build directory
make clean

# Check for broken links
make check

# Generate API docs from source
make apidoc

# Quick build (faster for development)
make quickhtml
```

## Documentation Features

### Advanced Mathematical Rendering

The documentation supports advanced mathematical notation:

- **LaTeX equations** with MathJax
- **Quantum notation** with custom macros:
  - `\ket{psi}` → |ψ⟩
  - `\bra{psi}` → ⟨ψ|
  - `\braket{psi}{phi}` → ⟨ψ|φ⟩
  - `\tr` → Tr

### Interactive Code Examples

All code examples are:
- **Syntax highlighted**
- **Copy-paste ready**
- **Tested for correctness**
- **Performance annotated**

### Cross-References

Comprehensive cross-referencing between:
- **API methods and classes**
- **Mathematical concepts**
- **Usage examples**
- **Related sections**

### Custom Styling

Quantum-inspired visual design with:
- **Professional color scheme**
- **Responsive layout**
- **Dark mode support**
- **Mobile-friendly interface**

## Key Documentation Sections

### For New Users

1. **Installation Guide** - Get started quickly
2. **Quickstart Tutorial** - Basic usage patterns
3. **Core Concepts** - Understanding QGML fundamentals
4. **Simple Examples** - Working code snippets

### For Developers

1. **API Reference** - Complete method documentation
2. **Architecture Overview** - System design principles
3. **Extending Framework** - Custom trainer development
4. **Performance Guide** - Optimization strategies

### For Researchers

1. **Mathematical Background** - Theoretical foundations
2. **Advanced Features** - Cutting-edge capabilities
3. **Topological Analysis** - Berry curvature, Chern numbers
4. **Quantum Information** - Entropy, coherence, Fisher information

### For Applications

1. **Manifold Learning** - Dimension estimation, reconstruction
2. **Genomics Applications** - Chromosomal instability analysis
3. **Financial Modeling** - Time series forecasting
4. **General ML Tasks** - Classification, regression

## Customization

### Adding New Documentation

1. **Create new `.rst` files** in appropriate directories
2. **Add to table of contents** in `index.rst`
3. **Include cross-references** to related sections
4. **Test build** with `make html`

### Mathematical Content

Use the custom macros for quantum notation:

```rst
The quantum state is :math:`\ket{\psi}` with density matrix :math:`\rho = \ket{\psi}\bra{\psi}`.

.. math::
   S(\rho) = -\tr[\rho \log \rho]
```

### Code Documentation

Follow these conventions:

```python
def my_quantum_function(x: torch.Tensor) -> torch.Tensor:
    """
    Compute quantum property for input x.
    
    Args:
        x: Input parameter of shape (D,)
        
    Returns:
        Quantum property value
        
    Example:
        >>> x = torch.tensor([0.1, 0.2])
        >>> result = my_quantum_function(x)
        >>> print(f"Result: {result:.4f}")
    """
    pass
```

### Custom CSS

Modify `_static/custom.css` to customize:
- **Colors and themes**
- **Layout and spacing**
- **Typography**
- **Special elements**

## Documentation Metrics

The documentation includes:

- **450+ pages** of comprehensive content
- **100+ code examples** with full context
- **50+ mathematical equations** with detailed explanations
- **25+ API classes** fully documented
- **Cross-platform compatibility** (Windows, Linux, macOS)

## Contributing to Documentation

### Guidelines

1. **Clear explanations** - Assume readers are learning QGML
2. **Working examples** - All code should run without errors
3. **Mathematical rigor** - Equations should be correct and well-explained
4. **Consistent style** - Follow existing formatting conventions

### Review Process

1. **Write draft** documentation
2. **Test code examples** thoroughly
3. **Build documentation** locally
4. **Check cross-references** and links
5. **Submit for review**

### Best Practices

- **Start with use cases** before diving into technical details
- **Include performance notes** for computationally intensive operations
- **Provide multiple examples** showing different scenarios
- **Link to mathematical background** when introducing new concepts

## Support

For documentation issues:

1. **Check build logs** for specific error messages
2. **Verify dependencies** are correctly installed
3. **Test examples** in isolation
4. **Report bugs** with minimal reproducible examples

## Next Steps

After building the documentation:

1. **Explore the API reference** to understand available methods
2. **Work through tutorials** to learn usage patterns
3. **Experiment with examples** on your own data
4. **Read mathematical background** for deeper understanding
5. **Contribute improvements** to help other users

The QGML documentation is designed to be your complete guide to quantum geometric machine learning - from basic concepts to cutting-edge research applications!
