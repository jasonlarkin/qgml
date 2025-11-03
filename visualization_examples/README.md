# QGML Visualization Examples

## Overview

This directory contains comprehensive examples demonstrating the visualization capabilities of the QGML framework across different use cases and analysis types.

## Example Categories

### Training Visualization Examples
**Location**: `visualization_examples/training/`

Examples of training progress and convergence visualization:

- **Basic Training Plots**: Loss curves, accuracy plots
- **Convergence Analysis**: Training stability and convergence
- **Hyperparameter Sensitivity**: Parameter optimization visualization
- **Training Diagnostics**: Error analysis and debugging

### Quantum State Visualization Examples
**Location**: `visualization_examples/quantum_states/`

Examples of quantum state property visualization:

- **State Amplitude Plots**: Quantum state component visualization
- **Phase Information**: Quantum phase analysis
- **Expectation Values**: Operator expectation value plots
- **State Evolution**: Time evolution of quantum states

### Geometric Visualization Examples
**Location**: `visualization_examples/geometry/`

Examples of geometric and topological visualization:

- **Manifold Structure**: Data manifold visualization
- **Berry Curvature Fields**: Topological field visualization
- **Quantum Metric Tensor**: Geometric metric visualization
- **Topological Invariants**: Chern numbers and winding numbers

### Performance Visualization Examples
**Location**: `visualization_examples/performance/`

Examples of performance analysis visualization:

- **Backend Comparison**: PyTorch vs JAX performance
- **Memory Usage**: Memory efficiency analysis
- **Scalability**: Performance scaling analysis
- **Benchmark Results**: Comprehensive benchmark visualization

## Running Examples

### Basic Execution
```bash
# Run specific example
python visualization_examples/training/basic_training_plots.py

# Run all examples in a category
python -m visualization_examples.training

# Run with custom parameters
python visualization_examples/performance/backend_comparison.py --N 16 --D 5
```

### Example Configuration
```python
# visualization_examples/config.py
EXAMPLE_CONFIG = {
    'training': {
        'n_epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001
    },
    'quantum_states': {
        'n_states': 10,
        'resolution': 100
    },
    'geometry': {
        'grid_size': 50,
        'contour_levels': 20
    }
}
```

## Example Documentation

### Training Visualization Examples

#### Basic Training Plots
**File**: `visualization_examples/training/basic_training_plots.py`

**Purpose**: Demonstrate basic training progress visualization.

**Key Features**:
- Loss curve plotting
- Accuracy visualization
- Training/validation comparison
- Convergence analysis

**Example Output**:
```
Generated training plots:
- loss_curves.png: Training and validation loss
- accuracy_plot.png: Training and validation accuracy
- convergence_analysis.png: Convergence rate analysis
```

#### Hyperparameter Sensitivity
**File**: `visualization_examples/training/hyperparameter_sensitivity.py`

**Purpose**: Visualize hyperparameter sensitivity analysis.

**Key Features**:
- Learning rate sensitivity
- Batch size optimization
- Regularization parameter analysis
- Grid search visualization

### Quantum State Visualization Examples

#### State Amplitude Visualization
**File**: `visualization_examples/quantum_states/state_amplitudes.py`

**Purpose**: Visualize quantum state amplitude distributions.

**Key Features**:
- State component plots
- Amplitude distribution analysis
- State comparison visualization
- Probability density plots

#### Expectation Value Analysis
**File**: `visualization_examples/quantum_states/expectation_values.py`

**Purpose**: Analyze and visualize expectation values.

**Key Features**:
- Operator expectation value plots
- Expectation value evolution
- Statistical analysis
- Correlation analysis

### Geometric Visualization Examples

#### Manifold Structure Visualization
**File**: `visualization_examples/geometry/manifold_structure.py`

**Purpose**: Visualize data manifold structure and properties.

**Key Features**:
- 2D/3D manifold visualization
- Manifold embedding analysis
- Distance matrix visualization
- Manifold reconstruction quality

#### Berry Curvature Fields
**File**: `visualization_examples/geometry/berry_curvature.py`

**Purpose**: Visualize Berry curvature fields and topological properties.

**Key Features**:
- 2D Berry curvature field plots
- Contour and heatmap visualization
- Topological charge analysis
- Phase transition visualization

### Performance Visualization Examples

#### Backend Comparison
**File**: `visualization_examples/performance/backend_comparison.py`

**Purpose**: Compare performance across different backends.

**Key Features**:
- Training time comparison
- Memory usage analysis
- Throughput comparison
- Numerical accuracy validation

#### Scalability Analysis
**File**: `visualization_examples/performance/scalability.py`

**Purpose**: Analyze performance scaling with problem size.

**Key Features**:
- Time complexity analysis
- Memory scaling analysis
- Batch size optimization
- Model size scaling

## Interactive Examples

### Jupyter Notebook Examples
**Location**: `visualization_examples/notebooks/`

Interactive Jupyter notebooks for exploration:

- **Quantum State Explorer**: Interactive quantum state analysis
- **Geometric Analyzer**: Interactive geometric property exploration
- **Performance Dashboard**: Interactive performance analysis
- **Training Monitor**: Real-time training monitoring

### Web-based Examples
**Location**: `visualization_examples/web/`

Web-based interactive visualizations:

- **Quantum State Visualizer**: Web-based state exploration
- **Geometric Analyzer**: Web-based geometric analysis
- **Performance Dashboard**: Web-based performance monitoring

## Custom Visualization Examples

### Custom Plot Creation
**File**: `visualization_examples/custom/custom_plots.py`

**Purpose**: Demonstrate custom visualization creation.

**Key Features**:
- Custom plot styling
- Advanced matplotlib usage
- Interactive plot creation
- Export to multiple formats

### Animation Examples
**File**: `visualization_examples/animations/state_evolution.py`

**Purpose**: Create animated visualizations.

**Key Features**:
- Quantum state evolution animation
- Training progress animation
- Parameter space exploration animation
- Topological evolution animation

## Results and Outputs

### Generated Visualizations
**Location**: `visualization_examples/outputs/`

All example outputs are saved here:

- **PNG Files**: High-resolution static plots
- **PDF Files**: Vector graphics for publications
- **SVG Files**: Scalable vector graphics
- **HTML Files**: Interactive web visualizations
- **MP4 Files**: Animation outputs

### Output Organization
```
visualization_examples/outputs/
├── training/
│   ├── loss_curves.png
│   ├── accuracy_plots.png
│   └── convergence_analysis.png
├── quantum_states/
│   ├── state_amplitudes.png
│   ├── expectation_values.png
│   └── phase_analysis.png
├── geometry/
│   ├── manifold_structure.png
│   ├── berry_curvature.png
│   └── topological_analysis.png
└── performance/
    ├── backend_comparison.png
    ├── memory_usage.png
    └── scalability_analysis.png
```

## Configuration and Customization

### Plot Styling
```python
# visualization_examples/style_config.py
PLOT_STYLE = {
    'figure_size': (10, 8),
    'dpi': 300,
    'font_size': 12,
    'color_scheme': 'quantum',
    'line_width': 2,
    'marker_size': 6
}
```

### Output Configuration
```python
# visualization_examples/output_config.py
OUTPUT_CONFIG = {
    'formats': ['png', 'pdf', 'svg'],
    'dpi': 300,
    'bbox_inches': 'tight',
    'transparent': False,
    'facecolor': 'white'
}
```

## Best Practices

### Visualization Design
1. **Clear Labels**: Use descriptive titles and axis labels
2. **Consistent Styling**: Maintain consistent colors and fonts
3. **Appropriate Scales**: Choose appropriate axis scales
4. **Legend Usage**: Include legends for multi-line plots

### Performance Optimization
1. **Sampling**: Use sampling for large datasets
2. **Batch Processing**: Process data in batches
3. **Memory Management**: Clear unused variables
4. **Output Optimization**: Use appropriate output formats

### Accessibility
1. **Color Blindness**: Use color-blind friendly palettes
2. **High Contrast**: Ensure sufficient contrast
3. **Text Size**: Use readable font sizes
4. **Alternative Formats**: Provide multiple output formats

## Contributing

### Adding New Examples
1. **Create Example File**: Follow naming convention
2. **Add Documentation**: Include purpose and usage
3. **Test Example**: Ensure example runs correctly
4. **Update Documentation**: Update this README

### Example Guidelines
1. **Reproducibility**: Use fixed random seeds
2. **Documentation**: Document all parameters
3. **Error Handling**: Include error handling
4. **Performance**: Optimize for reasonable execution time

## Troubleshooting

### Common Issues
- **Memory Issues**: Reduce dataset size or use sampling
- **Plot Quality**: Increase DPI or use vector formats
- **Performance Issues**: Use efficient plotting libraries
- **Import Issues**: Ensure all dependencies are installed

### Debug Mode
```python
# Enable debug mode for examples
import logging
logging.basicConfig(level=logging.DEBUG)

# Run example with debug output
python visualization_examples/training/basic_training_plots.py --debug
```

## References

- **QGML Framework**: Main documentation
- **Matplotlib**: https://matplotlib.org/
- **Seaborn**: https://seaborn.pydata.org/
- **Plotly**: https://plotly.com/python/
- **Bokeh**: https://bokeh.org/
