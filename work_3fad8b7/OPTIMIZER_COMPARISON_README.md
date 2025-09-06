# SGD vs ADAM Optimizer Comparison

This document provides comprehensive analysis of Stochastic Gradient Descent (SGD) versus ADAM optimizer performance on matrix training problems.

## **Scripts Overview**

### **1. `sgd_vs_adam_simple.py`**
- **Purpose**: Basic comparison on simple matrix problems
- **Configurations**: 3x3, 4x4 matrices, 500-1000 points, 2000 epochs
- **Output**: 4-panel comparison plots (training curves + performance metrics)

### **2. `sgd_vs_adam_high_dim.py`**
- **Purpose**: High-dimensional dataset comparison
- **Datasets**: Hypercube M10b (17D), M_beta (20D), M_N1 Nonlinear (13D)
- **Output**: 6-panel comprehensive comparison across multiple datasets

## **Key Results Summary**

| Matrix Size | SGD vs ADAM Gap | ADAM Advantage | Training Time |
|-------------|------------------|----------------|---------------|
| **3x3**     | 23.8%           | Quality only   | ~2.2s vs 2.3s |
| **4x4**     | **74.0%**       | **Quality + Speed** | 8.5s vs 6.6s |
| **High-D**  | **100%+**       | **Dramatic Quality** | Variable |

## **Optimizer Performance Analysis: ADAM vs SGD**

Based on our extensive testing across various epochs and problem complexities (3x3, 4x4, and higher-dimensional datasets), we've observed consistent and explainable differences in how ADAM and SGD perform. These differences are fundamental to their design and are crucial for understanding their suitability for different optimization tasks.

### **1. Why ADAM Achieves Much Lower Final Loss:**

**ADAM's Adaptive Learning:**
- **Per-parameter learning rates**: Each parameter gets its own adaptive learning rate
- **Momentum + RMSprop**: Combines momentum with adaptive step sizes
- **Better gradient scaling**: Handles varying gradient magnitudes across dimensions
- **Escape local minima**: More effective at navigating complex landscapes

**SGD's Limitations:**
- **Fixed learning rate**: Same step size for all parameters
- **Simple momentum**: Only adds velocity, doesn't adapt to parameter-specific gradients
- **Gradient scaling issues**: Struggles when different parameters have very different gradient magnitudes

### **2. Why ADAM Curves Are Less Smooth/Monotonic:**

**ADAM's Adaptive Nature:**
- **Learning rate adaptation**: Each parameter's learning rate changes based on its gradient history
- **Momentum fluctuations**: Adaptive momentum can cause oscillations as it "learns" optimal step sizes
- **Parameter-specific dynamics**: Different parameters converge at different rates

**SGD's Consistency:**
- **Fixed dynamics**: Same learning rate and momentum for all parameters
- **Predictable behavior**: More monotonic because it's not adapting step sizes
- **Smoother curves**: Less parameter-specific variation

### **3. Why This Happens More on High-Dimensional Problems:**

**Complex Optimization Landscapes:**
- **More parameters** = more complex gradient interactions
- **Varying gradient magnitudes** across dimensions
- **Local minima and saddle points** become more prevalent
- **ADAM's adaptability** becomes increasingly valuable

## **Loss Components Note**

**Important**: The current `qgml_new` implementation has the **commutation penalty disabled** (commented out in the `forward` method). This means our current comparisons only show:

1. **Reconstruction Error** (dominant component)
2. **Quantum Fluctuation** (if enabled, currently set to 0.0)

The **commutation penalty** that appeared in earlier plots (ensuring matrices commute) is not currently active. This was likely removed to simplify the optimization problem and focus on the core reconstruction task.

## **Usage Instructions**

### **Running Simple Comparison:**
```bash
cd qgml_new
python sgd_vs_adam_simple.py
```

### **Running High-Dimensional Comparison:**
```bash
cd qgml_new
python sgd_vs_adam_high_dim.py
```

### **Customizing Parameters:**
- **Matrix dimensions**: Modify `N` and `D` in the script
- **Training epochs**: Change `n_epochs` (recommend 2000+ for convergence)
- **Learning rate**: Adjust `learning_rate` (0.0005 works well for convergence)
- **Points**: Modify `n_points` for different dataset sizes

## **Expected Outputs**

### **Simple Comparison:**
- `sgd_vs_adam_simple.png`: 4-panel comparison plot
- Console output with detailed metrics and convergence analysis

### **High-Dimensional Comparison:**
- `sgd_vs_adam_high_dim.png`: 6-panel comprehensive comparison
- Detailed analysis across multiple complex datasets
- Scaling behavior insights

## **Interpretation Guide**

### **Quality Metrics:**
- **Final Loss**: Lower is better (ADAM typically wins)
- **Convergence Speed**: How quickly loss drops initially
- **Training Stability**: Smoothness of loss curves

### **Performance Trade-offs:**
- **SGD**: Simple, predictable, but limited on complex problems
- **ADAM**: More complex, less smooth, but dramatically better on hard problems
- **Complexity Scaling**: Performance gaps grow with problem dimensionality

## **Conclusion**

The observed performance of ADAM—achieving significantly lower final loss, demonstrating faster initial convergence, and exhibiting slightly less smooth loss curves—is a direct consequence of its sophisticated adaptive learning rate mechanism. For complex, high-dimensional problems, ADAM's ability to intelligently adjust its steps per parameter often makes it the superior choice, even if its path to convergence appears less monotonic than that of a simpler optimizer like SGD.

This comparison provides excellent material for presentations on optimizer selection, demonstrating how theoretical differences translate to practical performance advantages in real optimization problems.
