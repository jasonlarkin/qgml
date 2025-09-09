# Real Test Comparison: JAX vs PyTorch on qgml Test Cases

## 🎯 Test Cases Run

### test_fig1
- **Description**: Fuzzy Sphere (N=3, D=3, noise=0.0, w_qf=0.0)
- **Parameters**: N=3, D=3, w_qf=0.0
- **Training**: 2000 epochs, 2500 points

### test_supp_fig1
- **Description**: Noisy Circle (N=4, D=2, noise=0.1, w_qf=0.8)
- **Parameters**: N=4, D=2, w_qf=0.8
- **Training**: 1000 epochs, 2500 points

### test_supp_fig2
- **Description**: Swiss Roll (N=4, D=3, noise=0.0, w_qf=0.0)
- **Parameters**: N=4, D=3, w_qf=0.0
- **Training**: 10000 epochs, 2500 points

## 📊 Comparison Results

**Total Tests:** 3
**Reconstruction Matches:** 0/3
**Match Rate:** 0.0%

**Average JAX Speedup:** 1.01x
**Min Speedup:** 0.69x
**Max Speedup:** 1.54x

## 🔍 Detailed Results by Test Case

### test_fig1

| Metric | PyTorch | JAX | Difference |
|--------|---------|-----|------------|
| Final Loss | 0.036903 | 0.195932 | 0.159029 |
| Training Time | 26.73s | 38.54s | 0.69x |
| Reconstruction Match | | | ❌ |

### test_supp_fig1

| Metric | PyTorch | JAX | Difference |
|--------|---------|-----|------------|
| Final Loss | 0.191241 | 0.366655 | 0.175414 |
| Training Time | 47.07s | 30.51s | 1.54x |
| Reconstruction Match | | | ❌ |

### test_supp_fig2

| Metric | PyTorch | JAX | Difference |
|--------|---------|-----|------------|
| Final Loss | 0.424383 | 29.691545 | 29.267162 |
| Training Time | 199.73s | 254.25s | 0.79x |
| Reconstruction Match | | | ❌ |

