"""Test script for QCML Univariate Regression Trainer."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qgml.learning.specialized.regression import QGMLRegressionTrainer

def create_synthetic_data(n_samples=200, n_features=3, noise=0.1, random_state=42):
    """Create synthetic regression data."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    # Standardize features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X, y, scaler_X, scaler_y

def test_qgml_regression():
    """Test QCML regression on synthetic data."""
    print("=== QCML Univariate Regression Test ===\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("Creating synthetic regression data...")
    X, y, scaler_X, scaler_y = create_synthetic_data(n_samples=150, n_features=3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize QCML model
    print("\nInitializing QCML Regression Trainer...")
    N = 8  # Hilbert space dimension
    D = X_train.shape[1]  # Number of features
    
    model = QCMLRegressionTrainer(
        N=N,
        D=D,
        learning_rate=0.01,
        device=device,
        loss_type='mae'
    )
    
    print(f"Model parameters:")
    print(f"  Hilbert space dimension (N): {N}")
    print(f"  Number of features (D): {D}")
    print(f"  Number of feature operators: {len(model.feature_operators)}")
    print(f"  Target operator shape: {model.target_operator.shape}")
    
    # Train model
    print("\nTraining QCML model...")
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    # Make predictions
    print("\nMaking predictions...")
    model.eval()
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate metrics
    train_mae = torch.mean(torch.abs(train_predictions - y_train)).item()
    test_mae = torch.mean(torch.abs(test_predictions - y_test)).item()
    train_rmse = torch.sqrt(torch.mean((train_predictions - y_train) ** 2)).item()
    test_rmse = torch.sqrt(torch.mean((test_predictions - y_test) ** 2)).item()
    
    print(f"\nFinal Results:")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Analyze quantum states
    print("\n=== Quantum State Analysis ===")
    sample_idx = 0
    x_sample = X_test[sample_idx]
    y_true = y_test[sample_idx].item()
    y_pred = test_predictions[sample_idx].item()
    
    print(f"\nAnalyzing sample {sample_idx}:")
    print(f"Input features: {x_sample.cpu().numpy()}")
    print(f"True target: {y_true:.4f}")
    print(f"Predicted target: {y_pred:.4f}")
    
    # Get quantum state representation
    psi = model.get_quantum_state_representation(x_sample)
    print(f"Quantum state |ψ⟩ shape: {psi.shape}")
    print(f"Quantum state norm: {torch.norm(psi).item():.6f}")
    
    # Get feature operator expectations
    expectations = model.get_feature_operator_expectations(x_sample)
    print(f"Feature operator expectations ⟨ψ|Aₖ|ψ⟩: {expectations.cpu().numpy()}")
    
    # Compare with input features
    print(f"Original input features: {x_sample.cpu().numpy()}")
    print(f"Expectation vs input correlation: {torch.corrcoef(torch.stack([expectations, x_sample]))[0,1].item():.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Plot predictions vs true values
    plot_predictions(y_test.cpu().numpy(), test_predictions.cpu().numpy(), "Test Set")
    
    return model, history

def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE
    axes[0, 1].plot(history['train_mae'], label='Train')
    if 'val_mae' in history:
        axes[0, 1].plot(history['val_mae'], label='Validation')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # RMSE
    axes[1, 0].plot(history['train_rmse'], label='Train')
    if 'val_rmse' in history:
        axes[1, 0].plot(history['val_rmse'], label='Validation')
    axes[1, 0].set_title('Root Mean Squared Error')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # MSE
    axes[1, 1].plot(history['train_mse'], label='Train')
    if 'val_mse' in history:
        axes[1, 1].plot(history['val_mse'], label='Validation')
    axes[1, 1].set_title('Mean Squared Error')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('qcml_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(y_true, y_pred, title="Predictions"):
    """Plot predictions vs true values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title}: Predictions vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate R²
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    r_squared = correlation ** 2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(f'qcml_predictions_{title.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_classical():
    """Compare QCML regression with classical methods."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    print("\n=== Comparison with Classical Methods ===")
    
    # Create data
    X, y, scaler_X, scaler_y = create_synthetic_data(n_samples=200, n_features=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Classical models
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train classical models
    ridge.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    # Classical predictions
    ridge_pred = ridge.predict(X_test)
    rf_pred = rf.predict(X_test)
    
    # Classical metrics
    ridge_mae = np.mean(np.abs(ridge_pred - y_test))
    rf_mae = np.mean(np.abs(rf_pred - y_test))
    ridge_rmse = np.sqrt(np.mean((ridge_pred - y_test) ** 2))
    rf_rmse = np.sqrt(np.mean((rf_pred - y_test) ** 2))
    
    print(f"Ridge Regression - MAE: {ridge_mae:.4f}, RMSE: {ridge_rmse:.4f}")
    print(f"Random Forest - MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}")
    
    # Train QCML for comparison
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train_torch = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32, device=device)
    
    qcml_model = QCMLRegressionTrainer(N=8, D=X_train.shape[1], learning_rate=0.01, device=device)
    qcml_model.fit(X_train_torch, y_train_torch, epochs=300, verbose=False)
    qcml_pred = qcml_model.predict(X_test_torch).cpu().numpy()
    
    qcml_mae = np.mean(np.abs(qcml_pred - y_test))
    qcml_rmse = np.sqrt(np.mean((qcml_pred - y_test) ** 2))
    
    print(f"QCML Regression - MAE: {qcml_mae:.4f}, RMSE: {qcml_rmse:.4f}")

if __name__ == "__main__":
    # Test QCML regression
    model, history = test_qcml_regression()
    
    # Compare with classical methods
    compare_with_classical()
    
    print("\n=== Test completed successfully! ===")
