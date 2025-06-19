# pip install cython
# pip install git+https://github.com/coreylynch/pyFM




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# First, install PyFM if you haven't already:
# pip install git+https://github.com/coreylynch/pyFM

from pyfm import pylibfm

# Generate synthetic movie rating data to demonstrate FM
np.random.seed(42)
n_users = 1000
n_movies = 500
n_ratings = 5000

# Create user and movie features
users = np.random.randint(0, n_users, n_ratings)
movies = np.random.randint(0, n_movies, n_ratings)

# Create some categorical features
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']
movie_genres = np.random.choice(genres, n_ratings)
user_ages = np.random.choice(['18-25', '26-35', '36-45', '46+'], n_ratings)

# Generate ratings (1-5) with some pattern based on user-movie interactions
base_ratings = np.random.normal(3, 1, n_ratings)
# Add some interaction effects
interaction_effect = np.sin(users * 0.01) * np.cos(movies * 0.01) * 2
ratings = np.clip(base_ratings + interaction_effect, 1, 5)

# Create DataFrame
data = pd.DataFrame({
    'user_id': users,
    'movie_id': movies,
    'genre': movie_genres,
    'user_age': user_ages,
    'rating': ratings
})

print("Sample of the data:")
print(data.head())
print(f"\nDataset shape: {data.shape}")

# Prepare features for FM
# FM works well with sparse categorical data, so we'll create one-hot encodings

def prepare_fm_data(df):
    """Prepare data for factorization machine"""
    features = []
    feature_names = []
    
    # One-hot encode categorical features
    for col in ['user_id', 'movie_id', 'genre', 'user_age']:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col])
        
        # Create one-hot encoding manually for sparse matrix
        n_categories = len(le.classes_)
        one_hot = np.zeros((len(df), n_categories))
        one_hot[np.arange(len(df)), encoded] = 1
        
        features.append(one_hot)
        feature_names.extend([f"{col}_{cls}" for cls in le.classes_])
    
    # Combine all features
    X = np.hstack(features)
    return csr_matrix(X), feature_names

# Prepare the data
X, feature_names = prepare_fm_data(data)
y = data['rating'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.3f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train Factorization Machine
print("\nTraining Factorization Machine...")

# Initialize FM model
fm = pylibfm.FM(
    num_factors=10,      # Number of latent factors
    num_iter=100,        # Number of iterations
    verbose=True,        # Print training progress
    task="regression",   # Regression task
    initial_learning_rate=0.01,
    learning_rate_schedule="optimal"
)

# Fit the model
fm.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred_train = fm.predict(X_train)
y_pred_test = fm.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nModel Performance:")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title(f'Training Set\nRMSE: {train_rmse:.4f}, R²: {train_r2:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title(f'Test Set\nRMSE: {test_rmse:.4f}, R²: {test_r2:.4f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate the power of FM: it can capture interactions
print("\nExample predictions for a specific user-movie combination:")
print("FM can capture complex interactions between users, movies, genres, and demographics")

# Show some sample predictions
sample_indices = np.random.choice(len(y_test), 5, replace=False)
for i, idx in enumerate(sample_indices):
    actual = y_test[idx]
    predicted = y_pred_test[idx]
    print(f"Sample {i+1}: Actual={actual:.2f}, Predicted={predicted:.2f}, Error={abs(actual-predicted):.2f}")

print("\nKey advantages of Factorization Machines:")
print("1. Handles sparse data efficiently")
print("2. Captures feature interactions automatically")
print("3. Works well with categorical features")
print("4. Scales to large datasets")
print("5. No need for manual feature engineering of interactions")