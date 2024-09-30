import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your data into a Pandas DataFrame (replace with your data source)
# Assumes you have a CSV file or similar source containing the data
# Features: 'pupil_dilation', 'point_of_gaze', 'blinks', 'eye_movement'
# Target: 'focus_loss_score'
data = pd.read_csv('student_focus_data.csv')

# Define features (X) and target (y)
X = data[['pupil_dilation', 'point_of_gaze', 'blinks', 'eye_movement']]
y = data['focus_loss_score']

# Number of assets (features)
num_features = X.shape[1]

# Calculate the covariance matrix for the features
cov_matrix = np.cov(X.T)

# Mean focus loss score (expected return of each "asset")
mean_focus_loss = X.mean()

# Number of portfolios to simulate
num_portfolios = 10000

# Arrays to store results
results = np.zeros((num_portfolios, 3))
weights_record = []

# Monte Carlo simulation of portfolios
for i in range(num_portfolios):
    # Random weights for each feature (sum to 1)
    weights = np.random.random(num_features)
    weights /= np.sum(weights)
    
    # Portfolio expected return (weighted sum of focus loss scores)
    portfolio_return = np.dot(weights, mean_focus_loss)
    
    # Portfolio risk (variance) using the covariance matrix
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Store the results: risk, return, and weights
    results[i, 0] = portfolio_stddev  # Risk
    results[i, 1] = portfolio_return  # Return (focus loss score)
    results[i, 2] = results[i, 1] / results[i, 0]  # Sharpe ratio (focus risk-adjusted performance)
    
    # Save the weights for this portfolio
    weights_record.append(weights)

# Convert results into a DataFrame
results_df = pd.DataFrame(results, columns=['Risk', 'FocusLossScore', 'SharpeRatio'])

# Identify the portfolio with the maximum Sharpe ratio
max_sharpe_idx = results_df['SharpeRatio'].idxmax()
optimal_weights = weights_record[max_sharpe_idx]
max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]

# Identify the portfolio with the minimum risk
min_risk_idx = results_df['Risk'].idxmin()
min_risk_portfolio = results_df.iloc[min_risk_idx]

# Print the optimal results
print(f"Portfolio with Maximum Sharpe Ratio:")
print(f"Risk (Std Dev): {max_sharpe_portfolio['Risk']}")
print(f"Focus Loss Score: {max_sharpe_portfolio['FocusLossScore']}")
print(f"Optimal Weights: {optimal_weights}")

print(f"\nPortfolio with Minimum Risk:")
print(f"Risk (Std Dev): {min_risk_portfolio['Risk']}")
print(f"Focus Loss Score: {min_risk_portfolio['FocusLossScore']}")

# Plot the portfolios
plt.scatter(results_df['Risk'], results_df['FocusLossScore'], c=results_df['SharpeRatio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_portfolio['Risk'], max_sharpe_portfolio['FocusLossScore'], color='red', marker='*', s=100)  # Max Sharpe
plt.scatter(min_risk_portfolio['Risk'], min_risk_portfolio['FocusLossScore'], color='blue', marker='*', s=100)  # Min Risk
plt.title('Markowitz Simulation: Risk vs Focus Loss Score')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Focus Loss Score')
plt.show()

# Determine the threshold for "failing a test" based on a target focus loss score
fail_threshold = 7.0  # example threshold

# Find the portfolio closest to the threshold
closest_portfolio_idx = (results_df['FocusLossScore'] - fail_threshold).abs().idxmin()
closest_portfolio = results_df.iloc[closest_portfolio_idx]
closest_weights = weights_record[closest_portfolio_idx]

print(f"\nOptimal Portfolio for a Focus Loss Score close to the failing threshold ({fail_threshold}):")
print(f"Risk (Std Dev): {closest_portfolio['Risk']}")
print(f"Focus Loss Score: {closest_portfolio['FocusLossScore']}")
print(f"Optimal Weights: {closest_weights}")
