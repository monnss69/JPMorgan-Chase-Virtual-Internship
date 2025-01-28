import numpy as np
import pandas as pd

def log_likelihood(n, k):
    """
    Calculate log likelihood for a single bucket
    n: total number in bucket
    k: number of defaults in bucket
    """
    if n == 0:
        return 0
    p = k/n
    if p == 0 or p == 1:  # Handle edge cases
        return 1E-10
    return k * np.log(p) + (n-k) * np.log(1-p)

def quantization(bucket_number):
    """
    Find optimal bucket boundaries using dynamic programming to maximize log likelihood
    
    Parameters:
    bucket_number: desired number of buckets
    
    Returns:
    list: optimal boundary points
    """
    # Read and prepare data
    df = pd.read_csv('Loan_Data_CSV.csv')
    x = df['default'].tolist()
    y = df['fico_score'].tolist()
    n = len(x)
    
    # Initialize arrays for cumulative counts
    default = [0] * 851
    total = [0] * 851
    
    # Count defaults and totals for each FICO score
    for i in range(n):
        y[i] = int(y[i])
        default[y[i]-300] += x[i]
        total[y[i]-300] += 1
    
    # Create cumulative sums so that default[i] and total[i] represent the sum of corresponding properties up to i
    for i in range(1, 551):
        default[i] += default[i-1]
        total[i] += total[i-1]
    
    # Initialize dynamic programming array
    # dp[i][j] = [log_likelihood for i buckets up to position j, previous boundary]
    r = bucket_number
    dp = [[[-10**18, 0] for i in range(551)] for j in range(r+1)]
    
    for i in range(r+1):  # For each number of buckets
        for j in range(551):  # For each possible ending position
            if i == 0:  # Base case
                dp[i][j][0] = 0
            else:
                for k in range(j):  # Try each possible starting position
                    if total[j] == total[k]:  # Skip if no new records because likelihood is 0
                        continue
                    
                    if i == 1:  # First bucket
                        dp[i][j][0] = log_likelihood(total[j], default[j])
                    else:  # Multiple buckets
                        # Check if new configuration gives better likelihood
                        new_ll = dp[i-1][k][0] + log_likelihood(total[j]-total[k], default[j]-default[k])
                        if dp[i][j][0] < new_ll:
                            dp[i][j][0] = new_ll
                            dp[i][j][1] = k  # Store boundary point
    
    # Extract optimal boundaries
    boundaries = []
    k = 550
    while r >= 0:
        boundaries.append(k + 300)  # Convert back to FICO score
        k = dp[r][k][1]
        r -= 1
    
    return sorted(boundaries)  # Return sorted boundaries

# Test
r = 10  # Number of buckets
optimal_boundaries = quantization(r)
print(f"Optimal boundaries for {r} buckets: {optimal_boundaries}")