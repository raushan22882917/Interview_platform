def max_retrieve(values):
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    
    # Initialize two variables for space optimization
    prev2 = values[0]  # dp[i-2]
    prev1 = max(values[0], values[1])  # dp[i-1]
    
    # Iterate through the array starting from the third section
    for i in range(2, len(values)):
        current = max(prev1, values[i] + prev2)
        prev2 = prev1
        prev1 = current
    
    return prev1

# Example usage:
values = [1, 2, 3, 1]
print(max_retrieve(values))  # Output: 15
