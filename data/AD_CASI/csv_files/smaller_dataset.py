import pandas as pd

# Load the dataset
file_path = 'test.csv'
data = pd.read_csv(file_path)

# Randomly sample 10% of the data
sampled_data = data.sample(frac=0.7, random_state=42)  # 'frac' specifies the fraction of the data to sample

# Save the sampled data to a new CSV file
sampled_data.to_csv('test_small_70.csv', index=False)  # 'index=False' ensures the index is not written to the new file

print("Sampled data saved")