import pandas as pd
import numpy as np
import os

# Load the preprocessed dataset
data_path = "Data/processed_dataset.csv"
df = pd.read_csv(data_path)

# Label mapping (if needed): 0=Bitumin, 1=Block, 2=Concrete, 3=Kankar, 4=SpeedBreaker
KANKAR_LABEL = 3

# Filter Kankar road samples
kankar_df = df[df['Label'] == KANKAR_LABEL]

# How many synthetic samples to generate (you can change this)
augmentation_factor = 3  # Create 3x more samples
augmented_data = []

def augment_sample(row):
    # Extract sensor values (X_Acc to Z_Gyro)
    features = row[3:].values.astype(float)

    # Add Gaussian noise
    noisy = features + np.random.normal(0, 0.05, size=features.shape)

    # Apply small scaling
    scale = np.random.uniform(0.9, 1.1)
    scaled = noisy * scale

    # Return augmented row with the same label
    return np.concatenate([row[:3].values, scaled, [KANKAR_LABEL]])

# Generate synthetic data
for _ in range(augmentation_factor):
    for _, row in kankar_df.iterrows():
        augmented_row = augment_sample(row)
        augmented_data.append(augmented_row)

# Convert to DataFrame
augmented_df = pd.DataFrame(augmented_data, columns=df.columns)

# Combine with original data
combined_df = pd.concat([df, augmented_df], ignore_index=True)

# Shuffle the dataset
combined_df = combined_df.sample(frac=1.0).reset_index(drop=True)

# Save the augmented dataset
combined_df.to_csv("augmented_dataset.csv", index=False)
print(f"✅ Augmentation complete! Total samples: {len(combined_df)}")
