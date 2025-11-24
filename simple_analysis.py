# SIMPLE CANCER BIOMARKER ANALYSIS
print("🚀 Starting Cancer Biomarker Analysis")
import pandas as pd
import numpy as np
import os

print("✅ Libraries imported")

# Create output directory
os.makedirs('output', exist_ok=True)

# Create simple biomarker data
data = pd.DataFrame({
    'gene': ['BRCA1', 'TP53', 'EGFR', 'HER2', 'KRAS'],
    'cancer_expression': [15.2, 22.1, 18.5, 25.7, 12.3],
    'normal_expression': [8.7, 9.3, 10.2, 8.9, 11.1]
})

# Calculate fold change
data['fold_change'] = data['cancer_expression'] - data['normal_expression']

# Save results
data.to_csv('output/biomarkers.csv', index=False)

print("Top biomarkers:")
for i, row in data.iterrows():
    direction = "UP" if row['fold_change'] > 0 else "DOWN"
    print(f"  {row['gene']}: {direction} (FC = {row['fold_change']:.1f})")

print("✅ Analysis complete! Results saved to output/")