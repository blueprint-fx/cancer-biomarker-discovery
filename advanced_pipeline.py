# ADVANCED CANCER BIOMARKER PIPELINE
print("🎯 ADVANCED CANCER BIOMARKER DISCOVERY")
print("=" * 50)

import pandas as pd
import numpy as np
import os
import json

print("📁 Creating project structure...")
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# Generate realistic cancer dataset
print("🧬 Generating cancer dataset...")
np.random.seed(42)

n_samples = 100
n_genes = 500

# Create sample and gene names
sample_ids = [f"Sample_{i:03d}" for i in range(n_samples)]
gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]

# Generate expression data (normal vs tumor)
normal_data = np.random.normal(10, 2, (40, n_genes))
tumor_data = np.random.normal(15, 3, (60, n_genes))

# Make specific genes differentially expressed
tumor_data[:, :50] += 6   # Strongly upregulated
tumor_data[:, 50:100] -= 5  # Strongly downregulated
tumor_data[:, 100:150] += 3  # Moderately upregulated

# Combine data
expression_matrix = np.vstack([normal_data, tumor_data])

# Create expression DataFrame
expression_df = pd.DataFrame(
    expression_matrix,
    index=sample_ids,
    columns=gene_names
)

# Create clinical data
clinical_df = pd.DataFrame({
    'sample_id': sample_ids,
    'cancer_type': ['Normal'] * 40 + ['Tumor'] * 60,
    'stage': ['Normal'] * 40 + ['I'] * 20 + ['II'] * 25 + ['III'] * 15
})

print(f"✅ Dataset created: {n_samples} samples, {n_genes} genes")

# Perform differential expression analysis
print("\n🔬 Performing differential expression analysis...")

# Get tumor and normal samples
tumor_samples = clinical_df[clinical_df['cancer_type'] == 'Tumor']['sample_id']
normal_samples = clinical_df[clinical_df['cancer_type'] == 'Normal']['sample_id']

tumor_expression = expression_df.loc[tumor_samples]
normal_expression = expression_df.loc[normal_samples]

# Calculate fold changes for all genes
de_results = []
for gene in gene_names:
    tumor_mean = tumor_expression[gene].mean()
    normal_mean = normal_expression[gene].mean()
    fold_change = tumor_mean - normal_mean
    
    de_results.append({
        'gene': gene,
        'fold_change': round(fold_change, 3),
        'tumor_mean': round(tumor_mean, 2),
        'normal_mean': round(normal_mean, 2),
        'abs_fold_change': abs(round(fold_change, 3))
    })

# Create results DataFrame
de_df = pd.DataFrame(de_results)
de_df = de_df.sort_values('abs_fold_change', ascending=False)

# Find significant biomarkers (fold change > 2)
significant_biomarkers = de_df[de_df['abs_fold_change'] > 2]

print(f"📊 Analysis complete:")
print(f"   • Total genes analyzed: {len(de_df)}")
print(f"   • Significant biomarkers found: {len(significant_biomarkers)}")
print(f"   • Tumor samples: {len(tumor_samples)}")
print(f"   • Normal samples: {len(normal_samples)}")

print("\n🏆 Top 10 biomarkers:")
for i, row in de_df.head(10).iterrows():
    direction = "🔼 UP" if row['fold_change'] > 0 else "🔽 DOWN"
    print(f"   {row['gene']}: {direction} (FC={row['fold_change']:.2f})")

# Save all results
print("\n💾 Saving results...")

# Save expression data (sample)
expression_df.iloc[:20, :50].to_csv('results/expression_data_sample.csv')

# Save clinical data
clinical_df.to_csv('results/clinical_data.csv', index=False)

# Save differential expression results
de_df.to_csv('results/differential_expression.csv', index=False)
significant_biomarkers.to_csv('results/significant_biomarkers.csv', index=False)

# Create analysis summary
summary = {
    'pipeline_info': {
        'name': 'Advanced Cancer Biomarker Discovery',
        'version': '2.0',
        'date': pd.Timestamp.now().isoformat()
    },
    'dataset_stats': {
        'total_samples': n_samples,
        'tumor_samples': len(tumor_samples),
        'normal_samples': len(normal_samples),
        'total_genes': n_genes
    },
    'analysis_results': {
        'significant_biomarkers': len(significant_biomarkers),
        'top_biomarker': de_df.iloc[0]['gene'],
        'max_fold_change': float(de_df.iloc[0]['fold_change'])
    },
    'top_biomarkers': de_df.head(10).to_dict('records')
}

with open('results/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ All results saved!")
print("\n" + "="*50)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"📁 Results saved in 'results/' folder")
print(f"📊 Found {len(significant_biomarkers)} significant biomarkers")
print(f"🎯 Top biomarker: {de_df.iloc[0]['gene']} (FC={de_df.iloc[0]['fold_change']:.2f})")
print("\n🌟 Ready for machine learning and advanced analysis!")