# CANCER BIOMARKER MACHINE LEARNING PIPELINE
print("🤖 CANCER CLASSIFICATION WITH MACHINE LEARNING")
print("=" * 50)

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

print("📁 Loading previously generated data...")

# Load the data we created
expression_df = pd.read_csv('results/expression_data_sample.csv', index_col=0)
clinical_df = pd.read_csv('results/clinical_data.csv')
biomarkers_df = pd.read_csv('results/significant_biomarkers.csv')

print(f"✅ Loaded: {expression_df.shape[0]} samples, {expression_df.shape[1]} genes")
print(f"✅ Top biomarkers: {len(biomarkers_df)} significant genes")

# Prepare data for machine learning
print("\n🔧 Preparing data for machine learning...")

# Use top 20 biomarkers as features
top_biomarkers = biomarkers_df.head(20)['gene'].tolist()

# Create feature matrix (X) and target vector (y)
X = expression_df[top_biomarkers].copy()
y = clinical_df['cancer_type'].apply(lambda x: 1 if x == 'Tumor' else 0)

print(f"Feature matrix: {X.shape}")
print(f"Target classes: {y.value_counts().to_dict()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Train multiple machine learning models
print("\n🤖 Training machine learning models...")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"   Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_prob
    }
    
    print(f"     ✅ {name}: Accuracy = {accuracy:.3f}, AUC = {auc:.3f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]

print(f"\n🏆 Best model: {best_model_name}")
print(f"   Accuracy: {best_model['accuracy']:.3f}")
print(f"   AUC: {best_model['auc']:.3f}")

# Feature importance (for Random Forest)
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'gene': top_biomarkers,
        'importance': best_model['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 10 most important biomarkers for classification:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['gene']}: {row['importance']:.4f}")

# Create performance visualization
print("\n📊 Creating performance visualizations...")

plt.figure(figsize=(15, 5))

# Model comparison
plt.subplot(1, 3, 1)
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
auc_scores = [results[name]['auc'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
plt.bar(x + width/2, auc_scores, width, label='AUC', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Feature importance (if available)
if best_model_name == 'Random Forest':
    plt.subplot(1, 3, 2)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['gene'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Biomarkers for Classification')
    plt.gca().invert_yaxis()

# Confusion matrix style visualization
plt.subplot(1, 3, 3)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_model['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix\n({best_model_name})')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.savefig('results/figures/ml_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Performance visualizations saved!")

# Save ML results
print("\n💾 Saving machine learning results...")

ml_summary = {
    'best_model': best_model_name,
    'best_accuracy': float(best_model['accuracy']),
    'best_auc': float(best_model['auc']),
    'all_models': {
        name: {
            'accuracy': float(results[name]['accuracy']),
            'auc': float(results[name]['auc'])
        } for name in results.keys()
    },
    'dataset_info': {
        'training_samples': len(X_train),
        'testing_samples': len(X_test),
        'features_used': len(top_biomarkers),
        'feature_names': top_biomarkers
    }
}

# Add feature importance if available
if best_model_name == 'Random Forest':
    ml_summary['feature_importance'] = feature_importance.head(20).to_dict('records')

with open('results/ml_results.json', 'w') as f:
    json.dump(ml_summary, f, indent=2)

# Update main analysis summary
with open('results/analysis_summary.json', 'r') as f:
    main_summary = json.load(f)

main_summary['machine_learning'] = {
    'best_model': best_model_name,
    'accuracy': float(best_model['accuracy']),
    'status': 'COMPLETED'
}

with open('results/analysis_summary.json', 'w') as f:
    json.dump(main_summary, f, indent=2)

print("✅ Machine learning results saved!")

print("\n" + "="*50)
print("🎉 MACHINE LEARNING PIPELINE COMPLETED!")
print("="*50)
print(f"🏆 Best model: {best_model_name}")
print(f"📊 Accuracy: {best_model['accuracy']:.3f}")
print(f"📈 AUC: {best_model['auc']:.3f}")
print(f"🔬 Features: {len(top_biomarkers)} biomarkers")
print(f"📁 Results saved to results/ml_results.json")

print("\n🌟 Next: Survival analysis and clinical validation!")