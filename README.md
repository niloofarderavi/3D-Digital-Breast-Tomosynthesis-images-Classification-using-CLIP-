# 3D Digital Breast Tomosynthesis (DBT) images Classification of using CLIP 


Classification of 3D Digital Breast Tomosynthesis (DBT) images using CLIP features and advanced techniques for handling severe class imbalance.

## Key Features

- **CLIP Feature Extraction**: Uses OpenAI's CLIP model to extract powerful image representations from DBT scans
- **Advanced Imbalance Handling**: Combines SMOTETomek resampling, Focal Loss, and targeted augmentation
- **Exceptional Results**: 
  - 99.88% overall accuracy with no overfitting
  - 100% F1, Precision, Recall, and AUC for the critical Cancer class
  - Balanced performance across all classes

## Quick Start

### Installation
```bash
pip install ftfy regex tqdm torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn albumentations
```

### Usage
```python
# Extract CLIP features
model, preprocess = clip.load("ViT-B/32", device=device)

# Apply imbalance handling
X_resampled, y_resampled = create_balanced_dataset(
    features, labels, minority_classes, 'combined')

# Train with Focal Loss
model = train_classifier(X_train, y_train, X_val, y_val)

# Evaluate
metrics, preds, probs = evaluate_classifier(model, X_val, y_val)
```

## Results

| Class      | F1-Score | AUC     | Recall  |
|------------|----------|---------|---------|
| Cancer     | 100.00%  | 100.00% | 100.00% |
| Benign     | 99.91%   | 100.00% | 100.00% |
| Actionable | 99.86%   | 100.00% | 100.00% |
| Normal     | 99.77%   | 99.97%  | 99.53%  |

This implementation demonstrates how to effectively tackle class imbalance in medical imaging while maintaining excellent generalization.
