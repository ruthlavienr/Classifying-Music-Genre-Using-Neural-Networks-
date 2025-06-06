# 🎵 Music Genre Classification with Domain-Specific Neural Networks

A comprehensive approach to music genre classification using custom neural networks implemented from scratch, featuring domain-specific imputation methods and ensemble architectures inspired by music theory principles.

---

## 📊 Project Overview

This project implements a full music genre classification system **without using high-level machine learning libraries**, focusing on three main goals:

1. **Domain-Specific Imputation**  
   Leveraging music theory and genre characteristics to handle missing audio features.

2. **From-Scratch Neural Network**  
   Manual implementation of forward propagation, backpropagation, and training processes.

3. **Feature Importance Analysis**  
   Understanding model decision-making through specialized ensemble networks.

---

## 🎵 Features

### Core Implementation

- ✅ Built-from-scratch Neural Network (NumPy-only)
- ✅ Music Theory-Informed Imputation
- ✅ Professional Visualizations (accuracy, loss, confidence)
- ✅ Comprehensive Evaluation (reports, confusion matrices, feature importance)

### Technical Specifications

- **Architecture**: `14 → 128 → 64 → 32 → 11`
- **Activation Functions**: ReLU (hidden), Softmax (output)
- **Optimization**: Mini-batch gradient descent with learning rate decay
- **Initialization**: He initialization
- **Regularization**: Early stopping, gradient clipping, numerical stability

---

## 📁 Dataset Requirements

### `train_data.csv`

Must contain:

- `Artist Name`, `Track Name`, `popularity`, `danceability`, `energy`, `key`, `loudness`, `mode`,  
  `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_in minms`, `time_signature`, `Class`

### `test_data.csv`

Same as training data, excluding the `Class` column.

---

## 🚀 Installation & Usage

### Prerequisites

\`\`\`bash
pip install pandas numpy scikit-learn matplotlib
\`\`\`

### Running the Project

\`\`\`bash
# Clone and run
python music_classification.py
\`\`\`

### Output Files

- `music_predictions.csv`: Basic predictions
- `detailed_music_predictions.csv`: Includes predicted class, confidence score, and full probability vector
- `music_classification_results.png`: Visualization of training performance

---

## 🎯 Model Performance

- **Validation Accuracy**: ~53.19%
- **Total Parameters**: 12,619
- **Epochs**: 150
- **Batch Size**: 64
- **Learning Rate**: 0.001

### 🔍 Feature Importance (Top 5)

| Feature        | Importance |
|----------------|------------|
| Duration       | 0.1609     |
| Popularity     | 0.1460     |
| Acousticness   | 0.1420     |
| Speechiness    | 0.1339     |
| Energy         | 0.1314     |

---

## 🔬 Domain-Specific Imputation Strategies

### 1. **Popularity Imputation**
- **Method**: Genre-wise medians  
- **Rationale**: Different audience reach per genre

### 2. **Key Imputation**
- **Method**: Tonal center assignment  
- **Rationale**: Preferred keys vary by genre (e.g., EDM favors A minor)

### 3. **Instrumentalness Imputation**
- **Method**: Based on speechiness thresholds  
- **Rationale**: Inverse relationship between speechiness and instrumentalness  
- **Rules**:
  - High speechiness → Instrumentalness: 0.0–0.1  
  - Medium speechiness → 0.1–0.3  
  - Low speechiness → Genre-based estimate

---

## 🧠 Architecture Details

\`\`\`
Input Layer (14 features)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Hidden Layer 3 (32 neurons, ReLU)
    ↓
Output Layer (11 classes, Softmax)
\`\`\`

### Parameter Breakdown

| Layer     | Parameters |
|-----------|------------|
| Layer 1   | 1,920      |
| Layer 2   | 8,256      |
| Layer 3   | 2,080      |
| Output    | 363        |
| **Total** | **12,619** |

---

## 🎼 Music Theory Integration

### Genre-Specific Patterns

- **Electronic/Dance**: Prefer keys 2, 4, 9 (D, E, A)
- **Pop/Rock**: Common in keys 0, 7, 2 (C, G, D)
- **Classical/Jazz**: High instrumentalness
- **Vocal-heavy**: Low instrumentalness, high speechiness

### Acoustic Relationships

- **Energy ↔ Tempo**: Faster songs = more energetic
- **Speechiness ↔ Instrumentalness**: Inverse
- **Valence ↔ Mode**: Major mode = higher valence

---

## 📊 Evaluation Metrics

- **Precision / Recall / F1-Score** (macro & per-class)
- **Confusion Matrix**
- **Confidence Scores** (per prediction)

### Feature Analysis

- Based on first-layer weight magnitudes
- Validated against domain expectations

---

## 🔧 Customization Options

### Hyperparameter Tuning

\`\`\`python
model = NeuralNetwork(
    input_size=14,
    hidden_sizes=[128, 64, 32],
    output_size=11,
    learning_rate=0.001
)

history = model.train(
    epochs=150,
    batch_size=64,
    validation_data=(xVal, yVal)
)
\`\`\`

### Custom Imputation

- Adjust `musicDefaults` dictionary for genre-based imputation
- Tune speechiness thresholds
- Modify key preferences for genres

---

## 📚 Academic Usage

### Citation Highlights

This work illustrates:

- Domain-specific imputation
- Manual neural network construction
- Integration of musical knowledge into ML pipelines
- Interpretable model metrics and evaluation

### Educational Focus

- 💡 Learn NN internals through hands-on implementation  
- 🎼 Merge domain knowledge with ML  
- 📊 Understand feature importance and decision paths

---

## 🤝 Contributing

### Guidelines

- Use **camelCase** for variables/functions
- Ensure **clear documentation**
- Include **error handling**
- Maintain **musical domain logic**

### Future Work

- Dropout, L2 regularization
- Ensemble models
- Real-time audio inference
- Comparison with deep learning frameworks

---

## 📄 License

Developed for academic/educational use. Cite properly when using this work in research.

---

## 📞 Support

For questions related to:

- Model implementation
- Domain-specific strategies
- Academic applications

Refer to the in-code documentation and comments.

---

> **Note:** This implementation is optimized for educational clarity and interpretability, not industrial performance. For production use, consider integrating with high-level frameworks while retaining the domain-aware strategies presented here.
