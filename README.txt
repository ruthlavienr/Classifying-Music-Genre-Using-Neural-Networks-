# Music Genre Classification with Domain-Specific Neural Networks

A comprehensive approach to music genre classification using custom neural networks implemented from scratch, featuring domain-specific imputation methods and ensemble architectures based on music theory principles.

## 📊 Project Overview

This project implements a complete music genre classification system without using high-level machine learning libraries, focusing on three primary objectives

1. Domain-Specific Imputation Leveraging music theory and genre characteristics to handle missing audio features
2. From-Scratch Neural Network Manual implementation of forward propagation, backpropagation, and training processes
3. Feature Importance Analysis Understanding model decision-making through specialized ensemble networks

## 🎵 Features

### Core Implementation
- Custom Neural Network Built from scratch using only NumPy
- Music Theory-Informed Imputation Genre-based popularity, tonal center key assignment, speechiness-instrumentalness relationships
- Professional Visualizations Training curves, accuracy plots, and confidence distributions
- Comprehensive Evaluation Classification reports, confusion matrices, and feature importance analysis

### Technical Specifications
- Architecture 14 → 128 → 64 → 32 → 11 (input → hidden layers → output)
- Activation Functions ReLU (hidden layers), Softmax (output layer)
- Optimization Mini-batch gradient descent with learning rate decay
- Initialization He initialization for ReLU compatibility
- Regularization Early stopping, gradient clipping, numerical stability techniques

## 📁 Dataset Requirements

The system expects two CSV files

### `train_data.csv`
Must contain the following columns
- `Artist Name` Artist name
- `Track Name` Song title
- `popularity` Song popularity score
- `danceability` Danceability measure (0.0 to 1.0)
- `energy` Energy level (0.0 to 1.0)
- `key` Musical key (0-11, representing C, C#, D, etc.)
- `loudness` Loudness in decibels
- `mode` Musical mode (0 = minor, 1 = major)
- `speechiness` Speechiness measure (0.0 to 1.0)
- `acousticness` Acousticness measure (0.0 to 1.0)
- `instrumentalness` Instrumentalness measure (0.0 to 1.0)
- `liveness` Liveness measure (0.0 to 1.0)
- `valence` Musical valence (0.0 to 1.0)
- `tempo` Beats per minute
- `duration_in minms` Track duration
- `time_signature` Time signature
- `Class` Genre label (0-10)

### `test_data.csv`
Same columns as training data except `Class` (target variable)

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Quick Start
```python
# Clone and run the main script
python music_classification.py
```

### Expected Output Files
1. `music_predictions.csv` - Basic predictions with class labels
2. `detailed_music_predictions.csv` - Complete results including
   - Artist Name and Track Name
   - Predicted Class
   - Confidence Score
   - Probability distribution across all 11 classes
3. `music_classification_results.png` - Training visualization plots

## 🎯 Model Performance

### Key Metrics
- Validation Accuracy ~53.19% on 11-class classification
- Total Parameters 12,619
- Training Epochs 150
- Batch Size 64
- Learning Rate 0.001

### Feature Importance Rankings
1. Duration (0.1609)
2. Popularity (0.1460)
3. Acousticness (0.1420)
4. Speechiness (0.1339)
5. Energy (0.1314)

## 🔬 Domain-Specific Imputation Strategies

### 1. Popularity Imputation
- Method Genre-wise median values
- Rationale Different genres have distinct audience reach patterns
- Implementation Class-specific averages preserve genre characteristics

### 2. Key Imputation
- Method Music theory-based tonal center assignment
- Rationale Genres exhibit preferred keys (e.g., electronic music favors certain keys)
- Implementation Mode key per genre with fallback to common keys (C, D, G, A)

### 3. Instrumentalness Imputation
- Method Speechiness-acousticness relationship inference
- Rationale High speechiness indicates low instrumentalness
- Implementation 
  - High speechiness (0.33) → Low instrumentalness (0.0-0.1)
  - Medium speechiness (0.1-0.33) → Medium instrumentalness (0.1-0.3)
  - Low speechiness → Genre-specific patterns

## 📈 Architecture Details

### Neural Network Structure
```
Input Layer (14 features)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Hidden Layer 3 (32 neurons, ReLU)
    ↓
Output Layer (11 classes, Softmax)
```

### Parameter Distribution
- Layer 1 1,920 parameters (14×128 + 128)
- Layer 2 8,256 parameters (128×64 + 64)
- Layer 3 2,080 parameters (64×32 + 32)
- Layer 4 363 parameters (32×11 + 11)
- Total 12,619 parameters

## 🎼 Music Theory Integration

### Genre-Specific Patterns
- ElectronicDance Prefer keys 2, 4, 9 (D, E, A)
- PopRock Common in keys 0, 7, 2 (C, G, D)
- ClassicalJazz Higher instrumentalness values
- Vocal-heavy genres Lower instrumentalness, higher speechiness

### Acoustic Feature Relationships
- Energy ↔ Tempo High-energy tracks typically have faster tempos
- Speechiness ↔ Instrumentalness Inverse relationship preserved
- Valence ↔ Mode Major keys often correlate with higher valence

## 📊 Evaluation Metrics

### Classification Performance
- Precision, Recall, F1-Score Per-class and macroweighted averages
- Confusion Matrix Class-specific performance analysis
- Confidence Scores Model certainty for each prediction

### Feature Analysis
- Weight-based Importance First layer weight magnitudes
- Cross-class Consistency Feature importance across different genres
- Validation Alignment with music cognition research

## 🔧 Customization Options

### Hyperparameter Tuning
```python
model = NeuralNetwork(
    input_size=14,
    hidden_sizes=[128, 64, 32],  # Modify architecture
    output_size=11,
    learning_rate=0.001  # Adjust learning rate
)

# Training parameters
history = model.train(
    epochs=150,        # Number of training epochs
    batch_size=64,     # Batch size for mini-batch gradient descent
    validation_data=(xVal, yVal)
)
```

### Imputation Customization
- Modify genre-specific defaults in `musicDefaults` dictionary
- Adjust speechiness thresholds for instrumentalness imputation
- Add custom musical key preferences for different genres

## 📚 Academic Usage

### Citation Information
This implementation demonstrates
- Domain-specific preprocessing for music data
- From-scratch neural network development for educational purposes
- Integration of music theory with machine learning
- Statistical validation of ensemble approaches

### Educational Value
- Deep Learning Fundamentals Manual implementation reveals optimization mechanics
- Domain Expertise Integration Shows importance of subject matter knowledge
- Feature Engineering Music-informed preprocessing strategies
- Model Interpretability Weight analysis and confidence metrics

## 🤝 Contributing

### Development Guidelines
- Follow camelCase naming conventions
- Maintain professional code documentation
- Include comprehensive error handling
- Preserve music theory-based logic

### Future Enhancements
- Additional regularization techniques (dropout, L2)
- Advanced ensemble architectures
- Real-time audio processing capabilities
- Integration with deep learning frameworks for comparison

## 📄 License

This project is developed for academic and educational purposes. Please cite appropriately when using in research.

## 📞 Support

For questions regarding implementation details, music theory integration, or academic applications, please refer to the comprehensive code documentation and comments within the source files.

---

Note This implementation prioritizes educational value and understanding of neural network fundamentals over raw performance. For production applications, consider integrating with established deep learning frameworks while maintaining the domain-specific preprocessing strategies demonstrated here.