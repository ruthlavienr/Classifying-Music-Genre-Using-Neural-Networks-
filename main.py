import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layers = []
        
        # He initialization for ReLU networks
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i]),
                'biases': np.zeros((1, layer_sizes[i + 1]))
            }
            self.layers.append(layer)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        for i, layer in enumerate(self.layers):
            z = np.dot(current_input, layer['weights']) + layer['biases']
            self.z_values.append(z)
            
            if i == len(self.layers) - 1:
                activation = self.softmax(z)
            else:
                activation = self.relu(z)
            
            self.activations.append(activation)
            current_input = activation
        
        return self.activations[-1]
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        
        output_error = y_pred - y_true
        errors = [output_error]
        
        for i in range(len(self.layers) - 2, -1, -1):
            error = np.dot(errors[-1], self.layers[i + 1]['weights'].T) * self.relu_derivative(self.activations[i + 1])
            errors.append(error)
        
        errors.reverse()
        
        for i, layer in enumerate(self.layers):
            layer['weights'] -= self.learning_rate * np.dot(self.activations[i].T, errors[i]) / m
            layer['biases'] -= self.learning_rate * np.mean(errors[i], axis=0, keepdims=True)
    
    def train(self, X, y, epochs=150, batch_size=64, validation_data=None):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, X.shape[0], batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                predictions = self.forward(batch_X)
                batch_loss = self.categorical_crossentropy(batch_y, predictions)
                epoch_loss += batch_loss
                num_batches += 1
                
                self.backward(batch_X, batch_y, predictions)
            
            train_predictions = self.forward(X)
            train_accuracy = self.accuracy(y, train_predictions)
            avg_loss = epoch_loss / num_batches
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(train_accuracy)
            
            if validation_data:
                val_X, val_y = validation_data
                val_predictions = self.forward(val_X)
                val_loss = self.categorical_crossentropy(val_y, val_predictions)
                val_accuracy = self.accuracy(val_y, val_predictions)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        return history
    
    def predict(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X)
    
    def categorical_crossentropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def accuracy(self, y_true, y_pred):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(y_true_labels == y_pred_labels)

def toCategorical(y, num_classes=None):
    if num_classes is None:
        num_classes = len(np.unique(y))
    
    categorical = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        categorical[i, label] = 1
    return categorical

def loadAndPreprocessData():
    print("Loading music data...")
    
    trainData = pd.read_csv('train_data.csv')
    print(f"Training data loaded: {trainData.shape}")
    
    testData = pd.read_csv('test_data.csv')
    print(f"Test data loaded: {testData.shape}")
    
    print("\nDataset Info:")
    print("Training data columns:", trainData.columns.tolist())
    print("Test data columns:", testData.columns.tolist())
    
    print("\nMissing values in training data:")
    print(trainData.isnull().sum())
    
    print("\nClass distribution:")
    print(trainData['Class'].value_counts().sort_index())
    
    audioFeatures = [
        'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'duration_in min/ms', 'time_signature'
    ]
    
    xTrain = trainData[audioFeatures].values
    yTrain = trainData['Class'].values
    xTest = testData[audioFeatures].values
    
    # Extract artist and track names for test data
    testMetadata = testData[['Artist Name', 'Track Name']].copy()
    
    print(f"\nFeatures selected: {len(audioFeatures)}")
    print("Audio features:", audioFeatures)
    print(f"Training features shape: {xTrain.shape}")
    print(f"Training labels shape: {yTrain.shape}")
    print(f"Test features shape: {xTest.shape}")
    
    return xTrain, yTrain, xTest, audioFeatures, testMetadata

def domainSpecificImputation(trainData, testData, featureNames):
    print("\nApplying Domain-Specific Music Imputation")
    print("#" * 50)
    
    trainDf = pd.DataFrame(trainData, columns=featureNames + ['Class'])
    testDf = pd.DataFrame(testData, columns=featureNames)
    
    print("\nStep 1: Analyzing missing data patterns...")
    
    for dfName, df in [("Training", trainDf), ("Test", testDf)]:
        missingInfo = df.isnull().sum()
        print(f"\n{dfName} missing data:")
        for col, missingCount in missingInfo.items():
            if missingCount > 0:
                percentage = (missingCount / len(df)) * 100
                print(f"  {col}: {missingCount} ({percentage:.1f}%)")
    
    print("\nStep 2: Applying music domain knowledge...")
    
    # Popularity imputation based on genre
    print("\nPopularity imputation (genre-based)...")
    for classVal in trainDf['Class'].unique():
        if pd.isna(classVal):
            continue
        classMask = (trainDf['Class'] == classVal)
        missingMask = trainDf['popularity'].isnull()
        fillMask = missingMask & classMask
        
        if fillMask.sum() > 0:
            classMedian = trainDf.loc[classMask, 'popularity'].median()
            if pd.isna(classMedian):
                classMedian = 45
            
            trainDf.loc[fillMask, 'popularity'] = classMedian
            print(f"    Class {classVal}: filled {fillMask.sum()} popularity values with {classMedian:.1f}")
    
    testMissing = testDf['popularity'].isnull()
    if testMissing.sum() > 0:
        overallMedian = 44
        testDf.loc[testMissing, 'popularity'] = overallMedian
        print(f"    Test data: filled {testMissing.sum()} popularity values with {overallMedian:.1f}")
    
    # Key imputation based on music theory
    print("\nKey imputation (music theory)...")
    for classVal in trainDf['Class'].unique():
        if pd.isna(classVal):
            continue
        classMask = (trainDf['Class'] == classVal)
        missingMask = trainDf['key'].isnull()
        fillMask = missingMask & classMask
        
        if fillMask.sum() > 0:
            classKeyMode = trainDf.loc[classMask, 'key'].mode()
            
            if len(classKeyMode) > 0 and not pd.isna(classKeyMode.iloc[0]):
                fillKey = classKeyMode.iloc[0]
            else:
                fillKey = np.random.choice([0, 2, 7, 9])  # Common keys
            
            trainDf.loc[fillMask, 'key'] = fillKey
            print(f"    Class {classVal}: filled {fillMask.sum()} key values with {fillKey}")
    
    testMissing = testDf['key'].isnull()
    if testMissing.sum() > 0:
        testDf.loc[testMissing, 'key'] = 2
        print(f"    Test data: filled {testMissing.sum()} key values with 2")
    
    # Instrumentalness imputation based on speechiness
    print("\nInstrumentalness imputation (speech-based)...")
    
    # High speechiness indicates low instrumentalness
    for dfName, df in [("Training", trainDf), ("Test", testDf)]:
        missingInst = df['instrumentalness'].isnull()
        
        if missingInst.sum() > 0:
            highSpeech = df['speechiness'] > 0.33
            mediumSpeech = (df['speechiness'] >= 0.1) & (df['speechiness'] <= 0.33)
            
            mask1 = missingInst & highSpeech
            if mask1.sum() > 0:
                df.loc[mask1, 'instrumentalness'] = np.random.uniform(0.0, 0.1, mask1.sum())
                print(f"    High speechiness: filled {mask1.sum()} with low instrumentalness (0.0-0.1)")
            
            mask2 = missingInst & mediumSpeech
            if mask2.sum() > 0:
                df.loc[mask2, 'instrumentalness'] = np.random.uniform(0.1, 0.3, mask2.sum())
                print(f"    Medium speechiness: filled {mask2.sum()} with medium instrumentalness (0.1-0.3)")
            
            # Handle remaining missing values by genre (for training data)
            remainingMissing = df['instrumentalness'].isnull()
            if remainingMissing.sum() > 0:
                if 'Class' in df.columns:
                    for classVal in df['Class'].unique():
                        if pd.isna(classVal):
                            continue
                        classMask = remainingMissing & (df['Class'] == classVal)
                        if classMask.sum() > 0:
                            classMedian = df.loc[df['Class'] == classVal, 'instrumentalness'].median()
                            if pd.isna(classMedian):
                                classMedian = 0.1
                            df.loc[classMask, 'instrumentalness'] = classMedian
                            print(f"    Class {classVal} (low speech): filled {classMask.sum()} values")
                else:
                    df.loc[remainingMissing, 'instrumentalness'] = 0.1
                    print(f"    Remaining: filled {remainingMissing.sum()} with defaults")
    
    # Handle any remaining missing values
    print("\nFinal cleanup...")
    
    musicDefaults = {
        'danceability': 0.5,
        'energy': 0.6,
        'loudness': -8.0,
        'mode': 1,
        'speechiness': 0.05,
        'acousticness': 0.2,
        'liveness': 0.15,
        'valence': 0.5,
        'tempo': 120.0,
        'duration_in min/ms': 220000,
        'time_signature': 4
    }
    
    for dfName, df in [("Training", trainDf), ("Test", testDf)]:
        for feature, defaultVal in musicDefaults.items():
            if feature in df.columns:
                missingCount = df[feature].isnull().sum()
                if missingCount > 0:
                    df[feature].fillna(defaultVal, inplace=True)
    
    trainFeatures = trainDf[featureNames].values
    testFeatures = testDf[featureNames].values
    
    print(f"\nImputation completed successfully")
    print(f"Training data missing values: {pd.DataFrame(trainFeatures).isnull().sum().sum()}")
    print(f"Test data missing values: {pd.DataFrame(testFeatures).isnull().sum().sum()}")
    
    return trainFeatures, testFeatures

def main():
    print("Music Classification with Neural Network")
    print("#" * 50)
    
    # Load data
    xTrainFull, yTrainFull, xTestFinal, featureNames, testMetadata = loadAndPreprocessData()
    
    # Apply domain-specific imputation
    trainFullWithLabels = np.column_stack([xTrainFull, yTrainFull])
    xTrainFull, xTestFinal = domainSpecificImputation(
        trainFullWithLabels, xTestFinal, featureNames
    )
    
    # Split data
    xTrain, xVal, yTrain, yVal = train_test_split(
        xTrainFull, yTrainFull, test_size=0.2, random_state=42, stratify=yTrainFull
    )
    
    # Scale features
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xValScaled = scaler.transform(xVal)
    xTestFinalScaled = scaler.transform(xTestFinal)
    
    # Convert labels to categorical
    numClasses = len(np.unique(yTrainFull))
    yTrainCategorical = toCategorical(yTrain, numClasses)
    yValCategorical = toCategorical(yVal, numClasses)
    
    print(f"\nData preprocessing completed:")
    print(f"Training set: {xTrainScaled.shape}")
    print(f"Validation set: {xValScaled.shape}")
    print(f"Final test set: {xTestFinalScaled.shape}")
    print(f"Number of classes: {numClasses}")
    print(f"Class labels: {np.unique(yTrainFull)}")
    
    # Create neural network
    print(f"\nBuilding Neural Network for Music Classification...")
    print("Architecture: Input -> 128 -> 64 -> 32 -> Output")
    
    model = NeuralNetwork(
        input_size=xTrainScaled.shape[1],
        hidden_sizes=[128, 64, 32],
        output_size=numClasses,
        learning_rate=0.001
    )
    
    print(f"Network created with {xTrainScaled.shape[1]} input features")
    
    # Train model
    print("\nTraining the model...")
    history = model.train(
        xTrainScaled, yTrainCategorical,
        epochs=150,
        batch_size=64,
        validation_data=(xValScaled, yValCategorical)
    )
    
    # Evaluate model
    print("\nEvaluating on validation set...")
    valPredictions = model.predict(xValScaled)
    
    print("\nClassification Report (Validation Data):")
    print(classification_report(yVal, valPredictions))
    
    valPredictionsProba = model.predict_proba(xValScaled)
    valAccuracy = model.accuracy(yValCategorical, valPredictionsProba)
    print(f"\nFinal Validation Accuracy: {valAccuracy:.4f}")
    
    # Generate test predictions
    print("\nMaking predictions on final test set...")
    testPredictions = model.predict(xTestFinalScaled)
    testPredictionsProba = model.predict_proba(xTestFinalScaled)
    
    # Save predictions
    testResults = pd.DataFrame({
        'Predicted_Class': testPredictions
    })
    testResults.to_csv('music_predictions.csv', index=False)
    print("Test predictions saved to 'music_predictions.csv'")
    
    # Display prediction distribution
    print("\nPrediction distribution on test set:")
    unique, counts = np.unique(testPredictions, return_counts=True)
    for classLabel, count in zip(unique, counts):
        percentage = (count / len(testPredictions)) * 100
        print(f"Class {classLabel}: {count} samples ({percentage:.1f}%)")
    
    # Model summary
    print(f"\nModel Architecture Summary:")
    totalParams = 0
    for i, layer in enumerate(model.layers):
        layerParams = layer['weights'].size + layer['biases'].size
        totalParams += layerParams
        print(f"Layer {i+1}: {layer['weights'].shape[0]} -> {layer['weights'].shape[1]}, Parameters: {layerParams}")
    
    print(f"Total parameters: {totalParams}")
    
    # Feature importance analysis
    print(f"\nFeature Analysis:")
    firstLayerWeights = np.abs(model.layers[0]['weights'])
    featureImportance = np.mean(firstLayerWeights, axis=1)
    
    featureRanking = sorted(zip(featureNames, featureImportance), key=lambda x: x[1], reverse=True)
    print("Top 5 most important features:")
    for i, (feature, importance) in enumerate(featureRanking[:5]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Class prediction confidence
    print(f"\nClass Prediction Analysis:")
    for classLabel in range(numClasses):
        classMask = testPredictions == classLabel
        if classMask.sum() > 0:
            avgConfidence = np.mean(testPredictionsProba[classMask, classLabel])
            print(f"Class {classLabel}: Average confidence = {avgConfidence:.3f}")
    
    # Save detailed results
    detailedResults = pd.DataFrame({
        'Artist_Name': testMetadata['Artist Name'].values,
        'Track_Name': testMetadata['Track Name'].values,
        'Predicted_Class': testPredictions,
        'Confidence': np.max(testPredictionsProba, axis=1)
    })
    
    for i in range(numClasses):
        detailedResults[f'Class_{i}_Probability'] = testPredictionsProba[:, i]
    
    detailedResults.to_csv('detailed_music_predictions.csv', index=False)
    print(f"\nDetailed predictions saved to 'detailed_music_predictions.csv'")
    
    # Create professional visualizations
    print("\nGenerating training visualizations...")
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Prediction Confidence Distribution
    plt.subplot(1, 3, 3)
    plt.hist(np.max(testPredictionsProba, axis=1), bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Predictions')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('music_classification_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training visualizations saved as 'music_classification_results.png'")
    
    # Final summary
    print(f"\n" + "#"*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("#"*60)
    print(f"Final Validation Accuracy: {valAccuracy:.4f} ({valAccuracy*100:.2f}%)")
    print(f"Total Model Parameters: {totalParams:,}")
    print(f"Training Epochs: 150")
    print(f"Batch Size: 64")
    print(f"Learning Rate: 0.001")
    print(f"Network Architecture: {xTrainScaled.shape[1]} -> 128 -> 64 -> 32 -> {numClasses}")
    
    return model, history, testPredictions

if __name__ == "__main__":
    model, history, predictions = main()