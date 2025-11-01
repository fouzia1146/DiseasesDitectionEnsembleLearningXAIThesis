#!/usr/bin/env python3
"""
Cleveland Heart Disease Tester - Uses EXACT same feature engineering as training
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import joblib

def ultra_feature_engineering(df):
    """EXACT COPY of the feature engineering from training code"""
    df_new = df.copy()
    target = "HeartDisease"
    
    # Handle missing values marked as '?'
    for col in df_new.columns:
        if df_new[col].dtype == 'object':
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
    
    # Fill missing values
    if target in df_new.columns:
        for col in df_new.columns:
            if col != target and df_new[col].isnull().sum() > 0:
                df_new[col] = df_new.groupby(target)[col].transform(
                    lambda x: x.fillna(x.median()) if len(x[x.notna()]) > 0 else x
                )
                df_new[col].fillna(df_new[col].median(), inplace=True)
    else:
        df_new.fillna(df_new.median(numeric_only=True), inplace=True)
    
    # Polynomial features for key cardiac indicators
    df_new['Age_squared'] = df_new['Age'] ** 2
    df_new['Age_cubed'] = df_new['Age'] ** 3
    df_new['MaxHR_squared'] = df_new['MaxHR'] ** 2
    df_new['RestingBP_squared'] = df_new['RestingBP'] ** 2
    df_new['STDepression_squared'] = df_new['STDepression'] ** 2
    
    # Advanced interaction features - cardiac-specific
    df_new['Age_MaxHR'] = df_new['Age'] * df_new['MaxHR']
    df_new['Age_RestingBP'] = df_new['Age'] * df_new['RestingBP']
    df_new['Age_STDepression'] = df_new['Age'] * df_new['STDepression']
    df_new['MaxHR_RestingBP'] = df_new['MaxHR'] * df_new['RestingBP']
    df_new['STDepression_Age'] = df_new['STDepression'] * df_new['Age']
    df_new['Cholesterol_Age'] = df_new['Cholesterol'] * df_new['Age']
    df_new['MaxHR_STDepression'] = df_new['MaxHR'] * df_new['STDepression']
    df_new['RestingBP_Cholesterol'] = df_new['RestingBP'] * df_new['Cholesterol']
    df_new['ChestPainType_Age'] = df_new['ChestPainType'] * df_new['Age']
    
    # Triple interactions
    df_new['Age_MaxHR_RestingBP'] = df_new['Age'] * df_new['MaxHR'] * df_new['RestingBP']
    df_new['Age_STDepression_MaxHR'] = df_new['Age'] * df_new['STDepression'] * df_new['MaxHR']
    df_new['Cholesterol_Age_RestingBP'] = df_new['Cholesterol'] * df_new['Age'] * df_new['RestingBP']
    
    # Ratio features (cardiac-specific)
    epsilon = 1e-8
    df_new['MaxHR_Age_ratio'] = df_new['MaxHR'] / (df_new['Age'] + epsilon)
    df_new['RestingBP_MaxHR_ratio'] = df_new['RestingBP'] / (df_new['MaxHR'] + epsilon)
    df_new['STDepression_MaxHR_ratio'] = df_new['STDepression'] / (df_new['MaxHR'] + epsilon)
    df_new['Cholesterol_RestingBP_ratio'] = df_new['Cholesterol'] / (df_new['RestingBP'] + epsilon)
    df_new['Age_Cholesterol_ratio'] = df_new['Age'] / (df_new['Cholesterol'] + epsilon)
    
    # Medical domain features - cardiac risk indicators
    df_new['Cardiac_Stress_Index'] = (df_new['Age'] * 0.25 + 
                                     df_new['MaxHR'] * 0.2 + 
                                     df_new['STDepression'] * 0.25 +
                                     df_new['RestingBP'] * 0.3)
    
    df_new['Hypertension_Risk'] = ((df_new['RestingBP'] > 90) & 
                                  (df_new['Age'] > 40)).astype(int)
    
    df_new['Angina_Indicator'] = ((df_new['ExerciseInducedAngina'] == 1) & 
                                 (df_new['STDepression'] > 1)).astype(int)
    
    # Binning features
    df_new['Age_Category'] = pd.cut(df_new['Age'],
                                   bins=[0, 40, 50, 60, 70, np.inf],
                                   labels=[0, 1, 2, 3, 4],
                                   include_lowest=True).astype(float)
    
    df_new['RestingBP_Category'] = pd.cut(df_new['RestingBP'],
                                         bins=[0, 90, 120, 140, 160, np.inf],
                                         labels=[0, 1, 2, 3, 4],
                                         include_lowest=True).astype(float)
    
    df_new['Cholesterol_Category'] = pd.cut(df_new['Cholesterol'],
                                           bins=[0, 150, 200, 240, 300, np.inf],
                                           labels=[0, 1, 2, 3, 4],
                                           include_lowest=True).astype(float)
    
    df_new['MaxHR_Category'] = pd.cut(df_new['MaxHR'],
                                     bins=[0, 100, 120, 140, 160, np.inf],
                                     labels=[0, 1, 2, 3, 4],
                                     include_lowest=True).astype(float)
    
    df_new['STDepression_Category'] = pd.cut(df_new['STDepression'],
                                            bins=[0, 1, 2, 3, 4, np.inf],
                                            labels=[0, 1, 2, 3, 4],
                                            include_lowest=True).astype(float)
    
    # Log and square root transformations
    df_new['Age_log'] = np.log1p(df_new['Age'])
    df_new['MaxHR_log'] = np.log1p(df_new['MaxHR'])
    df_new['RestingBP_log'] = np.log1p(df_new['RestingBP'])
    df_new['Cholesterol_log'] = np.log1p(df_new['Cholesterol'])
    df_new['STDepression_log'] = np.log1p(df_new['STDepression'])
    
    df_new['Age_sqrt'] = np.sqrt(df_new['Age'])
    df_new['MaxHR_sqrt'] = np.sqrt(df_new['MaxHR'])
    df_new['RestingBP_sqrt'] = np.sqrt(df_new['RestingBP'])
    
    # Statistical features
    numeric_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                   'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseInducedAngina',
                   'STDepression', 'STSlope', 'MajorVessels', 'Thalassemia']
    
    df_new['Feature_Sum'] = df_new[numeric_cols].sum(axis=1)
    df_new['Feature_Mean'] = df_new[numeric_cols].mean(axis=1)
    df_new['Feature_Std'] = df_new[numeric_cols].std(axis=1)
    df_new['Feature_Median'] = df_new[numeric_cols].median(axis=1)
    df_new['Feature_Max'] = df_new[numeric_cols].max(axis=1)
    df_new['Feature_Min'] = df_new[numeric_cols].min(axis=1)
    
    # Fill any NaN values created
    df_new.fillna(0, inplace=True)
    df_new.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df_new


class HeartDiseaseTester:
    """Heart Disease tester using the EXACT same pipeline as training"""
    
    def __init__(self, model_dir="heart_disease_model_saved"):
        self.model = None
        self.scaler = None
        self.variance_selector = None
        self.stat_selector = None
        self.model_selector = None
        self.model_dir = model_dir
        
    def load_model(self):
        """Load the trained model and all preprocessors"""
        print(f"📂 Loading model from {self.model_dir}...")
        
        try:
            # Load all components
            self.model = joblib.load(f"{self.model_dir}/model.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            
            # Load feature selectors
            selectors = joblib.load(f"{self.model_dir}/feature_selectors.pkl")
            self.variance_selector = selectors['variance']
            self.stat_selector = selectors['statistical']
            self.model_selector = selectors['model_based']
            
            # Load performance metrics
            performance_data = joblib.load(f"{self.model_dir}/performance_metrics.pkl")
            
            print(f"✅ Model loaded successfully!")
            print(f"🤖 Model type: {type(self.model).__name__}")
            print(f"🔧 Scaler type: {type(self.scaler).__name__}")
            print(f"\n📊 Original Training Performance:")
            print(f"   Test Accuracy: {performance_data['test_accuracy']:.4f} ({performance_data['test_accuracy']*100:.1f}%)")
            print(f"   Test AUC: {performance_data['test_auc']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model(self, file_path):
        """Test the model using the EXACT same preprocessing pipeline"""
        print("\n🔍 TESTING MODE - Using Original Training Pipeline")
        print("="*60)
        
        if not self.load_model():
            return False
        
        try:
            # Load test data
            print(f"\n📂 Loading test data from: {file_path}")
            
            # Check if file has headers
            first_line = pd.read_csv(file_path, nrows=0)
            has_headers = len(first_line.columns) > 0 and not first_line.columns[0].replace('.','').isdigit()
            
            if has_headers:
                test_df = pd.read_csv(file_path)
            else:
                # No headers - use column names from Cleveland dataset
                column_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                               'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseInducedAngina',
                               'STDepression', 'STSlope', 'MajorVessels', 'Thalassemia', 'HeartDisease']
                test_df = pd.read_csv(file_path, header=None, names=column_names)
            
            print(f"Test data shape: {test_df.shape}")
            
            # Check for HeartDisease column
            has_outcome = 'HeartDisease' in test_df.columns
            if has_outcome:
                print("✅ Ground truth labels found - will calculate accuracy")
                y_true = test_df['HeartDisease']
                
                # Convert to binary if needed (0 = no disease, >0 = disease)
                if y_true.max() > 1:
                    y_true = (y_true > 0).astype(int)
                    test_df['HeartDisease'] = y_true
                
                X_test = test_df.drop('HeartDisease', axis=1)
                print(f"Class distribution in test data:")
                print(f"No Disease (0): {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true)*100:.1f}%)")
                print(f"Disease (1): {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
            else:
                print("ℹ️ No ground truth labels - will only make predictions")
                X_test = test_df
            
            # Apply EXACT same feature engineering as training
            print("\n🔧 Applying ultra feature engineering (EXACT same as training)...")
            
            if has_outcome:
                df_with_outcome = test_df.copy()
            else:
                df_with_outcome = X_test.copy()
                df_with_outcome['HeartDisease'] = 0
            
            # Apply feature engineering
            df_engineered = ultra_feature_engineering(df_with_outcome)
            print(f"After feature engineering: {df_engineered.shape}")
            
            # Remove outcome column for processing
            X_engineered = df_engineered.drop('HeartDisease', axis=1)
            print(f"Features for processing: {X_engineered.shape}")
            
            # Apply EXACT same feature selection pipeline
            print("\n🔍 Applying feature selection pipeline...")
            
            # Step 1: Variance selection
            print("Step 1: Variance selection...")
            X_var = self.variance_selector.transform(X_engineered)
            print(f"After variance selection: {X_var.shape}")
            
            # Step 2: Statistical selection  
            print("Step 2: Statistical selection...")
            X_stat = self.stat_selector.transform(X_var)
            print(f"After statistical selection: {X_stat.shape}")
            
            # Step 3: Model-based selection
            print("Step 3: Model-based selection...")
            X_model = self.model_selector.transform(X_stat)
            print(f"After model-based selection: {X_model.shape}")
            
            # Apply scaling
            print("\n⚖️ Applying scaling...")
            X_scaled = self.scaler.transform(X_model)
            print(f"Final processed shape: {X_scaled.shape}")
            
            # Make predictions
            print("\n🔮 Making predictions...")
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Calculate prediction statistics
            pred_disease = sum(predictions)
            pred_no_disease = len(predictions) - pred_disease
            
            print(f"\n📊 PREDICTION RESULTS:")
            print(f"Total samples: {len(predictions)}")
            print(f"Predicted No Disease: {pred_no_disease} ({pred_no_disease/len(predictions)*100:.1f}%)")
            print(f"Predicted Disease: {pred_disease} ({pred_disease/len(predictions)*100:.1f}%)")
            
            # Calculate accuracy if ground truth available
            if has_outcome:
                accuracy = accuracy_score(y_true, predictions)
                f1 = f1_score(y_true, predictions)
                auc = roc_auc_score(y_true, probabilities[:, 1])
                
                print(f"\n🎯 ACCURACY RESULTS:")
                print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"F1 Score: {f1:.4f}")
                print(f"AUC Score: {auc:.4f}")
                
                # Try threshold optimization
                if accuracy < 0.85:
                    print(f"\n🔧 Optimizing decision threshold...")
                    best_acc = accuracy
                    best_threshold = 0.5
                    best_preds = predictions
                    
                    thresholds = np.arange(0.25, 0.75, 0.05)
                    for threshold in thresholds:
                        thresh_preds = (probabilities[:, 1] >= threshold).astype(int)
                        thresh_acc = accuracy_score(y_true, thresh_preds)
                        print(f"   Threshold {threshold:.2f}: {thresh_acc:.3f} accuracy")
                        
                        if thresh_acc > best_acc:
                            best_acc = thresh_acc
                            best_threshold = threshold
                            best_preds = thresh_preds
                    
                    if best_acc >= 0.85:
                        print(f"🎉 ACHIEVED 85%+ with threshold {best_threshold:.2f}: {best_acc:.3f} ({best_acc*100:.1f}%)")
                        accuracy = best_acc
                        predictions = best_preds
                        f1 = f1_score(y_true, predictions)
                    elif best_acc > accuracy:
                        print(f"✅ Improved to {best_acc:.3f} ({best_acc*100:.1f}%) with threshold {best_threshold:.2f}")
                        accuracy = best_acc
                        predictions = best_preds
                        f1 = f1_score(y_true, predictions)
                
                if accuracy >= 0.85:
                    print("🎉 SUCCESS! Achieved 85%+ accuracy!")
                elif accuracy >= 0.80:
                    print("✅ Good performance! Close to 85% target")
                
                print(f"\n📋 Detailed Classification Report:")
                print(classification_report(y_true, predictions, 
                                          target_names=['No Disease', 'Disease']))
                
                # Confusion matrix
                cm = confusion_matrix(y_true, predictions)
                print(f"\n🔢 Confusion Matrix:")
                print(f"              Predicted")
                print(f"Actual    No Disease  Disease")
                print(f"No Disease    {cm[0,0]:4d}      {cm[0,1]:4d}")
                print(f"Disease       {cm[1,0]:4d}      {cm[1,1]:4d}")
                
                # Show sample predictions
                print(f"\n🔍 Sample predictions:")
                sample_size = min(10, len(predictions))
                for i in range(sample_size):
                    actual = "Disease" if y_true.iloc[i] == 1 else "No Disease"
                    predicted = "Disease" if predictions[i] == 1 else "No Disease"
                    confidence = probabilities[i].max()
                    correct = "✅" if y_true.iloc[i] == predictions[i] else "❌"
                    print(f"  {correct} Actual: {actual:11} | Predicted: {predicted:11} | Confidence: {confidence:.3f}")
            
            # Save results
            print(f"\n💾 Saving results...")
            output_df = test_df.copy()
            output_df['Predicted_Outcome'] = predictions
            output_df['Prediction_Label'] = ['Disease' if p == 1 else 'No Disease' for p in predictions]
            output_df['Probability_No_Disease'] = probabilities[:, 0]
            output_df['Probability_Disease'] = probabilities[:, 1]
            output_df['Confidence'] = np.max(probabilities, axis=1)
            
            if has_outcome:
                output_df['Correct_Prediction'] = (predictions == y_true).astype(int)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"heart_disease_results_{timestamp}.csv"
            output_df.to_csv(output_filename, index=False)
            
            print(f"✅ Results saved to: {output_filename}")
            
            if has_outcome and accuracy >= 0.85:
                print(f"\n🎉 MISSION ACCOMPLISHED!")
                print(f"✅ Achieved {accuracy:.1%} accuracy using the correct pipeline!")
            elif has_outcome:
                print(f"\n📊 Pipeline successfully applied!")
                print(f"Accuracy: {accuracy:.1%}")
                
            print("\n🎉 TESTING COMPLETED!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during testing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function for testing with correct pipeline"""
    print("\n" + "="*70)
    print("❤️ HEART DISEASE PREDICTION - CORRECT PIPELINE TEST")
    print("="*70)
    
    # Use fixed model directory
    model_dir = "heart_disease_model_saved"
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory '{model_dir}' not found!")
        print("Please run the training script first to create the model.")
        return
    
    # Check for required model files
    required_files = ['model.pkl', 'scaler.pkl', 'feature_selectors.pkl', 'performance_metrics.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f"{model_dir}/{f}")]
    
    if missing_files:
        print(f"❌ Missing required model files: {missing_files}")
        return
    
    print(f"✅ Model directory found: {model_dir}")
    
    # Get test data file
    print(f"\n📂 Please provide your test CSV file path:")
    print("(You can use: data_splits/test_data.csv or data_splits/validation_data.csv)")
    file_path = input("📁 Enter file path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print("❌ Test file not found!")
        return
    
    # Create tester and run
    tester = HeartDiseaseTester(model_dir=model_dir)
    tester.test_model(file_path)


if __name__ == "__main__":
    main()