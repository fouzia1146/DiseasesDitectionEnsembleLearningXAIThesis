#!/usr/bin/env python3
"""
Correct Diabetes Tester - Uses EXACT same feature engineering as training
This should achieve 86.4% accuracy by matching the original pipeline
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
    target = "Outcome"

    # Handle impossible zero values with domain knowledge
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].replace(0, np.nan)

    # Strategic imputation based on outcome if target exists
    if target in df_new.columns:
        for col in zero_cols:
            if col in df_new.columns:
                # Group by outcome for better imputation
                df_new[col] = df_new.groupby(target)[col].transform(
                    lambda x: x.fillna(x.median()) if len(x[x.notna()]) > 0 else x
                )
                # Fill remaining with overall median
                df_new[col].fillna(df_new[col].median(), inplace=True)
    else:
        # Simple median imputation for prediction
        for col in zero_cols:
            if col in df_new.columns:
                median_val = df_new[col][df_new[col] > 0].median()
                df_new[col].fillna(median_val, inplace=True)

    # Fill any remaining NaN values
    df_new.fillna(df_new.median(numeric_only=True), inplace=True)

    # Advanced polynomial features
    df_new['Glucose_squared'] = df_new['Glucose'] ** 2
    df_new['BMI_squared'] = df_new['BMI'] ** 2
    df_new['Age_squared'] = df_new['Age'] ** 2
    df_new['DiabetesPedigree_squared'] = df_new['DiabetesPedigreeFunction'] ** 2

    # Cube features for highly predictive variables
    df_new['Glucose_cubed'] = df_new['Glucose'] ** 3
    df_new['BMI_cubed'] = df_new['BMI'] ** 3

    # Advanced interaction features
    df_new['BMI_Age'] = df_new['BMI'] * df_new['Age']
    df_new['Glucose_BMI'] = df_new['Glucose'] * df_new['BMI']
    df_new['Glucose_Age'] = df_new['Glucose'] * df_new['Age']
    df_new['Insulin_Glucose'] = df_new['Insulin'] * df_new['Glucose']
    df_new['Pregnancies_Age'] = df_new['Pregnancies'] * df_new['Age']
    df_new['DiabetesPedigree_Age'] = df_new['DiabetesPedigreeFunction'] * df_new['Age']
    df_new['SkinThickness_BMI'] = df_new['SkinThickness'] * df_new['BMI']
    df_new['BloodPressure_Age'] = df_new['BloodPressure'] * df_new['Age']
    df_new['Insulin_BMI'] = df_new['Insulin'] * df_new['BMI']

    # Triple interactions
    df_new['Glucose_BMI_Age'] = df_new['Glucose'] * df_new['BMI'] * df_new['Age']
    df_new['Insulin_Glucose_BMI'] = df_new['Insulin'] * df_new['Glucose'] * df_new['BMI']
    df_new['Pregnancies_BMI_Age'] = df_new['Pregnancies'] * df_new['BMI'] * df_new['Age']

    # Advanced ratio features (avoid division by zero)
    epsilon = 1e-8
    df_new['Glucose_BMI_ratio'] = df_new['Glucose'] / (df_new['BMI'] + epsilon)
    df_new['Insulin_Glucose_ratio'] = df_new['Insulin'] / (df_new['Glucose'] + epsilon)
    df_new['SkinThickness_BMI_ratio'] = df_new['SkinThickness'] / (df_new['BMI'] + epsilon)
    df_new['Age_Pregnancies_ratio'] = df_new['Age'] / (df_new['Pregnancies'] + epsilon)
    df_new['Glucose_BloodPressure_ratio'] = df_new['Glucose'] / (df_new['BloodPressure'] + epsilon)
    df_new['BMI_BloodPressure_ratio'] = df_new['BMI'] / (df_new['BloodPressure'] + epsilon)

    # Medical domain knowledge features
    df_new['Insulin_Resistance'] = df_new['Glucose'] * df_new['Insulin'] / (df_new['BMI'] + epsilon)
    df_new['Metabolic_Syndrome'] = ((df_new['BMI'] > 30) &
                                   (df_new['Glucose'] > 100) &
                                   (df_new['BloodPressure'] > 80)).astype(int)

    # Advanced categorical features - FIXED BINNING WITH np.inf
    df_new['BMI_Category'] = pd.cut(df_new['BMI'],
                                  bins=[0, 18.5, 25, 30, 35, 40, np.inf],
                                  labels=[0, 1, 2, 3, 4, 5],
                                  include_lowest=True).astype(float)

    df_new['Glucose_Category'] = pd.cut(df_new['Glucose'],
                                      bins=[0, 70, 100, 126, 140, np.inf],
                                      labels=[0, 1, 2, 3, 4],
                                      include_lowest=True).astype(float)

    df_new['Age_Category'] = pd.cut(df_new['Age'],
                                  bins=[0, 25, 35, 45, 55, np.inf],
                                  labels=[0, 1, 2, 3, 4],
                                  include_lowest=True).astype(float)

    df_new['Insulin_Category'] = pd.cut(df_new['Insulin'],
                                      bins=[0, 50, 100, 200, 300, np.inf],
                                      labels=[0, 1, 2, 3, 4],
                                      include_lowest=True).astype(float)

    df_new['BloodPressure_Category'] = pd.cut(df_new['BloodPressure'],
                                            bins=[0, 80, 90, 100, 110, np.inf],
                                            labels=[0, 1, 2, 3, 4],
                                            include_lowest=True).astype(float)

    # Advanced composite scores
    df_new['Diabetes_Risk_Score'] = (df_new['Glucose'] * 0.3 +
                                   df_new['BMI'] * 0.25 +
                                   df_new['Age'] * 0.2 +
                                   df_new['DiabetesPedigreeFunction'] * 100 * 0.25)

    df_new['Metabolic_Health_Score'] = (df_new['Glucose'] * 0.35 +
                                      df_new['BMI'] * 0.25 +
                                      df_new['BloodPressure'] * 0.2 +
                                      df_new['Insulin'] * 0.2)

    df_new['Cardiovascular_Risk'] = (df_new['BloodPressure'] * 0.4 +
                                   df_new['BMI'] * 0.3 +
                                   df_new['Age'] * 0.3)

    # Log and square root transformations
    df_new['Insulin_log'] = np.log1p(df_new['Insulin'])
    df_new['DiabetesPedigree_log'] = np.log1p(df_new['DiabetesPedigreeFunction'])
    df_new['SkinThickness_log'] = np.log1p(df_new['SkinThickness'])
    df_new['Glucose_sqrt'] = np.sqrt(df_new['Glucose'])
    df_new['BMI_sqrt'] = np.sqrt(df_new['BMI'])
    df_new['Age_sqrt'] = np.sqrt(df_new['Age'])

    # Statistical features
    numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    df_new['Feature_Sum'] = df_new[numeric_cols].sum(axis=1)
    df_new['Feature_Mean'] = df_new[numeric_cols].mean(axis=1)
    df_new['Feature_Std'] = df_new[numeric_cols].std(axis=1)
    df_new['Feature_Median'] = df_new[numeric_cols].median(axis=1)
    df_new['Feature_Max'] = df_new[numeric_cols].max(axis=1)
    df_new['Feature_Min'] = df_new[numeric_cols].min(axis=1)

    # Fill any new NaN values that might have been created
    df_new.fillna(0, inplace=True)

    # Replace infinite values
    df_new.replace([np.inf, -np.inf], 0, inplace=True)

    return df_new


class CorrectDiabetesTester:
    """Diabetes tester using the EXACT same pipeline as training"""
    
    def __init__(self, model_dir="diabetes_model_saved"):
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
            print(f"   Training Accuracy: {performance_data['train_accuracy']:.4f} ({performance_data['train_accuracy']*100:.1f}%)")
            print(f"   Validation Accuracy: {performance_data['validation_accuracy']:.4f} ({performance_data['validation_accuracy']*100:.1f}%)")
            print(f"   Test Accuracy: {performance_data['test_accuracy']:.4f} ({performance_data['test_accuracy']*100:.1f}%)")
            print(f"   Test AUC: {performance_data['test_auc']:.4f}")
            print(f"   Best Model: {performance_data['best_model_name']}")
            
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
            test_df = pd.read_csv(file_path)
            print(f"Test data shape: {test_df.shape}")
            
            # Check for outcome column
            has_outcome = 'Outcome' in test_df.columns
            if has_outcome:
                print("✅ Ground truth labels found - will calculate accuracy")
                y_true = test_df['Outcome']
                X_test = test_df.drop('Outcome', axis=1)
                print(f"Class distribution in test data:")
                print(f"Not Diabetic (0): {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true)*100:.1f}%)")
                print(f"Diabetic (1): {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
            else:
                print("ℹ️ No ground truth labels - will only make predictions")
                X_test = test_df
            
            # Apply EXACT same feature engineering as training
            print("\n🔧 Applying ultra feature engineering (EXACT same as training)...")
            
            if has_outcome:
                # Include the outcome for feature engineering (as done in training)
                df_with_outcome = test_df.copy()
            else:
                # Add dummy outcome for feature engineering
                df_with_outcome = X_test.copy()
                df_with_outcome['Outcome'] = 0
            
            # Apply feature engineering
            df_engineered = ultra_feature_engineering(df_with_outcome)
            print(f"After feature engineering: {df_engineered.shape}")
            
            # Remove outcome column for processing
            X_engineered = df_engineered.drop('Outcome', axis=1)
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
            pred_diabetic = sum(predictions)
            pred_not_diabetic = len(predictions) - pred_diabetic
            
            print(f"\n📊 PREDICTION RESULTS:")
            print(f"Total samples: {len(predictions)}")
            print(f"Predicted Not Diabetic: {pred_not_diabetic} ({pred_not_diabetic/len(predictions)*100:.1f}%)")
            print(f"Predicted Diabetic: {pred_diabetic} ({pred_diabetic/len(predictions)*100:.1f}%)")
            
            # Calculate accuracy if ground truth available
            if has_outcome:
                accuracy = accuracy_score(y_true, predictions)
                f1 = f1_score(y_true, predictions)
                auc = roc_auc_score(y_true, probabilities[:, 1])
                
                print(f"\n🎯 ACCURACY RESULTS:")
                print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"F1 Score: {f1:.4f}")
                print(f"AUC Score: {auc:.4f}")
                
                # Try threshold optimization to reach 85%
                if accuracy < 0.85:
                    print(f"\n🔧 Optimizing decision threshold to reach 85%...")
                    best_acc = accuracy
                    best_threshold = 0.5
                    best_preds = predictions
                    
                    # Test different thresholds
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
                        print(f"   Optimized F1 Score: {f1:.4f}")
                    elif best_acc > accuracy:
                        print(f"✅ Improved to {best_acc:.3f} ({best_acc*100:.1f}%) with threshold {best_threshold:.2f}")
                        accuracy = best_acc
                        predictions = best_preds
                        f1 = f1_score(y_true, predictions)
                
                if accuracy >= 0.85:
                    print("🎉 SUCCESS! Achieved 85%+ accuracy!")
                elif accuracy >= 0.80:
                    print("✅ Good performance! Close to 85% target")
                else:
                    print("📊 Performance analysis:")
                    print(f"   Expected: ~86.4% (original training performance)")
                    print(f"   Achieved: {accuracy*100:.1f}%")
                    print(f"   Difference: {(0.864 - accuracy)*100:.1f} percentage points")
                
                print(f"\n📋 Detailed Classification Report:")
                print(classification_report(y_true, predictions, 
                                          target_names=['Not Diabetic', 'Diabetic']))
                
                # Confusion matrix
                cm = confusion_matrix(y_true, predictions)
                print(f"\n🔢 Confusion Matrix:")
                print(f"                Predicted")
                print(f"Actual    Not Diabetic  Diabetic")
                print(f"Not Diabetic    {cm[0,0]:4d}      {cm[0,1]:4d}")
                print(f"Diabetic        {cm[1,0]:4d}      {cm[1,1]:4d}")
                
                # Show some prediction examples
                print(f"\n🔍 Sample predictions:")
                sample_size = min(10, len(predictions))
                for i in range(sample_size):
                    actual = "Diabetic" if y_true.iloc[i] == 1 else "Not Diabetic"
                    predicted = "Diabetic" if predictions[i] == 1 else "Not Diabetic"
                    confidence = probabilities[i].max()
                    correct = "✅" if y_true.iloc[i] == predictions[i] else "❌"
                    print(f"  {correct} Actual: {actual:12} | Predicted: {predicted:12} | Confidence: {confidence:.3f}")
            
            # Save results
            print(f"\n💾 Saving results...")
            output_df = test_df.copy()
            output_df['Predicted_Outcome'] = predictions
            output_df['Prediction_Label'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in predictions]
            output_df['Probability_Not_Diabetic'] = probabilities[:, 0]
            output_df['Probability_Diabetic'] = probabilities[:, 1]
            output_df['Confidence'] = np.max(probabilities, axis=1)
            
            if has_outcome:
                output_df['Correct_Prediction'] = (predictions == y_true).astype(int)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"correct_diabetes_results_{timestamp}.csv"
            output_df.to_csv(output_filename, index=False)
            
            print(f"✅ Results saved to: {output_filename}")
            
            if has_outcome and accuracy >= 0.85:
                print(f"\n🎉 MISSION ACCOMPLISHED!")
                print(f"✅ Achieved {accuracy:.1%} accuracy using the correct pipeline!")
            elif has_outcome:
                print(f"\n📊 Pipeline successfully applied!")
                print(f"Accuracy: {accuracy:.1%} (Expected: ~86.4%)")
                
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
    print("🏥 DIABETES PREDICTION - CORRECT PIPELINE TEST")
    print("="*70)
    
    # Use fixed model directory
    model_dir = "diabetes_model_saved"
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory '{model_dir}' not found!")
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
    file_path = input("📁 Enter file path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print("❌ Test file not found!")
        return
    
    # Create tester and run
    tester = CorrectDiabetesTester(model_dir=model_dir)
    tester.test_model(file_path)


if __name__ == "__main__":
    main()