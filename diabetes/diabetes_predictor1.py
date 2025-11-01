#!/usr/bin/env python3
"""
Diabetes Prediction System - Clean Implementation
Goal: 80%+ Test Accuracy with proper train/validation/test splits
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Try importing optional libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️ Imbalanced-learn not available. Install with: pip install imbalanced-learn")


class DiabetesPredictionSystem:
    """Clean diabetes prediction system focused on 80%+ test accuracy"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_columns = None
        
    def create_features(self, df):
        """Create robust features that generalize well"""
        df = df.copy()
        
        # Handle missing values (zeros are likely missing)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in df.columns:
                # Replace zeros with median of non-zero values
                non_zero_median = df[df[col] > 0][col].median()
                df[col] = df[col].replace(0, non_zero_median)
        
        # Create meaningful features based on medical knowledge
        # 1. BMI Categories
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                    bins=[0, 18.5, 25, 30, 100], 
                                    labels=[0, 1, 2, 3]).astype(int)
        
        # 2. Glucose Categories (based on diabetes diagnosis thresholds)
        df['Glucose_Category'] = pd.cut(df['Glucose'],
                                        bins=[0, 100, 125, 200],
                                        labels=[0, 1, 2]).astype(int)
        
        # 3. Age Categories
        df['Age_Category'] = pd.cut(df['Age'],
                                   bins=[0, 30, 40, 50, 100],
                                   labels=[0, 1, 2, 3]).astype(int)
        
        # 4. Interaction features
        df['Glucose_BMI_Interaction'] = df['Glucose'] * df['BMI'] / 1000
        df['Age_BMI_Interaction'] = df['Age'] * df['BMI'] / 100
        df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1)
        
        # 5. Risk scores
        df['Diabetes_Risk_Score'] = (
            (df['Glucose'] > 125).astype(int) * 2 +
            (df['BMI'] > 30).astype(int) +
            (df['Age'] > 45).astype(int) +
            (df['BloodPressure'] > 80).astype(int)
        )
        
        # 6. Polynomial features for key indicators
        df['Glucose_Squared'] = df['Glucose'] ** 2 / 10000
        df['BMI_Squared'] = df['BMI'] ** 2 / 100
        
        # 7. Family history impact adjusted by age
        df['Pedigree_Age_Interaction'] = df['DiabetesPedigreeFunction'] * df['Age']
        
        return df
    
    def train_model(self, file_path):
        """Train model with focus on achieving 80%+ test accuracy"""
        print("\n🚀 TRAINING MODE")
        print("="*60)
        
        try:
            # Load data
            print("📂 Loading dataset...")
            df = pd.read_csv(file_path)
            print(f"Dataset shape: {df.shape}")
            print(f"Class distribution:\n{df['Outcome'].value_counts()}")
            
            # Create features
            print("\n🔧 Creating features...")
            df_featured = self.create_features(df)
            
            # Separate features and target
            X = df_featured.drop('Outcome', axis=1)
            y = df_featured['Outcome']
            
            # Store feature columns for later use
            self.feature_columns = X.columns.tolist()
            
            # Split data: 60% train, 20% validation, 20% test
            print("\n📊 Splitting dataset...")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )
            
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Validation set: {X_val.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            
            # Save splits
            print("\n💾 Saving data splits...")
            os.makedirs("data_splits", exist_ok=True)
            
            # Combine features and target for saving
            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            train_data.to_csv("data_splits/train_data.csv", index=False)
            val_data.to_csv("data_splits/validation_data.csv", index=False)
            test_data.to_csv("data_splits/test_data.csv", index=False)
            print("✅ Data splits saved successfully!")
            
            # Feature selection
            print("\n🔍 Selecting best features...")
            self.selector = SelectKBest(score_func=f_classif, k=min(15, X_train.shape[1]))
            X_train_selected = self.selector.fit_transform(X_train, y_train)
            X_val_selected = self.selector.transform(X_val)
            X_test_selected = self.selector.transform(X_test)
            
            # Get selected feature names
            selected_indices = self.selector.get_support(indices=True)
            selected_features = [self.feature_columns[i] for i in selected_indices]
            print(f"Selected {len(selected_features)} features")
            
            # Scaling
            print("\n⚖️ Scaling features...")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_selected)
            X_val_scaled = self.scaler.transform(X_val_selected)
            X_test_scaled = self.scaler.transform(X_test_selected)
            
            # Handle class imbalance if available
            if IMBLEARN_AVAILABLE:
                print("\n⚖️ Balancing classes...")
                try:
                    balancer = SMOTETomek(random_state=42)
                    X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_scaled, y_train)
                    print(f"Balanced training samples: {len(y_train_balanced)}")
                except:
                    X_train_balanced = X_train_scaled
                    y_train_balanced = y_train
            else:
                X_train_balanced = X_train_scaled
                y_train_balanced = y_train
            
            # Train multiple models
            print("\n🤖 Training models...")
            models = {}
            
            # 1. Logistic Regression with balanced weights
            models['lr'] = LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            
            # 2. Random Forest with balanced weights
            models['rf'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            # 3. Gradient Boosting
            models['gb'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
            
            # 4. XGBoost if available
            if XGBOOST_AVAILABLE:
                models['xgb'] = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            
            # 5. LightGBM if available
            if LIGHTGBM_AVAILABLE:
                models['lgb'] = LGBMClassifier(
                    n_estimators=100,
                    num_leaves=31,
                    learning_rate=0.05,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    is_unbalance=True,
                    random_state=42,
                    verbose=-1
                )
            
            # Train and evaluate models
            best_val_score = 0
            best_model_name = None
            model_scores = {}
            
            for name, model in models.items():
                print(f"\nTraining {name.upper()}...")
                
                # Train model
                model.fit(X_train_balanced, y_train_balanced)
                
                # Evaluate on validation set
                val_pred = model.predict(X_val_scaled)
                val_score = accuracy_score(y_val, val_pred)
                
                # Evaluate on training set (to check overfitting)
                train_pred = model.predict(X_train_scaled)
                train_score = accuracy_score(y_train, train_pred)
                
                model_scores[name] = {
                    'model': model,
                    'train_score': train_score,
                    'val_score': val_score,
                    'overfitting_gap': train_score - val_score
                }
                
                print(f"{name.upper()} - Train: {train_score:.4f}, Val: {val_score:.4f}")
                
                # Select best model based on validation score
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model_name = name
            
            # Create ensemble of top models
            print("\n🔗 Creating ensemble...")
            top_models = sorted(model_scores.items(), 
                              key=lambda x: x[1]['val_score'], 
                              reverse=True)[:3]
            
            ensemble_estimators = [(name, scores['model']) for name, scores in top_models]
            
            self.model = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft',
                n_jobs=-1
            )
            
            # Train ensemble
            self.model.fit(X_train_balanced, y_train_balanced)
            
            # Final evaluation on test set
            print("\n📊 Evaluating on test set...")
            test_pred = self.model.predict(X_test_scaled)
            test_score = accuracy_score(y_test, test_pred)
            
            # Also check individual model performances
            print("\n📈 Individual model test scores:")
            for name, scores in model_scores.items():
                model_test_pred = scores['model'].predict(X_test_scaled)
                model_test_score = accuracy_score(y_test, model_test_pred)
                print(f"{name.upper()}: {model_test_score:.4f}")
            
            print(f"\n🎯 ENSEMBLE TEST ACCURACY: {test_score:.4f} ({test_score*100:.2f}%)")
            
            print("\n📊 Test Classification Report:")
            print(classification_report(y_test, test_pred))
            
            # Save model and preprocessors
            print("\n💾 Saving model...")
            os.makedirs("model", exist_ok=True)
            
            joblib.dump(self.model, "model/diabetes_model.pkl")
            joblib.dump(self.scaler, "model/scaler.pkl")
            joblib.dump(self.selector, "model/selector.pkl")
            joblib.dump(self.feature_columns, "model/feature_columns.pkl")
            
            # Save training info
            training_info = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'best_model': best_model_name,
                'ensemble_models': [name for name, _ in ensemble_estimators],
                'test_accuracy': test_score,
                'selected_features': selected_features
            }
            joblib.dump(training_info, "model/training_info.pkl")
            
            print("✅ Model saved successfully!")
            print("\n🎉 TRAINING COMPLETED!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model(self, file_path):
        """Test the trained model on new data"""
        print("\n🔍 TESTING MODE")
        print("="*60)
        
        try:
            # Load model and preprocessors
            print("📂 Loading model...")
            self.model = joblib.load("model/diabetes_model.pkl")
            self.scaler = joblib.load("model/scaler.pkl")
            self.selector = joblib.load("model/selector.pkl")
            self.feature_columns = joblib.load("model/feature_columns.pkl")
            training_info = joblib.load("model/training_info.pkl")
            
            print(f"✅ Model loaded successfully!")
            print(f"Training test accuracy was: {training_info['test_accuracy']:.4f}")
            
            # Load test data
            print("\n📂 Loading test data...")
            test_df = pd.read_csv(file_path)
            print(f"Test data shape: {test_df.shape}")
            
            # Check if outcome column exists
            has_outcome = 'Outcome' in test_df.columns
            if has_outcome:
                y_true = test_df['Outcome']
                X_test = test_df.drop('Outcome', axis=1)
            else:
                X_test = test_df
            
            # Create features
            print("🔧 Creating features...")
            X_test_featured = self.create_features(X_test)
            
            # Ensure all required columns exist
            missing_cols = set(self.feature_columns) - set(X_test_featured.columns)
            if missing_cols:
                print(f"⚠️ Adding missing columns: {missing_cols}")
                for col in missing_cols:
                    X_test_featured[col] = 0
            
            # Select columns in the same order as training
            X_test_featured = X_test_featured[self.feature_columns]
            
            # Apply feature selection and scaling
            X_test_selected = self.selector.transform(X_test_featured)
            X_test_scaled = self.scaler.transform(X_test_selected)
            
            # Make predictions
            print("\n🔮 Making predictions...")
            predictions = self.model.predict(X_test_scaled)
            probabilities = self.model.predict_proba(X_test_scaled)
            
            # Create output dataframe
            output_df = test_df.copy()
            output_df['Predicted_Outcome'] = predictions
            output_df['Prediction_Label'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in predictions]
            output_df['Probability_Not_Diabetic'] = probabilities[:, 0]
            output_df['Probability_Diabetic'] = probabilities[:, 1]
            output_df['Confidence'] = np.max(probabilities, axis=1)
            
            # If we have true outcomes, calculate accuracy
            if has_outcome:
                accuracy = accuracy_score(y_true, predictions)
                f1 = f1_score(y_true, predictions)
                
                print(f"\n📈 TEST RESULTS:")
                print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"F1 Score: {f1:.4f}")
                
                print("\n📊 Classification Report:")
                print(classification_report(y_true, predictions))
                
                # Add correct prediction column
                output_df['Correct_Prediction'] = (predictions == y_true).astype(int)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"test_results_{timestamp}.csv"
            output_df.to_csv(output_filename, index=False)
            
            print(f"\n✅ Results saved to: {output_filename}")
            print("\n🎉 TESTING COMPLETED!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during testing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function"""
    print("\n" + "="*60)
    print("🏥 DIABETES PREDICTION SYSTEM")
    print("   Target: 80%+ Test Accuracy")
    print("="*60)
    
    system = DiabetesPredictionSystem()
    
    # Check if model exists
    model_exists = os.path.exists("model/diabetes_model.pkl")
    
    # Get file path
    print(f"\n📊 Model exists: {'Yes' if model_exists else 'No'}")
    print("\n📂 Please provide your CSV file path:")
    file_path = input("📁 Enter file path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print("❌ File not found!")
        return
    
    # Check if it's a full dataset or test file
    df = pd.read_csv(file_path)
    is_full_dataset = 'Outcome' in df.columns and len(df) > 200
    
    if is_full_dataset and not model_exists:
        # Train mode
        print("\n➡️ Full dataset detected. Starting training...")
        system.train_model(file_path)
    elif not is_full_dataset or (model_exists and len(df) < 200):
        # Test mode
        if not model_exists:
            print("❌ No trained model found! Please train first with a full dataset.")
            return
        print("\n➡️ Test file detected. Starting testing...")
        system.test_model(file_path)
    elif is_full_dataset and model_exists:
        # Ask user what to do
        print("\n⚠️ Full dataset detected but model already exists.")
        choice = input("Do you want to retrain? (y/n): ").lower()
        if choice == 'y':
            print("\n➡️ Starting retraining...")
            system.train_model(file_path)
        else:
            print("\n➡️ Starting testing...")
            system.test_model(file_path)
    else:
        print("❌ Unable to determine operation mode.")


if __name__ == "__main__":
    main()