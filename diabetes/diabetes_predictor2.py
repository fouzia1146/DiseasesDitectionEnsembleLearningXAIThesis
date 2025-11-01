#!/usr/bin/env python3
"""
Diabetes Prediction System - Anti-Overfitting Version
Designed to achieve 85%+ test accuracy by preventing overfitting
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.decomposition import PCA
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
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available. Install with: pip install catboost")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️ Imbalanced-learn not available. Install with: pip install imbalanced-learn")


class DiabetesPredictionSystem:
    """Anti-overfitting system for 85%+ test accuracy"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_columns = None
        self.pca = None
        
    def create_features(self, df):
        """Create features with focus on generalization"""
        df = df.copy()
        
        # Handle missing values (zeros are likely missing)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in df.columns:
                # Use median of non-zero values
                non_zero_median = df[df[col] > 0][col].median()
                df[col] = df[col].replace(0, non_zero_median)
        
        # 1. Essential medical features only
        # Glucose categories based on medical thresholds
        df['PreDiabetic'] = ((df['Glucose'] >= 100) & (df['Glucose'] < 126)).astype(int)
        df['Diabetic_Glucose'] = (df['Glucose'] >= 126).astype(int)
        
        # BMI categories
        df['Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
        df['Obese'] = (df['BMI'] >= 30).astype(int)
        
        # Age risk
        df['Age_Risk'] = (df['Age'] > 45).astype(int)
        
        # 2. Key interaction features (fewer to prevent overfitting)
        df['Glucose_BMI'] = df['Glucose'] * df['BMI'] / 1000
        df['Age_BMI'] = df['Age'] * df['BMI'] / 100
        df['Glucose_Age'] = df['Glucose'] * df['Age'] / 1000
        
        # 3. Important ratios
        epsilon = 1e-8
        df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + epsilon)
        df['BMI_Age_Ratio'] = df['BMI'] / (df['Age'] + epsilon)
        
        # 4. Risk scores (simplified)
        df['Metabolic_Risk'] = (
            df['Diabetic_Glucose'] * 2 +
            df['Obese'] +
            df['Age_Risk'] +
            (df['BloodPressure'] > 80).astype(int)
        )
        
        # 5. Polynomial features (only most important)
        df['Glucose_Squared'] = df['Glucose'] ** 2 / 10000
        df['BMI_Squared'] = df['BMI'] ** 2 / 1000
        
        # 6. Log transformations (key features only)
        df['Insulin_Log'] = np.log1p(df['Insulin'])
        df['DiabetesPedigree_Log'] = np.log1p(df['DiabetesPedigreeFunction'])
        
        # 7. Family history impact
        df['Strong_Family_History'] = (df['DiabetesPedigreeFunction'] > 0.5).astype(int)
        df['Pedigree_Age'] = df['DiabetesPedigreeFunction'] * df['Age']
        
        # 8. Pregnancy risk
        df['Multiple_Pregnancies'] = (df['Pregnancies'] > 3).astype(int)
        
        # 9. Statistical features (simplified)
        key_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
        df['Key_Features_Mean'] = df[key_features].mean(axis=1)
        df['Key_Features_Std'] = df[key_features].std(axis=1)
        
        return df
    
    def train_model(self, file_path):
        """Train with strong regularization to prevent overfitting"""
        print("\n🚀 TRAINING MODE - Anti-Overfitting for 85%+ Accuracy")
        print("="*60)
        
        try:
            # Load data
            print("📂 Loading dataset...")
            df = pd.read_csv(file_path)
            print(f"Dataset shape: {df.shape}")
            print(f"Class distribution:\n{df['Outcome'].value_counts()}")
            
            # Create features
            print("\n🔧 Creating features (simplified to prevent overfitting)...")
            df_featured = self.create_features(df)
            print(f"Total features created: {df_featured.shape[1] - 1}")
            
            # Separate features and target
            X = df_featured.drop('Outcome', axis=1)
            y = df_featured['Outcome']
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Split data with different random state for better generalization
            print("\n📊 Splitting dataset...")
            # Try different random states to find best split
            best_split_score = 0
            best_random_state = 42
            
            for rs in [42, 123, 456, 789, 2024]:
                X_temp, X_test_temp, y_temp, y_test_temp = train_test_split(
                    X, y, test_size=0.20, random_state=rs, stratify=y
                )
                # Check if split maintains class balance
                test_ratio = y_test_temp.sum() / len(y_test_temp)
                overall_ratio = y.sum() / len(y)
                if abs(test_ratio - overall_ratio) < 0.02:  # Good balance
                    best_random_state = rs
                    break
            
            print(f"Using random_state={best_random_state} for better splits")
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.20, random_state=best_random_state, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=best_random_state, stratify=y_temp
            )
            
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Validation set: {X_val.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            
            # Save splits
            print("\n💾 Saving data splits...")
            os.makedirs("data_splits", exist_ok=True)
            
            train_indices = X_train.index
            val_indices = X_val.index
            test_indices = X_test.index
            
            df.loc[train_indices].to_csv("data_splits/train_data.csv", index=False)
            df.loc[val_indices].to_csv("data_splits/validation_data.csv", index=False)
            df.loc[test_indices].to_csv("data_splits/test_data.csv", index=False)
            print("✅ Data splits saved!")
            
            # Feature selection with cross-validation
            print("\n🔍 Selecting features with RFECV...")
            # Use RFECV for better feature selection
            estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            self.selector = RFECV(estimator, cv=5, scoring='accuracy', n_jobs=-1)
            X_train_selected = self.selector.fit_transform(X_train, y_train)
            X_val_selected = self.selector.transform(X_val)
            X_test_selected = self.selector.transform(X_test)
            
            print(f"Optimal features selected: {X_train_selected.shape[1]}")
            
            # Apply PCA for dimensionality reduction if too many features
            if X_train_selected.shape[1] > 20:
                print("\n📉 Applying PCA to reduce dimensionality...")
                self.pca = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
                X_train_pca = self.pca.fit_transform(X_train_selected)
                X_val_pca = self.pca.transform(X_val_selected)
                X_test_pca = self.pca.transform(X_test_selected)
                print(f"PCA components: {X_train_pca.shape[1]}")
            else:
                X_train_pca = X_train_selected
                X_val_pca = X_val_selected
                X_test_pca = X_test_selected
                self.pca = None
            
            # Use RobustScaler for better handling of outliers
            print("\n⚖️ Scaling features with RobustScaler...")
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_pca)
            X_val_scaled = self.scaler.transform(X_val_pca)
            X_test_scaled = self.scaler.transform(X_test_pca)
            
            # Simple SMOTE for class balancing (not SMOTETomek which can overfit)
            if IMBLEARN_AVAILABLE:
                print("\n⚖️ Balancing classes with SMOTE...")
                try:
                    balancer = SMOTE(random_state=42, k_neighbors=5)
                    X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_scaled, y_train)
                    print(f"Balanced samples: {len(y_train_balanced)}")
                except:
                    X_train_balanced = X_train_scaled
                    y_train_balanced = y_train
            else:
                X_train_balanced = X_train_scaled
                y_train_balanced = y_train
            
            # Train REGULARIZED models
            print("\n🤖 Training heavily regularized models...")
            models = {}
            
            # 1. Logistic Regression with strong regularization
            models['lr'] = LogisticRegression(
                C=0.01,  # Very strong regularization
                class_weight='balanced',
                penalty='l2',
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
            
            # 2. Random Forest with constraints
            models['rf'] = RandomForestClassifier(
                n_estimators=200,  # Not too many
                max_depth=6,       # Shallow trees
                min_samples_split=20,  # High minimum
                min_samples_leaf=10,   # High minimum
                max_features='sqrt',
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            
            # 3. Extra Trees with constraints
            models['et'] = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            
            # 4. Gradient Boosting with early stopping
            models['gb'] = GradientBoostingClassifier(
                n_estimators=100,  # Fewer trees
                learning_rate=0.01,  # Very slow learning
                max_depth=4,       # Shallow
                subsample=0.7,     # More regularization
                max_features='sqrt',
                random_state=42
            )
            
            # 5. XGBoost with heavy regularization
            if XGBOOST_AVAILABLE:
                models['xgb'] = XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.01,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=1.0,    # L1 regularization
                    reg_lambda=2.0,   # L2 regularization
                    gamma=1.0,        # Minimum loss reduction
                    min_child_weight=5,
                    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            
            # 6. LightGBM with regularization
            if LIGHTGBM_AVAILABLE:
                models['lgb'] = LGBMClassifier(
                    n_estimators=100,
                    num_leaves=20,    # Fewer leaves
                    learning_rate=0.01,
                    feature_fraction=0.7,
                    bagging_fraction=0.7,
                    bagging_freq=5,
                    min_child_samples=20,
                    min_split_gain=0.1,
                    reg_alpha=1.0,
                    reg_lambda=2.0,
                    is_unbalance=True,
                    random_state=42,
                    verbose=-1
                )
            
            # Train and evaluate
            model_scores = {}
            
            for name, model in models.items():
                print(f"\nTraining {name.upper()}...")
                
                # Use cross-validation for better evaluation
                cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                          cv=5, scoring='accuracy')
                print(f"{name} CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
                # Train on full training set
                model.fit(X_train_balanced, y_train_balanced)
                
                # Evaluate
                val_pred = model.predict(X_val_scaled)
                val_score = accuracy_score(y_val, val_pred)
                
                train_pred = model.predict(X_train_scaled)
                train_score = accuracy_score(y_train, train_pred)
                
                model_scores[name] = {
                    'model': model,
                    'cv_score': cv_scores.mean(),
                    'train_score': train_score,
                    'val_score': val_score,
                    'overfitting_gap': train_score - val_score
                }
                
                print(f"{name.upper()} - Train: {train_score:.4f}, Val: {val_score:.4f}, Gap: {train_score - val_score:.4f}")
            
            # Select models with low overfitting
            print("\n🔗 Selecting models with low overfitting...")
            good_models = [(name, scores) for name, scores in model_scores.items() 
                          if scores['overfitting_gap'] < 0.10]
            
            if len(good_models) < 3:
                # If not enough low-overfitting models, take best by validation score
                good_models = sorted(model_scores.items(), 
                                   key=lambda x: x[1]['val_score'], 
                                   reverse=True)[:5]
            else:
                # Sort good models by validation score
                good_models = sorted(good_models, 
                                   key=lambda x: x[1]['val_score'], 
                                   reverse=True)[:5]
            
            print(f"Selected {len(good_models)} models for ensemble")
            
            # Create ensemble
            ensemble_estimators = [(name, scores['model']) for name, scores in good_models]
            
            self.model = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft',
                n_jobs=-1
            )
            
            # Train ensemble
            self.model.fit(X_train_balanced, y_train_balanced)
            
            # Final evaluation
            print("\n📊 Final evaluation on test set...")
            test_pred = self.model.predict(X_test_scaled)
            test_score = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred)
            
            # Individual model test scores
            print("\n📈 Individual model test scores:")
            for name, scores in model_scores.items():
                model_test_pred = scores['model'].predict(X_test_scaled)
                model_test_score = accuracy_score(y_test, model_test_pred)
                print(f"{name.upper()}: {model_test_score:.4f}")
            
            print(f"\n🎯 ENSEMBLE TEST ACCURACY: {test_score:.4f} ({test_score*100:.2f}%)")
            print(f"F1 Score: {test_f1:.4f}")
            
            if test_score >= 0.85:
                print("\n🎉 SUCCESS! Achieved 85%+ accuracy!")
            else:
                print(f"\n📊 Current accuracy: {test_score:.4f}")
                print("💡 The model is optimized to prevent overfitting")
            
            print("\n📊 Test Classification Report:")
            print(classification_report(y_test, test_pred))
            
            # Save model
            print("\n💾 Saving model...")
            os.makedirs("model", exist_ok=True)
            
            joblib.dump(self.model, "model/diabetes_model.pkl")
            joblib.dump(self.scaler, "model/scaler.pkl")
            joblib.dump(self.selector, "model/selector.pkl")
            joblib.dump(self.feature_columns, "model/feature_columns.pkl")
            if self.pca:
                joblib.dump(self.pca, "model/pca.pkl")
            
            # Save training info
            training_info = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'ensemble_models': [name for name, _ in ensemble_estimators],
                'test_accuracy': test_score,
                'test_f1': test_f1,
                'random_state': best_random_state
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
        """Test the trained model"""
        print("\n🔍 TESTING MODE")
        print("="*60)
        
        try:
            # Load model
            print("📂 Loading model...")
            self.model = joblib.load("model/diabetes_model.pkl")
            self.scaler = joblib.load("model/scaler.pkl")
            self.selector = joblib.load("model/selector.pkl")
            self.feature_columns = joblib.load("model/feature_columns.pkl")
            training_info = joblib.load("model/training_info.pkl")
            
            # Load PCA if exists
            if os.path.exists("model/pca.pkl"):
                self.pca = joblib.load("model/pca.pkl")
            else:
                self.pca = None
            
            print(f"✅ Model loaded!")
            print(f"Training test accuracy: {training_info['test_accuracy']:.4f}")
            
            # Load test data
            print("\n📂 Loading test data...")
            test_df = pd.read_csv(file_path)
            print(f"Test data shape: {test_df.shape}")
            
            # Check for outcome
            has_outcome = 'Outcome' in test_df.columns
            if has_outcome:
                y_true = test_df['Outcome']
                X_test = test_df.drop('Outcome', axis=1)
            else:
                X_test = test_df
            
            # Create features
            print("🔧 Creating features...")
            X_test_featured = self.create_features(X_test)
            
            # Ensure all columns exist
            for col in self.feature_columns:
                if col not in X_test_featured.columns:
                    X_test_featured[col] = 0
            
            X_test_featured = X_test_featured[self.feature_columns]
            
            # Apply transformations
            X_test_selected = self.selector.transform(X_test_featured)
            
            if self.pca:
                X_test_pca = self.pca.transform(X_test_selected)
                X_test_scaled = self.scaler.transform(X_test_pca)
            else:
                X_test_scaled = self.scaler.transform(X_test_selected)
            
            # Make predictions
            print("\n🔮 Making predictions...")
            predictions = self.model.predict(X_test_scaled)
            probabilities = self.model.predict_proba(X_test_scaled)
            
            # Create output
            output_df = test_df.copy()
            output_df['Predicted_Outcome'] = predictions
            output_df['Prediction_Label'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in predictions]
            output_df['Probability_Not_Diabetic'] = probabilities[:, 0]
            output_df['Probability_Diabetic'] = probabilities[:, 1]
            output_df['Confidence'] = np.max(probabilities, axis=1)
            
            # Calculate accuracy if outcomes available
            if has_outcome:
                accuracy = accuracy_score(y_true, predictions)
                f1 = f1_score(y_true, predictions)
                
                print(f"\n📈 TEST RESULTS:")
                print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"F1 Score: {f1:.4f}")
                
                if accuracy >= 0.85:
                    print("\n🎉 SUCCESS! Achieved 85%+ accuracy!")
                
                print("\n📊 Classification Report:")
                print(classification_report(y_true, predictions))
                
                output_df['Correct_Prediction'] = (predictions == y_true).astype(int)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"test_results_{timestamp}.csv"
            output_df.to_csv(output_filename, index=False)
            
            print(f"\n✅ Results saved to: {output_filename}")
            print("\n🎉 TESTING COMPLETED!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function"""
    print("\n" + "="*70)
    print("🏥 DIABETES PREDICTION - ANTI-OVERFITTING FOR 85%+ ACCURACY")
    print("="*70)
    
    system = DiabetesPredictionSystem()
    
    # Check if model exists
    model_exists = os.path.exists("model/diabetes_model.pkl")
    
    print(f"\n📊 Model exists: {'Yes' if model_exists else 'No'}")
    print("\n📂 Please provide your CSV file path:")
    file_path = input("📁 Enter file path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print("❌ File not found!")
        return
    
    # Detect operation mode
    df = pd.read_csv(file_path)
    is_full_dataset = 'Outcome' in df.columns and len(df) > 200
    
    if is_full_dataset and not model_exists:
        print("\n➡️ Starting training...")
        system.train_model(file_path)
    elif not is_full_dataset or (model_exists and len(df) < 200):
        if not model_exists:
            print("❌ No model found! Train first.")
            return
        print("\n➡️ Starting testing...")
        system.test_model(file_path)
    elif is_full_dataset and model_exists:
        choice = input("\nRetrain model? (y/n): ").lower()
        if choice == 'y':
            print("\n➡️ Retraining...")
            system.train_model(file_path)
        else:
            print("\n➡️ Testing...")
            system.test_model(file_path)


if __name__ == "__main__":
    main()