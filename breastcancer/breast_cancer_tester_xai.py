#!/usr/bin/env python3
"""
Correct Breast Cancer Tester with SHAP and LIME explanations
This should achieve 95%+ accuracy by matching the original pipeline
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not installed. Install with: pip install shap")
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️ LIME not installed. Install with: pip install lime")
    LIME_AVAILABLE = False

def ultra_feature_engineering(df):
    """EXACT COPY of the feature engineering from training code"""
    df_new = df.copy()
    target = "Diagnosis"
    
    # Drop ID column if exists
    if 'ID' in df_new.columns:
        df_new = df_new.drop(columns=['ID'])
    
    # Get feature groups
    mean_features = [col for col in df_new.columns if '_mean' in col]
    se_features = [col for col in df_new.columns if '_se' in col]
    worst_features = [col for col in df_new.columns if '_worst' in col]
    
    # ==================== POLYNOMIAL FEATURES ====================
    # Square features for key predictors
    for feature in ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                    'concavity_mean', 'concave_points_mean']:
        if feature in df_new.columns:
            df_new[f'{feature}_squared'] = df_new[feature] ** 2
            df_new[f'{feature}_cubed'] = df_new[feature] ** 3
    
    # Square worst features
    for feature in ['radius_worst', 'area_worst', 'perimeter_worst', 'concavity_worst']:
        if feature in df_new.columns:
            df_new[f'{feature}_squared'] = df_new[feature] ** 2
    
    # ==================== INTERACTION FEATURES ====================
    # Area and radius relationships
    if 'radius_mean' in df_new.columns and 'area_mean' in df_new.columns:
        df_new['radius_area_mean'] = df_new['radius_mean'] * df_new['area_mean']
        df_new['radius_area_ratio'] = df_new['radius_mean'] / (df_new['area_mean'] + 1e-8)
    
    # Perimeter and area relationships
    if 'perimeter_mean' in df_new.columns and 'area_mean' in df_new.columns:
        df_new['perimeter_area_mean'] = df_new['perimeter_mean'] * df_new['area_mean']
        df_new['perimeter_area_ratio'] = df_new['perimeter_mean'] / (df_new['area_mean'] + 1e-8)
    
    # Texture interactions
    if 'texture_mean' in df_new.columns and 'radius_mean' in df_new.columns:
        df_new['texture_radius_mean'] = df_new['texture_mean'] * df_new['radius_mean']
    
    # Concavity interactions
    if 'concavity_mean' in df_new.columns and 'concave_points_mean' in df_new.columns:
        df_new['concavity_points_mean'] = df_new['concavity_mean'] * df_new['concave_points_mean']
        df_new['concavity_points_ratio'] = df_new['concavity_mean'] / (df_new['concave_points_mean'] + 1e-8)
    
    # Compactness interactions
    if 'compactness_mean' in df_new.columns and 'area_mean' in df_new.columns:
        df_new['compactness_area_mean'] = df_new['compactness_mean'] * df_new['area_mean']
    
    # Smoothness interactions
    if 'smoothness_mean' in df_new.columns and 'symmetry_mean' in df_new.columns:
        df_new['smoothness_symmetry_mean'] = df_new['smoothness_mean'] * df_new['symmetry_mean']
    
    # Worst features interactions
    if 'radius_worst' in df_new.columns and 'texture_worst' in df_new.columns:
        df_new['radius_texture_worst'] = df_new['radius_worst'] * df_new['texture_worst']
    
    if 'area_worst' in df_new.columns and 'concavity_worst' in df_new.columns:
        df_new['area_concavity_worst'] = df_new['area_worst'] * df_new['concavity_worst']
    
    # Triple interactions
    if all(col in df_new.columns for col in ['radius_mean', 'texture_mean', 'area_mean']):
        df_new['radius_texture_area_mean'] = df_new['radius_mean'] * df_new['texture_mean'] * df_new['area_mean']
    
    if all(col in df_new.columns for col in ['concavity_mean', 'concave_points_mean', 'area_mean']):
        df_new['concavity_points_area_mean'] = df_new['concavity_mean'] * df_new['concave_points_mean'] * df_new['area_mean']
    
    # ==================== RATIO FEATURES ====================
    epsilon = 1e-8
    
    # Mean to worst ratios
    if 'radius_mean' in df_new.columns and 'radius_worst' in df_new.columns:
        df_new['radius_mean_worst_ratio'] = df_new['radius_mean'] / (df_new['radius_worst'] + epsilon)
    
    if 'area_mean' in df_new.columns and 'area_worst' in df_new.columns:
        df_new['area_mean_worst_ratio'] = df_new['area_mean'] / (df_new['area_worst'] + epsilon)
    
    if 'concavity_mean' in df_new.columns and 'concavity_worst' in df_new.columns:
        df_new['concavity_mean_worst_ratio'] = df_new['concavity_mean'] / (df_new['concavity_worst'] + epsilon)
    
    # SE to mean ratios (variability indicators)
    if 'radius_se' in df_new.columns and 'radius_mean' in df_new.columns:
        df_new['radius_variability'] = df_new['radius_se'] / (df_new['radius_mean'] + epsilon)
    
    if 'area_se' in df_new.columns and 'area_mean' in df_new.columns:
        df_new['area_variability'] = df_new['area_se'] / (df_new['area_mean'] + epsilon)
    
    if 'texture_se' in df_new.columns and 'texture_mean' in df_new.columns:
        df_new['texture_variability'] = df_new['texture_se'] / (df_new['texture_mean'] + epsilon)
    
    # ==================== COMPOSITE SCORES ====================
    # Tumor size score
    if all(col in df_new.columns for col in ['radius_mean', 'area_mean', 'perimeter_mean']):
        df_new['tumor_size_score'] = (df_new['radius_mean'] * 0.3 + 
                                      df_new['area_mean'] * 0.4 + 
                                      df_new['perimeter_mean'] * 0.3)
    
    # Tumor irregularity score
    if all(col in df_new.columns for col in ['concavity_mean', 'concave_points_mean', 'compactness_mean']):
        df_new['tumor_irregularity_score'] = (df_new['concavity_mean'] * 0.4 + 
                                              df_new['concave_points_mean'] * 0.4 + 
                                              df_new['compactness_mean'] * 0.2)
    
    # Texture complexity score
    if all(col in df_new.columns for col in ['texture_mean', 'smoothness_mean', 'symmetry_mean']):
        df_new['texture_complexity_score'] = (df_new['texture_mean'] * 0.4 + 
                                             df_new['smoothness_mean'] * 0.3 + 
                                             df_new['symmetry_mean'] * 0.3)
    
    # Malignancy risk score
    if all(col in df_new.columns for col in ['radius_worst', 'concavity_worst', 'area_worst']):
        df_new['malignancy_risk_score'] = (df_new['radius_worst'] * 0.3 + 
                                          df_new['concavity_worst'] * 0.4 + 
                                          df_new['area_worst'] * 0.3)
    
    # ==================== LOG TRANSFORMATIONS ====================
    # Log transform for skewed features
    for feature in ['area_mean', 'area_se', 'area_worst']:
        if feature in df_new.columns:
            df_new[f'{feature}_log'] = np.log1p(df_new[feature])
    
    for feature in ['concavity_mean', 'concave_points_mean']:
        if feature in df_new.columns:
            df_new[f'{feature}_log'] = np.log1p(df_new[feature])
    
    # ==================== SQRT TRANSFORMATIONS ====================
    for feature in ['radius_mean', 'perimeter_mean', 'texture_mean']:
        if feature in df_new.columns:
            df_new[f'{feature}_sqrt'] = np.sqrt(df_new[feature])
    
    # ==================== STATISTICAL FEATURES ====================
    # Statistics across mean features
    if mean_features:
        df_new['mean_features_sum'] = df_new[mean_features].sum(axis=1)
        df_new['mean_features_mean'] = df_new[mean_features].mean(axis=1)
        df_new['mean_features_std'] = df_new[mean_features].std(axis=1)
        df_new['mean_features_max'] = df_new[mean_features].max(axis=1)
        df_new['mean_features_min'] = df_new[mean_features].min(axis=1)
        df_new['mean_features_range'] = df_new['mean_features_max'] - df_new['mean_features_min']
    
    # Statistics across worst features
    if worst_features:
        df_new['worst_features_sum'] = df_new[worst_features].sum(axis=1)
        df_new['worst_features_mean'] = df_new[worst_features].mean(axis=1)
        df_new['worst_features_std'] = df_new[worst_features].std(axis=1)
        df_new['worst_features_max'] = df_new[worst_features].max(axis=1)
        df_new['worst_features_min'] = df_new[worst_features].min(axis=1)
    
    # Statistics across SE features (variability)
    if se_features:
        df_new['se_features_sum'] = df_new[se_features].sum(axis=1)
        df_new['se_features_mean'] = df_new[se_features].mean(axis=1)
        df_new['se_features_std'] = df_new[se_features].std(axis=1)
    
    # ==================== CROSS-GROUP FEATURES ====================
    # Mean to worst differences
    if 'radius_mean' in df_new.columns and 'radius_worst' in df_new.columns:
        df_new['radius_mean_worst_diff'] = df_new['radius_worst'] - df_new['radius_mean']
    
    if 'area_mean' in df_new.columns and 'area_worst' in df_new.columns:
        df_new['area_mean_worst_diff'] = df_new['area_worst'] - df_new['area_mean']
    
    if 'concavity_mean' in df_new.columns and 'concavity_worst' in df_new.columns:
        df_new['concavity_mean_worst_diff'] = df_new['concavity_worst'] - df_new['concavity_mean']
    
    # ==================== CATEGORICAL FEATURES ====================
    # Binning based on medical thresholds
    if 'radius_mean' in df_new.columns:
        df_new['radius_category'] = pd.cut(df_new['radius_mean'],
                                          bins=[0, 10, 12, 14, 16, np.inf],
                                          labels=[0, 1, 2, 3, 4],
                                          include_lowest=True).astype(float)
    
    if 'area_mean' in df_new.columns:
        df_new['area_category'] = pd.cut(df_new['area_mean'],
                                        bins=[0, 300, 500, 700, 900, np.inf],
                                        labels=[0, 1, 2, 3, 4],
                                        include_lowest=True).astype(float)
    
    if 'concavity_mean' in df_new.columns:
        df_new['concavity_category'] = pd.cut(df_new['concavity_mean'],
                                             bins=[0, 0.05, 0.1, 0.15, 0.2, np.inf],
                                             labels=[0, 1, 2, 3, 4],
                                             include_lowest=True).astype(float)
    
    # Fill any NaN values that might have been created
    df_new.fillna(0, inplace=True)
    
    # Replace infinite values
    df_new.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df_new


class CorrectBreastCancerTester:
    """Breast Cancer tester using the EXACT same pipeline as training with SHAP and LIME"""
    
    def __init__(self, model_dir="breast_cancer_model_saved"):
        self.model = None
        self.scaler = None
        self.variance_selector = None
        self.stat_selector = None
        self.model_selector = None
        self.label_encoder = None
        self.model_dir = model_dir
        self.feature_names = None
        self.X_train_scaled = None  # For SHAP and LIME
        
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
            
            # Load label encoder
            self.label_encoder = joblib.load(f"{self.model_dir}/label_encoder.pkl")
            
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
    
    def generate_shap_explanations(self, X_scaled, predictions):
        """Generate SHAP explanations - importance and summary plots only"""
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available. Skipping SHAP explanations.")
            return None
        
        print("\n🔮 Generating SHAP explanations...")
        
        try:
            # Create SHAP explainer based on model type
            model_type = type(self.model).__name__
            
            if 'Tree' in model_type or 'Forest' in model_type or 'XGB' in model_type:
                # Tree-based models
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_scaled)
                
                # For binary classification, take values for positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                # Use KernelExplainer for other models
                # Sample background data for efficiency
                background_data = shap.sample(X_scaled, min(100, len(X_scaled)))
                explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x)[:, 1],
                    background_data
                )
                
                # Calculate SHAP values for a subset
                sample_indices = np.random.choice(len(X_scaled), 
                                                min(50, len(X_scaled)), 
                                                replace=False)
                shap_values = explainer.shap_values(X_scaled[sample_indices])
                X_scaled = X_scaled[sample_indices]
            
            # Create visualizations
            print("📊 Creating SHAP visualizations...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Summary plot (beeswarm plot)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_scaled, 
                            feature_names=self.feature_names,
                            show=False)
            plt.title("SHAP Summary Plot - Feature Impact on Model Output")
            plt.tight_layout()
            plt.savefig(f'shap_summary_{timestamp}.png', dpi=100, bbox_inches='tight')
            plt.show()
            
            # 2. Feature importance bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_scaled, 
                            feature_names=self.feature_names,
                            plot_type="bar", show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig(f'shap_importance_{timestamp}.png', dpi=100, bbox_inches='tight')
            plt.show()
            
            print("✅ SHAP explanations generated successfully!")
            return shap_values
            
        except Exception as e:
            print(f"❌ Error generating SHAP explanations: {str(e)}")
            return None
    
    def generate_lime_explanations(self, X_scaled, predictions, num_samples=5):
        """Generate LIME explanations - top 15 features summary only"""
        if not LIME_AVAILABLE:
            print("⚠️ LIME not available. Skipping LIME explanations.")
            return None
        
        print("\n🍋 Generating LIME explanations...")
        
        try:
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_scaled,
                feature_names=self.feature_names,
                class_names=['Benign', 'Malignant'],
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            
            # Generate explanations for selected samples
            lime_explanations = []
            sample_indices = np.random.choice(len(X_scaled), 
                                            min(num_samples, len(X_scaled)), 
                                            replace=False)
            
            print(f"Analyzing {len(sample_indices)} samples for LIME explanations...")
            
            for idx in sample_indices:
                # Get explanation
                exp = explainer.explain_instance(
                    X_scaled[idx],
                    self.model.predict_proba,
                    num_features=15,  # Top 15 features
                    num_samples=5000
                )
                lime_explanations.append(exp)
            
            # Create summary visualization of top 15 features
            if lime_explanations:
                # Extract feature importances
                all_features = {}
                for exp in lime_explanations:
                    for feature, importance in exp.as_list():
                        feature_name = feature.split(' ')[0]  # Get base feature name
                        if feature_name not in all_features:
                            all_features[feature_name] = []
                        all_features[feature_name].append(abs(importance))
                
                # Calculate average importance
                avg_importance = {k: np.mean(v) for k, v in all_features.items()}
                
                # Sort and get top 15 features
                top_features = dict(sorted(avg_importance.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:15])
                
                # Create visualization
                plt.figure(figsize=(10, 8))
                features = list(top_features.keys())
                importances = list(top_features.values())
                
                # Create horizontal bar plot
                plt.barh(range(len(features)), importances, color='skyblue')
                plt.yticks(range(len(features)), features)
                plt.xlabel('Average Absolute Importance')
                plt.title('LIME - Top 15 Most Important Features')
                plt.gca().invert_yaxis()  # Highest importance at top
                
                # Add value labels on bars
                for i, v in enumerate(importances):
                    plt.text(v, i, f' {v:.4f}', va='center')
                
                plt.tight_layout()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(f'lime_top15_features_{timestamp}.png', dpi=100, bbox_inches='tight')
                plt.show()
                
                # Print text summary
                print("\n📝 LIME Top 15 Most Important Features:")
                for i, (feature, importance) in enumerate(top_features.items(), 1):
                    print(f"   {i:2d}. {feature:30s}: {importance:.4f}")
            
            print("✅ LIME explanations generated successfully!")
            return lime_explanations
            
        except Exception as e:
            print(f"❌ Error generating LIME explanations: {str(e)}")
            return None
    
    def test_model(self, file_path):
        """Test the model using the EXACT same preprocessing pipeline with explanations"""
        print("\n🔍 TESTING MODE - Using Original Training Pipeline")
        print("="*60)
        
        if not self.load_model():
            return False
        
        try:
            # Load test data
            print(f"\n📂 Loading test data from: {file_path}")
            test_df = pd.read_csv(file_path)
            print(f"Test data shape: {test_df.shape}")
            
            # Check for diagnosis column
            has_diagnosis = 'Diagnosis' in test_df.columns
            if has_diagnosis:
                print("✅ Ground truth labels found - will calculate accuracy")
                y_true_raw = test_df['Diagnosis']
                
                # Encode if it's M/B format
                if y_true_raw.dtype == 'object':
                    y_true = self.label_encoder.transform(y_true_raw)
                    print(f"Encoded labels: M={self.label_encoder.transform(['M'])[0]}, B={self.label_encoder.transform(['B'])[0]}")
                else:
                    y_true = y_true_raw.values
                
                X_test = test_df.drop('Diagnosis', axis=1)
                if 'ID' in X_test.columns:
                    X_test = X_test.drop('ID', axis=1)
                
                print(f"Class distribution in test data:")
                print(f"Benign (0): {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true)*100:.1f}%)")
                print(f"Malignant (1): {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
            else:
                print("ℹ️ No ground truth labels - will only make predictions")
                X_test = test_df.copy()
                if 'ID' in X_test.columns:
                    X_test = X_test.drop('ID', axis=1)
            
            # Apply EXACT same feature engineering as training
            print("\n🔧 Applying ultra feature engineering (EXACT same as training)...")
            
            # Prepare data for feature engineering
            if has_diagnosis:
                df_with_diagnosis = test_df.copy()
                if df_with_diagnosis['Diagnosis'].dtype == 'object':
                    df_with_diagnosis['Diagnosis'] = self.label_encoder.transform(df_with_diagnosis['Diagnosis'])
            else:
                df_with_diagnosis = test_df.copy()
                df_with_diagnosis['Diagnosis'] = 0
            
            # Apply feature engineering
            df_engineered = ultra_feature_engineering(df_with_diagnosis)
            print(f"After feature engineering: {df_engineered.shape}")
            
            # Remove diagnosis and ID columns for processing
            cols_to_drop = ['Diagnosis']
            if 'ID' in df_engineered.columns:
                cols_to_drop.append('ID')
            X_engineered = df_engineered.drop(columns=cols_to_drop, errors='ignore')
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
            
            # Store feature names after selection
            feature_names_var = X_engineered.columns[self.variance_selector.get_support()]
            feature_names_stat = feature_names_var[self.stat_selector.get_support()]
            self.feature_names = feature_names_stat[self.model_selector.get_support()].tolist()
            print(f"Final features ({len(self.feature_names)}): {self.feature_names[:5]}...")
            
            # Apply scaling
            print("\n⚖️ Applying scaling...")
            X_scaled = self.scaler.transform(X_model)
            print(f"Final processed shape: {X_scaled.shape}")
            
            # Make predictions
            print("\n🔮 Making predictions...")
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Calculate prediction statistics
            pred_malignant = sum(predictions)
            pred_benign = len(predictions) - pred_malignant
            
            print(f"\n📊 PREDICTION RESULTS:")
            print(f"Total samples: {len(predictions)}")
            print(f"Predicted Benign: {pred_benign} ({pred_benign/len(predictions)*100:.1f}%)")
            print(f"Predicted Malignant: {pred_malignant} ({pred_malignant/len(predictions)*100:.1f}%)")
            
            # Calculate accuracy if ground truth available
            if has_diagnosis:
                accuracy = accuracy_score(y_true, predictions)
                f1 = f1_score(y_true, predictions)
                auc = roc_auc_score(y_true, probabilities[:, 1])
                
                print(f"\n🎯 ACCURACY RESULTS:")
                print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"F1 Score: {f1:.4f}")
                print(f"AUC Score: {auc:.4f}")
                
                if accuracy >= 0.95:
                    print("🎉 SUCCESS! Achieved 95%+ accuracy!")
                elif accuracy >= 0.90:
                    print("🌟 EXCELLENT! Achieved 90%+ accuracy!")
                
                print(f"\n📋 Detailed Classification Report:")
                print(classification_report(y_true, predictions, 
                                          target_names=['Benign', 'Malignant']))
                
                # Confusion matrix
                cm = confusion_matrix(y_true, predictions)
                print(f"\n🔢 Confusion Matrix:")
                print(f"              Predicted")
                print(f"Actual    Benign  Malignant")
                print(f"Benign     {cm[0,0]:4d}      {cm[0,1]:4d}")
                print(f"Malignant  {cm[1,0]:4d}      {cm[1,1]:4d}")
            
            # Generate explanations automatically if libraries are available
            if SHAP_AVAILABLE or LIME_AVAILABLE:
                print("\n" + "="*60)
                print("🔬 MODEL INTERPRETABILITY ANALYSIS")
                print("="*60)
                
                # SHAP Explanations (Summary and Importance plots only)
                if SHAP_AVAILABLE:
                    shap_values = self.generate_shap_explanations(X_scaled, predictions)
                
                # LIME Explanations (Top 15 features only)
                if LIME_AVAILABLE:
                    lime_explanations = self.generate_lime_explanations(
                        X_scaled, predictions, num_samples=min(10, len(X_scaled))
                    )
            
            # Save results
            print(f"\n💾 Saving results...")
            output_df = test_df.copy()
            output_df['Predicted_Diagnosis'] = predictions
            output_df['Prediction_Label'] = self.label_encoder.inverse_transform(predictions)
            output_df['Probability_Benign'] = probabilities[:, 0]
            output_df['Probability_Malignant'] = probabilities[:, 1]
            output_df['Confidence'] = np.max(probabilities, axis=1)
            
            if has_diagnosis:
                output_df['Correct_Prediction'] = (predictions == y_true).astype(int)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"breast_cancer_results_{timestamp}.csv"
            output_df.to_csv(output_filename, index=False)
            
            print(f"✅ Results saved to: {output_filename}")
            
            if has_diagnosis and accuracy >= 0.95:
                print(f"\n🎉 MISSION ACCOMPLISHED!")
                print(f"✅ Achieved {accuracy:.1%} accuracy!")
                
            print("\n🎉 TESTING COMPLETED!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during testing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function for testing with correct pipeline and explanations"""
    print("\n" + "="*70)
    print("🎗️  BREAST CANCER PREDICTION WITH SHAP & LIME EXPLANATIONS")
    print("="*70)
    
    # Check if SHAP and LIME are available
    if not SHAP_AVAILABLE:
        print("\n⚠️ SHAP is not installed. To install:")
        print("   pip install shap")
    
    if not LIME_AVAILABLE:
        print("\n⚠️ LIME is not installed. To install:")
        print("   pip install lime")
    
    if not SHAP_AVAILABLE and not LIME_AVAILABLE:
        response = input("\n❓ Continue without explanations? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            return
    
    # Use fixed model directory
    model_dir = "breast_cancer_model_saved"
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory '{model_dir}' not found!")
        print("Please run the training script first to create the model.")
        return
    
    # Check for required model files
    required_files = ['model.pkl', 'scaler.pkl', 'feature_selectors.pkl', 
                     'label_encoder.pkl', 'performance_metrics.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f"{model_dir}/{f}")]
    
    if missing_files:
        print(f"❌ Missing required model files: {missing_files}")
        return
    
    print(f"✅ Model directory found: {model_dir}")
    
    # Get test data file
    print(f"\n📂 Please provide your test CSV file path:")
    print("   (Can be from bc_data_splits/test_data.csv or any other test file)")
    file_path = input("📁 Enter file path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print("❌ Test file not found!")
        print(f"Tried to load: {file_path}")
        return
    
    # Create tester and run (explanations will be generated automatically if libraries are available)
    tester = CorrectBreastCancerTester(model_dir=model_dir)
    tester.test_model(file_path)


if __name__ == "__main__":
    main()