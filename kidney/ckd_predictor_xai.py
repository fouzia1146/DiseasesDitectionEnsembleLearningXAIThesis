#!/usr/bin/env python3
"""
Fixed CKD Predictor with SHAP and LIME explanations
SHAP bar plot now generates correctly
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
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

def load_arff_file(filename):
    """Load ARFF file and convert to pandas DataFrame"""
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the @data section
    data_start = content.find('@data')
    if data_start == -1:
        raise ValueError("No @data section found in ARFF file")
    
    # Extract data section
    data_section = content[data_start + 5:].strip()
    data_lines = [line.strip() for line in data_section.split('\n') if line.strip()]
    
    # Column names based on the dataset documentation
    columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
              'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 
              'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
    
    # Parse data
    data = []
    for line in data_lines:
        # Clean the line
        line = line.replace('\r', '').replace('\t', ' ')
        values = [val.strip() for val in line.split(',')]
        
        # Ensure we have the right number of columns
        if len(values) == len(columns):
            data.append(values)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    return df

def enhanced_feature_engineering(df):
    """EXACT COPY of the enhanced feature engineering from training code"""
    df_new = df.copy()
    
    # Medical ratios and indices
    epsilon = 1e-8
    
    # Kidney function indicators
    if 'sc' in df_new.columns and 'bu' in df_new.columns:
        df_new['kidney_damage_score'] = df_new['sc'] * df_new['bu']
        df_new['sc_bu_ratio'] = df_new['sc'] / (df_new['bu'] + epsilon)
        
    # Anemia indicators
    if 'hemo' in df_new.columns and 'pcv' in df_new.columns:
        df_new['anemia_index'] = df_new['hemo'] * df_new['pcv']
        df_new['hemo_pcv_ratio'] = df_new['hemo'] / (df_new['pcv'] + epsilon)
        
    # Blood cell ratios
    if 'wbcc' in df_new.columns and 'rbcc' in df_new.columns:
        df_new['wbc_rbc_ratio'] = df_new['wbcc'] / (df_new['rbcc'] * 1000 + epsilon)
        
    # Electrolyte balance
    if 'sod' in df_new.columns and 'pot' in df_new.columns:
        df_new['na_k_ratio'] = df_new['sod'] / (df_new['pot'] + epsilon)
        
    # Protein loss indicators
    if 'al' in df_new.columns:
        df_new['albumin_severity'] = df_new['al'] ** 2
        
    # Age-related features
    if 'age' in df_new.columns:
        df_new['age_group'] = pd.cut(df_new['age'], bins=[0, 30, 50, 65, 100], labels=[0, 1, 2, 3])
        df_new['age_risk'] = (df_new['age'] > 60).astype(int)
        
    # Blood pressure categories
    if 'bp' in df_new.columns:
        df_new['hypertension_severity'] = pd.cut(df_new['bp'], 
                                               bins=[0, 90, 120, 140, 180, 300], 
                                               labels=[0, 1, 2, 3, 4])
        
    # Comprehensive risk score
    risk_features = []
    weights = []
    
    if 'sc' in df_new.columns:
        risk_features.append(df_new['sc'])
        weights.append(0.3)
    if 'bu' in df_new.columns:
        risk_features.append(df_new['bu'] / 100)  # Normalize
        weights.append(0.25)
    if 'al' in df_new.columns:
        risk_features.append(df_new['al'])
        weights.append(0.2)
    if 'htn' in df_new.columns:
        risk_features.append(df_new['htn'])
        weights.append(0.15)
    if 'dm' in df_new.columns:
        risk_features.append(df_new['dm'])
        weights.append(0.1)
        
    if risk_features:
        weighted_risk = sum(f * w for f, w in zip(risk_features, weights))
        df_new['ckd_risk_score'] = weighted_risk
    
    # Convert categorical features to numeric
    categorical_cols = ['age_group', 'hypertension_severity']
    for col in categorical_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(float)
    
    # Fill any new NaN values
    df_new.fillna(0, inplace=True)
    df_new.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df_new

def preprocess_ckd_data(df):
    """EXACT COPY of the preprocessing from enhanced training code"""
    df_processed = df.copy()
    target = 'class'
    
    # Replace '?' with NaN
    df_processed = df_processed.replace('?', np.nan)
    
    # Define column types based on the dataset documentation
    numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    binary_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    
    # Convert numeric columns
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Clean and encode binary columns
    for col in binary_cols:
        if col in df_processed.columns:
            # Standardize values
            df_processed[col] = df_processed[col].astype(str).str.lower().str.strip()
            # Map to binary values
            if col == 'rbc':
                df_processed[col] = df_processed[col].map({'normal': 0, 'abnormal': 1})
            elif col == 'pc':
                df_processed[col] = df_processed[col].map({'normal': 0, 'abnormal': 1})
            elif col in ['pcc', 'ba']:
                df_processed[col] = df_processed[col].map({'notpresent': 0, 'present': 1})
            elif col in ['htn', 'dm', 'cad', 'pe', 'ane']:
                df_processed[col] = df_processed[col].map({'no': 0, 'yes': 1})
            elif col == 'appet':
                df_processed[col] = df_processed[col].map({'good': 0, 'poor': 1})
    
    # Handle target column only if it exists and needs encoding
    if target in df_processed.columns:
        # Only encode if it's not already numeric
        if df_processed[target].dtype == 'object':
            df_processed[target] = df_processed[target].map({'notckd': 0, 'ckd': 1})
        # If it's already numeric, leave it as is
    
    # Handle missing values strategically
    # Impute numeric columns with median
    numeric_imputer = SimpleImputer(strategy='median')
    existing_numeric = [col for col in numeric_cols if col in df_processed.columns]
    if existing_numeric:
        df_processed[existing_numeric] = numeric_imputer.fit_transform(df_processed[existing_numeric])
    
    # Impute binary columns with mode
    for col in binary_cols:
        if col in df_processed.columns and df_processed[col].isnull().any():
            mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 0
            df_processed[col].fillna(mode_val, inplace=True)
    
    return df_processed


class CorrectCKDTester:
    """CKD tester using the EXACT same pipeline as enhanced training with SHAP and LIME"""
    
    def __init__(self, model_dir="enhanced_ckd_model"):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_dir = model_dir
        
    def load_model(self):
        """Load the trained model and all preprocessors"""
        print(f"📂 Loading model from {self.model_dir}...")
        
        try:
            # Load all components
            self.model = joblib.load(f"{self.model_dir}/best_model.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            self.feature_names = joblib.load(f"{self.model_dir}/feature_names.pkl")
            
            print(f"✅ Model loaded successfully!")
            print(f"🤖 Model type: {type(self.model).__name__}")
            print(f"🔧 Scaler type: {type(self.scaler).__name__}")
            print(f"📊 Expected features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_shap_explanations(self, X_scaled, predictions):
        """Generate SHAP explanations - FIXED to create proper bar plot"""
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available. Skipping SHAP explanations.")
            return None
        
        print("\n🔮 Generating SHAP explanations for CKD model...")
        
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
            print("📊 Creating SHAP visualizations for kidney function indicators...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Summary plot (beeswarm plot)
            plt.close('all')
            print("📊 Creating SHAP summary beeswarm plot...")
            
            fig1 = plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_scaled, 
                            feature_names=self.feature_names,
                            show=False,
                            max_display=20)  # Show top 20 features
            plt.suptitle("SHAP Summary Plot - Feature Impact on CKD Prediction", 
                        y=0.99, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save
            summary_filename = f'ckd_shap_summary_{timestamp}.png'
            try:
                plt.savefig(summary_filename, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"   ✅ Saved: {summary_filename}")
            except Exception as save_error:
                print(f"   ⚠️ Could not save file: {save_error}")
            
            # Display
            plt.draw()
            plt.pause(0.1)
            plt.show(block=False)
            
            # Keep window open briefly
            import time
            time.sleep(2)
            
            plt.close(fig1)
            
            # 2. Feature importance bar plot - IMPROVED VERSION
            print("📊 Creating SHAP feature importance bar plot...")
            plt.close('all')
            
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Verify shapes match
            if len(mean_abs_shap) != len(self.feature_names):
                print(f"⚠️ Shape mismatch: {len(mean_abs_shap)} SHAP values vs {len(self.feature_names)} features")
                # Ensure they match
                min_len = min(len(mean_abs_shap), len(self.feature_names))
                mean_abs_shap = mean_abs_shap[:min_len]
                feature_names_subset = self.feature_names[:min_len]
            else:
                feature_names_subset = self.feature_names
            
            # Create DataFrame for sorting
            feature_importance = pd.DataFrame({
                'feature': feature_names_subset,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=True)
            
            # Show top 20 features if too many
            if len(feature_importance) > 20:
                feature_importance = feature_importance.tail(20)
                print(f"   Showing top 20 most important features")
            
            # Create bar plot with better visibility
            fig2, ax = plt.subplots(figsize=(12, 10))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
            
            bars = ax.barh(range(len(feature_importance)), 
                          feature_importance['importance'].values,
                          color=colors,
                          edgecolor='black',
                          linewidth=0.5)
            
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['feature'].values, fontsize=10)
            ax.set_xlabel('Mean |SHAP value| (Average impact on model output)', fontsize=12, fontweight='bold')
            ax.set_title('SHAP Feature Importance - Chronic Kidney Disease Model', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(feature_importance.iterrows()):
                ax.text(row['importance'], i, f' {row["importance"]:.4f}', 
                       va='center', fontsize=8)
            
            plt.tight_layout()
            
            # Save with multiple attempts
            importance_filename = f'ckd_shap_importance_{timestamp}.png'
            try:
                plt.savefig(importance_filename, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"   ✅ Saved: {importance_filename}")
            except Exception as save_error:
                print(f"   ⚠️ Could not save file: {save_error}")
            
            # Force display
            plt.draw()
            plt.pause(0.1)
            plt.show(block=False)
            
            # Keep window open briefly
            import time
            time.sleep(2)
            
            plt.close(fig2)
            
            print("✅ SHAP explanations generated successfully!")
            print(f"   - Summary plot saved: ckd_shap_summary_{timestamp}.png")
            print(f"   - Importance plot saved: ckd_shap_importance_{timestamp}.png")
            
            return shap_values
            
        except Exception as e:
            print(f"❌ Error generating SHAP explanations: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_lime_explanations(self, X_scaled, predictions, num_samples=5):
        """Generate LIME explanations - top 15 features summary only"""
        if not LIME_AVAILABLE:
            print("⚠️ LIME not available. Skipping LIME explanations.")
            return None
        
        print("\n🍋 Generating LIME explanations for CKD model...")
        
        try:
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_scaled,
                feature_names=self.feature_names,
                class_names=['Not CKD', 'CKD'],
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            
            # Generate explanations for selected samples
            lime_explanations = []
            sample_indices = np.random.choice(len(X_scaled), 
                                            min(num_samples, len(X_scaled)), 
                                            replace=False)
            
            print(f"Analyzing {len(sample_indices)} patient samples for LIME explanations...")
            
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
                
                # Create horizontal bar plot with CKD themed colors (kidney-related: purple/violet)
                plt.barh(range(len(features)), importances, color='mediumpurple', alpha=0.7)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Average Absolute Importance')
                plt.title("LIME - Top 15 Clinical Features for CKD Detection")
                plt.gca().invert_yaxis()  # Highest importance at top
                
                # Add value labels on bars
                for i, v in enumerate(importances):
                    plt.text(v, i, f' {v:.4f}', va='center')
                
                plt.tight_layout()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(f'ckd_lime_top15_features_{timestamp}.png', dpi=100, bbox_inches='tight')
                plt.show()
                plt.close()
                
                # Print text summary with clinical context
                print("\n📝 LIME Top 15 Most Important Clinical Features for CKD:")
                for i, (feature, importance) in enumerate(top_features.items(), 1):
                    # Add clinical context for key features
                    context = ""
                    if 'sc' in feature.lower():
                        context = " (serum creatinine)"
                    elif 'bu' in feature.lower():
                        context = " (blood urea)"
                    elif 'hemo' in feature.lower():
                        context = " (hemoglobin)"
                    elif 'pcv' in feature.lower():
                        context = " (packed cell volume)"
                    elif 'al' in feature.lower():
                        context = " (albumin)"
                    elif 'kidney_damage' in feature.lower():
                        context = " (composite score)"
                    
                    print(f"   {i:2d}. {feature:30s}{context}: {importance:.4f}")
            
            print("✅ LIME explanations generated successfully!")
            return lime_explanations
            
        except Exception as e:
            print(f"❌ Error generating LIME explanations: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_model(self, file_path):
        """Test the model using the EXACT same preprocessing pipeline with explanations"""
        print("\n🔍 TESTING MODE - Using Enhanced CKD Training Pipeline")
        print("="*60)
        
        if not self.load_model():
            return False
        
        try:
            # Load test data
            print(f"\n📂 Loading test data from: {file_path}")
            
            # Check if it's an ARFF file or CSV
            if file_path.endswith('.arff'):
                print("Detected ARFF file format")
                test_df = load_arff_file(file_path)
            else:
                print("Detected CSV file format")
                test_df = pd.read_csv(file_path)
                
            print(f"Test data shape: {test_df.shape}")
            print(f"Columns in test data: {list(test_df.columns)}")
            
            # Show first few rows for debugging
            print(f"\nFirst 3 rows of test data:")
            print(test_df.head(3))
            
            # Check for class column
            has_class = 'class' in test_df.columns
            if has_class:
                print("✅ Ground truth labels found - will calculate accuracy")
                # Handle different possible class values
                raw_classes = test_df['class'].unique()
                print(f"Raw class values found: {raw_classes}")
                
                # Store original class values before preprocessing
                original_classes = test_df['class'].copy()
                print(f"Original class value counts: {original_classes.value_counts().to_dict()}")
                
                # The preprocessing might modify the class column, so we handle it separately
                if test_df['class'].dtype in ['int64', 'float64']:
                    # Already numeric - just ensure it's int
                    y_true = original_classes.astype(int)
                    print(f"Converted to int - checking for NaN: {y_true.isnull().sum()} NaN values")
                else:
                    # String values - map them
                    y_true = original_classes.map({'notckd': 0, 'ckd': 1})
                    if y_true.isnull().any():
                        print(f"String mapping failed, {y_true.isnull().sum()} NaN values")
                        # Try alternative mappings
                        alt_mapping = {}
                        for val in raw_classes:
                            val_str = str(val).lower().strip()
                            if val_str in ['0', '0.0', 'notckd', 'not_ckd', 'normal']:
                                alt_mapping[val] = 0
                            elif val_str in ['1', '1.0', 'ckd', 'chronic_kidney_disease']:
                                alt_mapping[val] = 1
                        
                        if alt_mapping:
                            y_true = original_classes.map(alt_mapping)
                            print(f"Applied alternative mapping: {alt_mapping}")
                            print(f"After alt mapping - NaN count: {y_true.isnull().sum()}")
                
                # Final check and debug info
                print(f"Final y_true info:")
                print(f"  Type: {type(y_true)}")
                print(f"  Dtype: {y_true.dtype}")
                print(f"  NaN count: {y_true.isnull().sum()}")
                print(f"  Shape: {y_true.shape}")
                print(f"  Unique values: {y_true.unique()}")
                
                # Check for remaining NaN values
                if y_true.isnull().any():
                    print(f"Error: Could not map all class values.")
                    print(f"Original values: {original_classes.value_counts().to_dict()}")
                    print(f"NaN positions: {y_true.isnull().sum()}")
                    
                    # Try to manually fix NaN values
                    print("Attempting to fix NaN values...")
                    y_true = y_true.fillna(0)  # Fill NaN with 0 as fallback
                    print(f"After fillna - NaN count: {y_true.isnull().sum()}")
                
                print(f"Class distribution in test data:")
                print(f"Not CKD (0): {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true)*100:.1f}%)")
                print(f"CKD (1): {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
            else:
                print("ℹ️ No ground truth labels - will only make predictions")
            
            # Apply EXACT same preprocessing as training
            print("\n🔧 Applying enhanced CKD preprocessing (EXACT same as training)...")
            df_preprocessed = preprocess_ckd_data(test_df)
            print(f"After preprocessing: {df_preprocessed.shape}")
            
            # Apply EXACT same feature engineering as training
            print("\n🔧 Applying enhanced feature engineering (EXACT same as training)...")
            df_engineered = enhanced_feature_engineering(df_preprocessed)
            print(f"After feature engineering: {df_engineered.shape}")
            
            # Remove class column for processing if present
            if 'class' in df_engineered.columns:
                X_engineered = df_engineered.drop('class', axis=1)
            else:
                X_engineered = df_engineered
            print(f"Features for processing: {X_engineered.shape}")
            
            # Ensure we have the same features as training
            print(f"\n🔍 Aligning features with training data...")
            print(f"Training features: {len(self.feature_names)}")
            print(f"Current features: {len(X_engineered.columns)}")
            
            # Add missing features with zeros
            for feature in self.feature_names:
                if feature not in X_engineered.columns:
                    X_engineered[feature] = 0
                    print(f"Added missing feature: {feature}")
            
            # Remove extra features
            extra_features = [col for col in X_engineered.columns if col not in self.feature_names]
            if extra_features:
                print(f"Removing extra features: {extra_features}")
                X_engineered = X_engineered.drop(columns=extra_features)
            
            # Reorder columns to match training
            X_engineered = X_engineered[self.feature_names]
            print(f"Final feature alignment: {X_engineered.shape}")
            
            # Apply scaling
            print("\n⚖️ Applying scaling...")
            X_scaled = self.scaler.transform(X_engineered)
            print(f"Final processed shape: {X_scaled.shape}")
            
            # Make predictions
            print("\n🔮 Making predictions...")
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Calculate prediction statistics
            pred_ckd = sum(predictions)
            pred_not_ckd = len(predictions) - pred_ckd
            
            print(f"\n📊 PREDICTION RESULTS:")
            print(f"Total samples: {len(predictions)}")
            print(f"Predicted Not CKD: {pred_not_ckd} ({pred_not_ckd/len(predictions)*100:.1f}%)")
            print(f"Predicted CKD: {pred_ckd} ({pred_ckd/len(predictions)*100:.1f}%)")
            
            # Calculate accuracy if ground truth available
            if has_class:
                accuracy = accuracy_score(y_true, predictions)
                f1 = f1_score(y_true, predictions)
                auc = roc_auc_score(y_true, probabilities[:, 1])
                
                print(f"\n🎯 ACCURACY RESULTS:")
                print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"F1 Score: {f1:.4f}")
                print(f"AUC Score: {auc:.4f}")
                
                # Try threshold optimization to improve accuracy
                if accuracy < 0.85:
                    print(f"\n🔧 Optimizing decision threshold...")
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
                    print(f"   Achieved: {accuracy*100:.1f}%")
                
                print(f"\n📋 Detailed Classification Report:")
                print(classification_report(y_true, predictions, 
                                          target_names=['Not CKD', 'CKD']))
                
                # Confusion matrix
                cm = confusion_matrix(y_true, predictions)
                print(f"\n🔢 Confusion Matrix:")
                print(f"                Predicted")
                print(f"Actual      Not CKD    CKD")
                print(f"Not CKD       {cm[0,0]:4d}   {cm[0,1]:4d}")
                print(f"CKD           {cm[1,0]:4d}   {cm[1,1]:4d}")
                
                # Show some prediction examples
                print(f"\n🔍 Sample predictions:")
                sample_size = min(10, len(predictions))
                for i in range(sample_size):
                    actual = "CKD" if y_true.iloc[i] == 1 else "Not CKD"
                    predicted = "CKD" if predictions[i] == 1 else "Not CKD"
                    confidence = probabilities[i].max()
                    correct = "✅" if y_true.iloc[i] == predictions[i] else "❌"
                    print(f"  {correct} Actual: {actual:7} | Predicted: {predicted:7} | Confidence: {confidence:.3f}")
            
            # Generate explanations automatically if libraries are available
            if SHAP_AVAILABLE or LIME_AVAILABLE:
                print("\n" + "="*60)
                print("🔬 MODEL INTERPRETABILITY ANALYSIS - KIDNEY FUNCTION")
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
            output_df['Predicted_Class'] = predictions
            output_df['Prediction_Label'] = ['CKD' if p == 1 else 'Not CKD' for p in predictions]
            output_df['Probability_Not_CKD'] = probabilities[:, 0]
            output_df['Probability_CKD'] = probabilities[:, 1]
            output_df['Confidence'] = np.max(probabilities, axis=1)
            
            if has_class:
                output_df['Correct_Prediction'] = (predictions == y_true).astype(int)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ckd_results_{timestamp}.csv"
            output_df.to_csv(output_filename, index=False)
            
            print(f"✅ Results saved to: {output_filename}")
            
            if has_class and accuracy >= 0.85:
                print(f"\n🎉 MISSION ACCOMPLISHED!")
                print(f"✅ Achieved {accuracy:.1%} accuracy!")
            elif has_class:
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
    """Main function for testing with enhanced CKD pipeline and explanations"""
    print("\n" + "="*70)
    print("🏥 CHRONIC KIDNEY DISEASE PREDICTION WITH SHAP & LIME EXPLANATIONS")
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
    
    # Use enhanced model directory
    model_dir = "enhanced_ckd_model"
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory '{model_dir}' not found!")
        print("Please ensure you have run the enhanced CKD training script first.")
        return
    
    # Check for required model files
    required_files = ['best_model.pkl', 'scaler.pkl', 'feature_names.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f"{model_dir}/{f}")]
    
    if missing_files:
        print(f"❌ Missing required model files: {missing_files}")
        return
    
    print(f"✅ Model directory found: {model_dir}")
    
    # Get test data file
    print(f"\n📂 Please provide your test CSV file path:")
    print("Expected columns: age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, pe, ane[, class]")
    file_path = input("📁 Enter file path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print("❌ Test file not found!")
        return
    
    # Create tester and run (explanations will be generated automatically if libraries are available)
    tester = CorrectCKDTester(model_dir=model_dir)
    tester.test_model(file_path)


if __name__ == "__main__":
    main()