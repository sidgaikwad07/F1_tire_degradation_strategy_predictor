"""
Created on Fri Jun 13 09:15:17 2025

@author: sid

F1 Tire Degradation Training Script
Trains the model and saves it for later predictions
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class F1TireDegradationTrainer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_importance = {}
        
    def load_data(self):
        """Load the F1 data"""
        print("📊 Loading F1 tire degradation data...")
        
        data_path = "/Users/sid/Downloads/F1_tire_degradation_strategy_predictor/processed_ml_data"
        train_data = pd.read_csv(f"{data_path}/f1_tire_degradation_train_data.csv")
        test_data = pd.read_csv(f"{data_path}/f1_tire_degradation_test_data.csv")
        
        print(f"✅ Train data: {len(train_data)} laps")
        print(f"✅ Test data: {len(test_data)} laps")
        
        return train_data, test_data
    
    def advanced_feature_engineering(self, data):
        """Create advanced features for better performance"""
        print("🔧 Advanced feature engineering...")
        
        data = data.copy()
        original_columns = len(data.columns)
        
        # 1. Tire degradation physics
        if 'TyreLife' in data.columns:
            data['TyreLife_nonlinear'] = np.power(data['TyreLife'], 1.5)
            data['TyreLife_exponential'] = np.exp(data['TyreLife'] / 15)
            data['TyreLife_log'] = np.log1p(data['TyreLife'])
        
        # 2. Temperature stress modeling
        if 'TrackTemp' in data.columns and 'AirTemp' in data.columns:
            data['TempStress'] = np.maximum(0, data['TrackTemp'] - 42)
            data['TempStress_squared'] = data['TempStress'] ** 2
            data['OptimalTemp_deviation'] = np.abs(data['TrackTemp'] - 48)
        
        # 3. Performance degradation indicators
        if 'LapTime_rolling_3' in data.columns and 'LapTime' in data.columns:
            data['PerformanceDrop'] = (data['LapTime'] - data['LapTime_rolling_3']) / data['LapTime_rolling_3']
            data['PerformanceDrop_cumulative'] = data.groupby(['Driver', 'StintNumber'])['PerformanceDrop'].cumsum()
        
        # 4. Speed degradation patterns
        speed_cols = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
        available_speed = [col for col in speed_cols if col in data.columns]
        if len(available_speed) >= 2:
            data['AvgSpeed'] = data[available_speed].mean(axis=1)
            data['SpeedVariability'] = data[available_speed].std(axis=1)
            if 'DriverAvgPace' in data.columns:
                data['SpeedDegradation'] = (data['DriverAvgPace'] - data['AvgSpeed']) / data['DriverAvgPace']
        
        # 5. Stint progression features
        if 'StintProgression' in data.columns:
            data['StintPhase_early'] = (data['StintProgression'] < 0.3).astype(int)
            data['StintPhase_middle'] = ((data['StintProgression'] >= 0.3) & (data['StintProgression'] < 0.7)).astype(int)
            data['StintPhase_late'] = (data['StintProgression'] >= 0.7).astype(int)
        
        # 6. Compound-specific degradation
        if 'Compound' in data.columns and 'TyreLife' in data.columns:
            compound_degradation_rates = {'Soft': 1.5, 'Medium': 1.0, 'Hard': 0.7, 'Intermediate': 1.2, 'Wet': 0.8}
            data['CompoundDegradationFactor'] = data['Compound'].map(compound_degradation_rates).fillna(1.0)
            data['AdjustedTyreLife'] = data['TyreLife'] * data['CompoundDegradationFactor']
        
        # 7. Circuit-specific features
        if 'circuit' in data.columns:
            high_deg_circuits = ['Spain', 'Hungary', 'Singapore', 'Abu Dhabi']
            data['HighDegradationCircuit'] = data['circuit'].isin(high_deg_circuits).astype(int)
            street_circuits = ['Monaco', 'Singapore', 'Baku']
            data['StreetCircuit'] = data['circuit'].isin(street_circuits).astype(int)
        
        # 8. Advanced interaction features
        if 'TyreLife' in data.columns and 'TrackTemp' in data.columns:
            data['TyreLife_TempStress'] = data['TyreLife'] * data.get('TempStress', 0)
        
        new_features = len(data.columns) - original_columns
        print(f"   Created {new_features} new features")
        return data
    
    def select_best_features(self, train_data):
        """Use ML to select the most predictive features"""
        print("🎯 Selecting best features using Random Forest...")
        
        # Get all numerical features
        numerical_features = []
        for col in train_data.columns:
            if (train_data[col].dtype in ['int64', 'float64'] and 
                col not in ['DegradationRate', 'DegradationCategory', 'Year', 'RaceID', 'LapNumber']):
                numerical_features.append(col)
        
        # Clean data for feature selection
        feature_data = train_data[numerical_features + ['DegradationRate']].dropna()
        
        if len(feature_data) < 1000:
            print("   Using manual feature selection...")
            return self.get_manual_features(train_data)
        
        X = feature_data[numerical_features]
        y = feature_data['DegradationRate']
        
        # Fit Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=15)
        rf.fit(X, y)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': numerical_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = feature_importance.head(35)['feature'].tolist()
        
        print(f"   Selected {len(top_features)} most important features")
        print(f"   Top 10 features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"     {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        self.feature_importance = feature_importance
        return top_features
    
    def get_manual_features(self, data):
        """Manually selected high-impact features"""
        manual_features = [
            # Core tire features
            'TyreLife', 'TyreLife_sqrt', 'TyreLife_log', 'TyreLife_squared', 'SeasonAdjusted_TyreLife',
            'TyreLife_nonlinear', 'TyreLife_exponential', 'AdjustedTyreLife',
            
            # Performance features
            'LapTime', 'LapTime_rolling_3', 'LapTime_rolling_5', 'LapTime_trend', 'LapTimeVsBest',
            'PerformanceDrop', 'PerformanceDrop_cumulative', 'DriverAvgPace', 'PaceVsDriverAvg',
            
            # Temperature features
            'TrackTemp', 'AirTemp', 'TempDelta', 'TempStress', 'TempStress_squared', 'OptimalTemp_deviation',
            
            # Speed features
            'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'AvgSpeed', 'SpeedVariability', 'SpeedDegradation',
            
            # Context features
            'Position', 'StintNumber', 'StintProgression', 'StintLength', 'Humidity',
            'CompoundHardness', 'CompoundDegradationFactor'
        ]
        
        return [f for f in manual_features if f in data.columns]
    
    def prepare_data(self, train_data, test_data):
        """Prepare data for training"""
        print("📊 Preparing data for training...")
        
        # Advanced feature engineering
        train_data = self.advanced_feature_engineering(train_data)
        test_data = self.advanced_feature_engineering(test_data)
        
        # Select best features
        best_features = self.select_best_features(train_data)
        
        # Categorical features
        categorical_features = ['Driver', 'Team', 'circuit', 'Compound', 'session_type']
        
        # Handle missing values
        for feature in best_features:
            if feature in train_data.columns and feature in test_data.columns:
                median_val = train_data[feature].median()
                train_data[feature].fillna(median_val, inplace=True)
                test_data[feature].fillna(median_val, inplace=True)
        
        # Remove samples with missing targets
        train_clean = train_data.dropna(subset=['DegradationRate', 'DegradationCategory']).copy()
        test_clean = test_data.dropna(subset=['DegradationRate', 'DegradationCategory']).copy()
        
        print(f"   Clean train data: {len(train_clean)} samples")
        print(f"   Clean test data: {len(test_clean)} samples")
        print(f"   Using {len(best_features)} features")
        
        # Prepare numerical features
        available_features = [f for f in best_features if f in train_clean.columns and f in test_clean.columns]
        X_train_num = train_clean[available_features].values
        X_test_num = test_clean[available_features].values
        
        # Scale features
        X_train_num = self.scaler.fit_transform(X_train_num)
        X_test_num = self.scaler.transform(X_test_num)
        
        # Enhanced categorical processing
        categorical_data = {}
        for feature in categorical_features:
            if feature in train_clean.columns:
                le = LabelEncoder()
                combined_data = pd.concat([train_clean[feature], test_clean[feature]]).astype(str)
                le.fit(combined_data)
                categorical_data[feature] = {
                    'encoder': le,
                    'vocab_size': len(le.classes_),
                    'train': le.transform(train_clean[feature].astype(str)),
                    'test': le.transform(test_clean[feature].astype(str))
                }
        
        # Prepare targets
        y_train_reg = train_clean['DegradationRate'].values
        y_test_reg = test_clean['DegradationRate'].values
        
        # Apply log transformation
        y_train_reg_log = np.log1p(np.maximum(0, y_train_reg))
        y_test_reg_log = np.log1p(np.maximum(0, y_test_reg))
        
        # Classification targets
        self.label_encoder.fit(train_clean['DegradationCategory'])
        y_train_clf = self.label_encoder.transform(train_clean['DegradationCategory'])
        y_test_clf = self.label_encoder.transform(test_clean['DegradationCategory'])
        
        print(f"   Classes: {self.label_encoder.classes_}")
        
        return {
            'X_train_num': X_train_num,
            'X_test_num': X_test_num,
            'categorical_data': categorical_data,
            'y_train_reg': y_train_reg,
            'y_test_reg': y_test_reg,
            'y_train_reg_log': y_train_reg_log,
            'y_test_reg_log': y_test_reg_log,
            'y_train_clf': y_train_clf,
            'y_test_clf': y_test_clf,
            'n_features': len(available_features),
            'feature_names': available_features,
            'train_clean': train_clean,
            'test_clean': test_clean
        }
    
    def build_model(self, n_features, categorical_data, n_classes):
        """Build the model"""
        print("🏗️ Building model...")
        
        # Numerical input
        num_input = Input(shape=(n_features,), name='numerical')
        
        # Enhanced numerical processing
        x1 = layers.Dense(256, activation='relu')(num_input)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.3)(x1)
        
        x2 = layers.Dense(128, activation='relu')(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.3)(x2)
        
        x3 = layers.Dense(64, activation='relu')(x2)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Dropout(0.2)(x3)
        
        # Residual connection
        x1_proj = layers.Dense(64, activation='relu')(x1)
        x_combined = layers.Add()([x3, x1_proj])
        
        # Categorical inputs
        cat_inputs = []
        cat_embeddings = []
        
        for feature, data in categorical_data.items():
            cat_input = Input(shape=(1,), name=f'cat_{feature}')
            embedding_dim = min(32, max(4, data['vocab_size'] // 2))
            embedding = layers.Embedding(data['vocab_size'], embedding_dim)(cat_input)
            embedding = layers.Flatten()(embedding)
            embedding = layers.Dropout(0.2)(embedding)
            cat_inputs.append(cat_input)
            cat_embeddings.append(embedding)
        
        # Combine features
        if cat_embeddings:
            cat_combined = layers.Concatenate()(cat_embeddings)
            cat_processed = layers.Dense(32, activation='relu')(cat_combined)
            cat_processed = layers.BatchNormalization()(cat_processed)
            
            # Attention-like weighting
            attention_weights = layers.Dense(32, activation='softmax')(cat_processed)
            cat_attended = layers.Multiply()([cat_processed, attention_weights])
            
            combined = layers.Concatenate()([x_combined, cat_attended])
        else:
            combined = x_combined
        
        # Shared layers
        shared = layers.Dense(128, activation='relu')(combined)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.4)(shared)
        
        shared = layers.Dense(64, activation='relu')(shared)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)
        
        # Task-specific heads
        # Regression head
        reg_head = layers.Dense(32, activation='relu')(shared)
        reg_head = layers.BatchNormalization()(reg_head)
        reg_head = layers.Dropout(0.2)(reg_head)
        reg_head = layers.Dense(16, activation='relu')(reg_head)
        regression_output = layers.Dense(1, activation='linear', name='regression')(reg_head)
        
        # Classification head
        clf_head = layers.Dense(32, activation='relu')(shared)
        clf_head = layers.BatchNormalization()(clf_head)
        clf_head = layers.Dropout(0.3)(clf_head)
        clf_head = layers.Dense(16, activation='relu')(clf_head)
        classification_output = layers.Dense(n_classes, activation='softmax', name='classification')(clf_head)
        
        # Create model
        inputs = [num_input] + cat_inputs
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.0008, clipnorm=1.0),
            loss=['huber', 'sparse_categorical_crossentropy'],
            loss_weights=[2.0, 1.0],
            metrics=[['mae'], ['accuracy']]
        )
        
        print(f"✅ Model built successfully")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, model, data):
        """Train the model"""
        print("🚀 Training model...")
        
        # Prepare inputs
        train_inputs = [data['X_train_num']]
        for feature, cat_data in data['categorical_data'].items():
            train_inputs.append(cat_data['train'].reshape(-1, 1))
        
        # Use log-transformed targets
        train_outputs = [data['y_train_reg_log'].reshape(-1, 1), data['y_train_clf']]
        
        # Sample weighting
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(data['y_train_clf']),
            y=data['y_train_clf']
        )
        
        sample_weights = np.ones(len(data['y_train_clf']))
        for i, class_idx in enumerate(data['y_train_clf']):
            base_weight = class_weights[class_idx]
            focal_weight = 1 + (data['y_train_reg'][i] ** 0.5)
            sample_weights[i] = base_weight * focal_weight
        
        print(f"   Sample weights range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=20, 
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.3, 
                patience=8, 
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_enhanced_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        history = model.fit(
            train_inputs,
            train_outputs,
            epochs=150,
            batch_size=128,
            validation_split=0.15,
            callbacks=callbacks,
            sample_weight=sample_weights,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, data):
        """Evaluate the model"""
        print("📊 Evaluating model...")
        
        # Prepare test inputs
        test_inputs = [data['X_test_num']]
        for feature, cat_data in data['categorical_data'].items():
            test_inputs.append(cat_data['test'].reshape(-1, 1))
        
        # Make predictions
        predictions = model.predict(test_inputs)
        reg_pred_log = predictions[0].flatten()
        clf_pred = np.argmax(predictions[1], axis=1)
        
        # Transform back from log space
        reg_pred = np.expm1(reg_pred_log)
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(data['y_test_reg'], reg_pred))
        mae = np.mean(np.abs(data['y_test_reg'] - reg_pred))
        r2 = r2_score(data['y_test_reg'], reg_pred)
        
        print(f"🎯 Regression Results:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        # Classification metrics
        accuracy = accuracy_score(data['y_test_clf'], clf_pred)
        
        print(f"🎯 Classification Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(data['y_test_clf'], clf_pred, target_names=self.label_encoder.classes_))
        
        return {
            'regression': {'rmse': rmse, 'mae': mae, 'r2': r2},
            'classification': {'accuracy': accuracy}
        }
    
    def save_model_and_preprocessors(self, model, data):
        """Save the trained model and all preprocessors"""
        print("💾 Saving model and preprocessors...")
        
        # Save the trained model
        model.save('best_enhanced_model.keras')
        
        # Save scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save label encoder
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save categorical encoders
        with open('categorical_encoders.pkl', 'wb') as f:
            pickle.dump(data['categorical_data'], f)
        
        # Save feature names and other metadata
        metadata = {
            'feature_names': data['feature_names'],
            'n_features': data['n_features'],
            'classes': self.label_encoder.classes_.tolist(),
            'categorical_features': list(data['categorical_data'].keys())
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Model and preprocessors saved successfully!")
        print("   Files saved:")
        print("   - best_enhanced_model.keras")
        print("   - scaler.pkl")
        print("   - label_encoder.pkl")
        print("   - categorical_encoders.pkl")
        print("   - model_metadata.json")

def main():
    """Main training function"""
    print("🧠 F1 Tire Degradation Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = F1TireDegradationTrainer()
    
    # Load data
    train_data, test_data = trainer.load_data()
    
    # Prepare data
    data = trainer.prepare_data(train_data, test_data)
    
    # Build model
    n_classes = len(trainer.label_encoder.classes_)
    model = trainer.build_model(data['n_features'], data['categorical_data'], n_classes)
    
    # Train model
    history = trainer.train_model(model, data)
    
    # Evaluate model
    results = trainer.evaluate_model(model, data)
    
    # Save everything
    trainer.save_model_and_preprocessors(model, data)
    
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Final Results:")
    print(f"Regression R²: {results['regression']['r2']:.4f}")
    print(f"Classification Accuracy: {results['classification']['accuracy']:.4f}")
    print("=" * 50)
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()