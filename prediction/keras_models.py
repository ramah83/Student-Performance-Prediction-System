"""
Keras/TensorFlow implementation for H5 model files - Separate models for each subject
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow غير متاح - سيتم استخدام النماذج التقليدية فقط")
    TENSORFLOW_AVAILABLE = False
import joblib
class KerasPredictor:
    def __init__(self):
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow غير متاح - سيتم استخدام النماذج التقليدية فقط")
            self.tensorflow_available = False
        else:
            self.tensorflow_available = True
        self.models = {}
        self.encoders = None
        self.scaler = None
        self.model_dir = 'models/keras/'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_real_data(self):
        try:
            
            df = pd.read_csv('Data/StudentsPerformance.csv')
            print(f"Loaded {len(df)} records from CSV")
            return df
        except FileNotFoundError:
            print("CSV file not found, using sample data")
            
            data = {
                'gender': ['female', 'female', 'female', 'male', 'male'] * 200,
                'race/ethnicity': ['group B', 'group C', 'group B', 'group A', 'group C'] * 200,
                'parental level of education': ['bachelor\'s degree', 'some college', 'master\'s degree', 'associate\'s degree', 'some college'] * 200,
                'lunch': ['standard', 'standard', 'standard', 'free/reduced', 'standard'] * 200,
                'test preparation course': ['none', 'completed', 'none', 'none', 'none'] * 200,
                'math score': np.random.randint(0, 100, 1000),
                'reading score': np.random.randint(0, 100, 1000),
                'writing score': np.random.randint(0, 100, 1000)
            }
            return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Preprocess the data for neural networks"""
        
        le_gender = LabelEncoder()
        le_race = LabelEncoder()
        le_education = LabelEncoder()
        le_lunch = LabelEncoder()
        le_prep = LabelEncoder()
        
        
        df_encoded = df.copy()
        df_encoded['gender'] = le_gender.fit_transform(df['gender'])
        df_encoded['race/ethnicity'] = le_race.fit_transform(df['race/ethnicity'])
        df_encoded['parental level of education'] = le_education.fit_transform(df['parental level of education'])
        df_encoded['lunch'] = le_lunch.fit_transform(df['lunch'])
        df_encoded['test preparation course'] = le_prep.fit_transform(df['test preparation course'])
        
        self.encoders = (le_gender, le_race, le_education, le_lunch, le_prep)
        
        return df_encoded
    
    def add_enhanced_features(self, df):
        """Add enhanced features to improve model performance - NON-SCORE BASED ONLY"""
        
        
        df['advantaged_background'] = ((df['lunch'] == 'standard') & 
                                      (df['parental level of education'].isin(['bachelor\'s degree', 'master\'s degree']))).astype(int)
        
        df['disadvantaged_background'] = ((df['lunch'] == 'free/reduced') & 
                                         (df['parental level of education'].isin(['some high school', 'high school']))).astype(int)
        
        
        df['prep_with_support'] = ((df['test preparation course'] == 'completed') & 
                                  (df['advantaged_background'] == 1)).astype(int)
        
        
        df['female_language_advantage'] = ((df['gender'] == 'female') & 
                                          (df['parental level of education'].isin(['bachelor\'s degree', 'master\'s degree']))).astype(int)
        
        df['male_math_advantage'] = ((df['gender'] == 'male') & 
                                    (df['parental level of education'].isin(['bachelor\'s degree', 'master\'s degree']))).astype(int)
        
        
        education_scores = {
            'some high school': 1,
            'high school': 2, 
            'some college': 3,
            'associate\'s degree': 4,
            'bachelor\'s degree': 5,
            'master\'s degree': 6
        }
        df['education_level_numeric'] = df['parental level of education'].map(education_scores).fillna(2)
        
        
        df['strong_support_system'] = ((df['lunch'] == 'standard') & 
                                      (df['test preparation course'] == 'completed') &
                                      (df['education_level_numeric'] >= 4)).astype(int)
        
        
        df['multiple_risk_factors'] = ((df['lunch'] == 'free/reduced') & 
                                      (df['test preparation course'] == 'none') &
                                      (df['education_level_numeric'] <= 2)).astype(int)
        
        
        high_performing_groups = ['group E', 'group D']  
        df['high_performing_ethnicity'] = df['race/ethnicity'].isin(high_performing_groups).astype(int)
        
        return df
    
    def create_optimized_model(self, input_dim):
        """Create an enhanced neural network model with more features"""
        model = keras.Sequential([
            
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.15),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            
            layers.Dense(1, activation='linear')  
        ])
        
        
        optimizer = keras.optimizers.Adam(
            learning_rate=0.002,  
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0  
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  
            metrics=['mae', 'mse', 'mape']  
        )
        
        return model
    
    def train_models(self):
        """Train neural network models and save as H5 files"""
        if not self.tensorflow_available:
            raise ImportError("TensorFlow غير متاح - يرجى تثبيته أولاً")
            
        
        df = self.load_real_data()
        
        
        df = self.add_enhanced_features(df)
        
        df_encoded = self.preprocess_data(df)
        
        
        
        demographic_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        
        
        engineered_features = [
            'advantaged_background', 'disadvantaged_background', 'prep_with_support',
            'female_language_advantage', 'male_math_advantage', 'education_level_numeric',
            'strong_support_system', 'multiple_risk_factors', 'high_performing_ethnicity'
        ]
        
        
        features = demographic_features + engineered_features
        
        
        available_features = [f for f in features if f in df_encoded.columns]
        print(f"Using {len(available_features)} features: {available_features}")
        
        X = df_encoded[available_features].values
        
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        
        subjects = ['math score', 'reading score', 'writing score']
        scores = {}
        
        for subject in subjects:
            print(f"Training model for {subject}...")
            
            y = df_encoded[subject].values
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            
            model = self.create_optimized_model(X_train.shape[1])
            
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.001
            )
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=0
            )
            
            
            checkpoint_path = os.path.join(self.model_dir, f"best_{subject.replace(' ', '_')}_model.h5")
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            
            callbacks = [early_stopping, reduce_lr, model_checkpoint]
            
            
            history = model.fit(
                X_train, y_train,
                epochs=200,  
                batch_size=16,  
                validation_split=0.25,  
                callbacks=callbacks,
                verbose=0,
                shuffle=True
            )
            
            
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            scores[subject] = {'loss': test_loss, 'mae': test_mae}
            
            
            model_filename = f"{subject.replace(' ', '_')}_model.h5"
            model_path = os.path.join(self.model_dir, model_filename)
            model.save(model_path)
            
            self.models[subject] = model
            print(f"Model saved: {model_path}")
        
        
        joblib.dump(self.encoders, os.path.join(self.model_dir, 'encoders.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))
        
        return scores
    
    def load_models(self):
        """Load trained H5 models"""
        try:
            subjects = ['math_score', 'reading_score', 'writing_score']
            
            for subject in subjects:
                model_path = os.path.join(self.model_dir, f"{subject}_model.h5")
                if os.path.exists(model_path):
                    self.models[subject] = keras.models.load_model(model_path)
            
            
            self.encoders = joblib.load(os.path.join(self.model_dir, 'encoders.joblib'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict(self, input_data):
        """Make predictions using loaded H5 models"""
        if not self.models or not self.encoders or not self.scaler:
            raise ValueError("Models not loaded. Please train or load models first.")
        
        
        le_gender, le_race, le_education, le_lunch, le_prep = self.encoders
        
        encoded_data = [
            le_gender.transform([input_data['gender']])[0],
            le_race.transform([input_data['race_ethnicity']])[0],
            le_education.transform([input_data['parental_education']])[0],
            le_lunch.transform([input_data['lunch']])[0],
            le_prep.transform([input_data['test_preparation']])[0]
        ]
        
        
        input_array = np.array([encoded_data])
        input_scaled = self.scaler.transform(input_array)
        
        
        predictions = {}
        subject_mapping = {
            'math_score': 'math_score',
            'reading_score': 'reading_score', 
            'writing_score': 'writing_score'
        }
        
        for subject, model_key in subject_mapping.items():
            if model_key in self.models:
                pred = self.models[model_key].predict(input_scaled, verbose=0)[0][0]
                predictions[subject] = float(pred)
        
        return predictions


if __name__ == "__main__":
    predictor = KerasPredictor()
    
    
    print("Training Keras models...")
    scores = predictor.train_models()
    print("Training completed!")
    print("Model scores:", scores)
    
    
    test_data = {
        'gender': 'female',
        'race_ethnicity': 'group B',
        'parental_education': 'bachelor\'s degree',
        'lunch': 'standard',
        'test_preparation': 'completed'
    }
    
    predictions = predictor.predict(test_data)
    print("Predictions:", predictions)