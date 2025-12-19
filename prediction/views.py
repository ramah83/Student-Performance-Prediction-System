from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns


try:
    import matplotlib_config  
except ImportError:
    pass
import base64
from io import BytesIO
from .models import StudentData, PredictionResult
from .serializers import StudentDataSerializer
from django.http import HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display


def setup_arabic_matplotlib():
    """Setup matplotlib for Arabic text rendering with Windows-compatible fonts"""
    try:
        import matplotlib.font_manager as fm
        import warnings
        
        
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
        warnings.filterwarnings('ignore', message='findfont: Font family.*not found')
        
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        
        arabic_fonts = ['Tahoma', 'Arial', 'Segoe UI', 'Calibri', 'Times New Roman', 'DejaVu Sans']
        
        
        selected_font = 'DejaVu Sans'  
        for font in arabic_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        
        plt.rcParams['font.family'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12
        
        print(f"📝 Using font for Arabic text: {selected_font}")
        return selected_font
        
    except Exception as e:
        print(f"⚠️ Font setup warning: {e}")
        
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12
        return 'DejaVu Sans'

def safe_arabic_text(text):
    """Safely process Arabic text with fallback to English"""
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception as e:
        
        fallback_map = {
            'مصفوفة الخلط': 'Confusion Matrix',
            'التنبؤ': 'Predicted',
            'القيمة الفعلية': 'Actual',
            'مخطط البواقي': 'Residuals Plot',
            'القيم المتنبأ بها': 'Predicted Values',
            'البواقي (الفعلي - المتنبأ)': 'Residuals (Actual - Predicted)',
            'الفعلي مقابل المتنبأ': 'Actual vs Predicted',
            'الدرجات الفعلية': 'Actual Scores',
            'الدرجات المتنبأ بها': 'Predicted Scores',
            'ممتاز (80+)': 'Excellent (80+)',
            'جيد جداً (70-79)': 'Very Good (70-79)',
            'جيد (60-69)': 'Good (60-69)',
            'مقبول (50-59)': 'Acceptable (50-59)',
            'ضعيف (<50)': 'Poor (<50)'
        }
        
        
        if ' - ' in text:
            parts = text.split(' - ')
            if len(parts) == 2:
                arabic_part = fallback_map.get(parts[0], parts[0])
                return f"{arabic_part} - {parts[1]}"
        
        return fallback_map.get(text, text)

def load_real_data():
    """Load real student performance data from CSV with enhanced features"""
    try:
        
        df = pd.read_csv('Data/StudentsPerformance.csv')
        
        
        df.columns = df.columns.str.strip()
        
        
        score_columns = ['math score', 'reading score', 'writing score']
        for col in score_columns:
            if col in df.columns:
                max_score = df[col].max()
                min_score = df[col].min()
                print(f"Data validation - {col}: min={min_score}, max={max_score}")
                
                if max_score > 100:
                    print(f"⚠️ Warning: {col} has values above 100, capping at 100")
                    df[col] = df[col].clip(0, 100)
                
                if min_score < 0:
                    print(f"⚠️ Warning: {col} has negative values, setting to 0")
                    df[col] = df[col].clip(0, 100)
        
        
        
        return df
        
    except FileNotFoundError:
        print("CSV file not found, using enhanced sample data")
        
        np.random.seed(42)
        n_students = 2000  
        
        
        base_ability = np.random.normal(70, 15, n_students)  
        socioeconomic_factor = np.random.normal(0, 1, n_students)  
        motivation_factor = np.random.normal(0, 1, n_students)  
        
        
        math_scores = np.clip(base_ability + np.random.normal(0, 8, n_students) + 
                             socioeconomic_factor * 5 + motivation_factor * 3, 0, 100)
        reading_scores = np.clip(base_ability + np.random.normal(0, 6, n_students) + 
                               socioeconomic_factor * 4 + motivation_factor * 4, 0, 100)
        writing_scores = np.clip(reading_scores * 0.8 + np.random.normal(0, 4, n_students) + 
                               motivation_factor * 2, 0, 100)  
        
        
        genders = np.random.choice(['male', 'female'], n_students, p=[0.48, 0.52])
        races = np.random.choice(['group A', 'group B', 'group C', 'group D', 'group E'], 
                                n_students, p=[0.15, 0.25, 0.30, 0.20, 0.10])
        
        
        education_levels = []
        lunch_types = []
        
        for i in range(n_students):
            
            if socioeconomic_factor[i] > 1:
                education_levels.append(np.random.choice(['bachelor\'s degree', 'master\'s degree'], p=[0.7, 0.3]))
                lunch_types.append('standard')
                
                math_scores[i] = min(100, math_scores[i] + 5)
                reading_scores[i] = min(100, reading_scores[i] + 5)
                writing_scores[i] = min(100, writing_scores[i] + 5)
            elif socioeconomic_factor[i] > 0:
                education_levels.append(np.random.choice(['some college', 'associate\'s degree', 'bachelor\'s degree'], 
                                                       p=[0.4, 0.4, 0.2]))
                lunch_types.append(np.random.choice(['standard', 'free/reduced'], p=[0.8, 0.2]))
            else:
                education_levels.append(np.random.choice(['some high school', 'high school', 'some college'], 
                                                       p=[0.2, 0.5, 0.3]))
                lunch_types.append(np.random.choice(['standard', 'free/reduced'], p=[0.3, 0.7]))
                
                math_scores[i] = max(0, math_scores[i] - 3)
                reading_scores[i] = max(0, reading_scores[i] - 3)
                writing_scores[i] = max(0, writing_scores[i] - 3)
        
        
        test_prep = []
        for i in range(n_students):
            
            if socioeconomic_factor[i] > 0.5:
                prep_choice = np.random.choice(['none', 'completed'], p=[0.4, 0.6])
            else:
                prep_choice = np.random.choice(['none', 'completed'], p=[0.7, 0.3])
            
            test_prep.append(prep_choice)
            
            
            if prep_choice == 'completed':
                math_scores[i] = min(100, math_scores[i] + np.random.randint(3, 8))
                reading_scores[i] = min(100, reading_scores[i] + np.random.randint(2, 6))
                writing_scores[i] = min(100, writing_scores[i] + np.random.randint(2, 6))
        
        data = {
            'gender': genders,
            'race/ethnicity': races,
            'parental level of education': education_levels,
            'lunch': lunch_types,
            'test preparation course': test_prep,
            'math score': math_scores.astype(int),
            'reading score': reading_scores.astype(int),
            'writing score': writing_scores.astype(int)
        }
        
        df = pd.DataFrame(data)
        
        
        df = add_enhanced_features(df)
        
        return df



def add_enhanced_features(df):
    """Add enhanced features to improve model performance"""
    
    
    df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
    df['average_score'] = df['total_score'] / 3
    
    
    df['math_reading_diff'] = df['math score'] - df['reading score']
    df['reading_writing_diff'] = df['reading score'] - df['writing score']
    df['math_writing_diff'] = df['math score'] - df['writing score']
    
    
    df['score_std'] = df[['math score', 'reading score', 'writing score']].std(axis=1)
    df['is_consistent'] = (df['score_std'] < 10).astype(int)  
    
    
    df['high_math'] = (df['math score'] >= 80).astype(int)
    df['high_reading'] = (df['reading score'] >= 80).astype(int)
    df['high_writing'] = (df['writing score'] >= 80).astype(int)
    df['high_all_subjects'] = (df['high_math'] & df['high_reading'] & df['high_writing']).astype(int)
    
    
    df['math_strongest'] = ((df['math score'] >= df['reading score']) & 
                           (df['math score'] >= df['writing score'])).astype(int)
    df['reading_strongest'] = ((df['reading score'] >= df['math score']) & 
                              (df['reading score'] >= df['writing score'])).astype(int)
    df['writing_strongest'] = ((df['writing score'] >= df['math score']) & 
                              (df['writing score'] >= df['reading score'])).astype(int)
    
    
    df['advantaged_background'] = ((df['lunch'] == 'standard') & 
                                  (df['parental level of education'].isin(['bachelor\'s degree', 'master\'s degree']))).astype(int)
    
    df['disadvantaged_background'] = ((df['lunch'] == 'free/reduced') & 
                                     (df['parental level of education'].isin(['some high school', 'high school']))).astype(int)
    
    
    df['prep_with_support'] = ((df['test preparation course'] == 'completed') & 
                              (df['advantaged_background'] == 1)).astype(int)
    
    
    df['female_language_advantage'] = ((df['gender'] == 'female') & 
                                      ((df['reading score'] > df['math score']) | 
                                       (df['writing score'] > df['math score']))).astype(int)
    
    df['male_math_advantage'] = ((df['gender'] == 'male') & 
                                (df['math score'] > df[['reading score', 'writing score']].max(axis=1))).astype(int)
    
    
    df['excellent_performer'] = (df['average_score'] >= 90).astype(int)
    df['good_performer'] = ((df['average_score'] >= 70) & (df['average_score'] < 90)).astype(int)
    df['average_performer'] = ((df['average_score'] >= 50) & (df['average_score'] < 70)).astype(int)
    df['at_risk'] = (df['average_score'] < 50).astype(int)
    
    
    for subject in ['math score', 'reading score', 'writing score']:
        df[f'{subject}_normalized'] = (df[subject] - df[subject].mean()) / df[subject].std()
    
    
    df['math_percentile'] = df['math score'].rank(pct=True)
    df['reading_percentile'] = df['reading score'].rank(pct=True)
    df['writing_percentile'] = df['writing score'].rank(pct=True)
    df['overall_percentile'] = df['average_score'].rank(pct=True)
    
    
    df['balanced_performance'] = (df['score_std'] < 8).astype(int)  
    df['language_focused'] = ((df['reading score'] + df['writing score']) > (2 * df['math score'])).astype(int)
    df['math_focused'] = (df['math score'] > (df['reading score'] + df['writing score']) / 2).astype(int)
    
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning with optimization"""
    
    df_clean = df.copy()
    
    
    df_clean = df_clean.dropna()
    
    
    categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    
    for col in categorical_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_education = LabelEncoder()
    le_lunch = LabelEncoder()
    le_prep = LabelEncoder()
    
    
    all_genders = ['male', 'female']
    all_races = ['group a', 'group b', 'group c', 'group d', 'group e']
    all_education = ['some high school', 'high school', 'some college', 'associate\'s degree', 'bachelor\'s degree', 'master\'s degree']
    all_lunch = ['standard', 'free/reduced']
    all_prep = ['none', 'completed']
    
    
    le_gender.fit(all_genders)
    le_race.fit(all_races)
    le_education.fit(all_education)
    le_lunch.fit(all_lunch)
    le_prep.fit(all_prep)
    
    
    df_encoded = df_clean.copy()
    
    try:
        df_encoded['gender'] = le_gender.transform(df_clean['gender'])
        df_encoded['race/ethnicity'] = le_race.transform(df_clean['race/ethnicity'])
        df_encoded['parental level of education'] = le_education.transform(df_clean['parental level of education'])
        df_encoded['lunch'] = le_lunch.transform(df_clean['lunch'])
        df_encoded['test preparation course'] = le_prep.transform(df_clean['test preparation course'])
    except ValueError as e:
        print(f"Encoding error: {e}")
        
        for col, encoder in zip(categorical_columns, [le_gender, le_race, le_education, le_lunch, le_prep]):
            if col in df_clean.columns:
                unique_values = df_clean[col].unique()
                unknown_values = [val for val in unique_values if val not in encoder.classes_]
                if unknown_values:
                    print(f"Unknown values in {col}: {unknown_values}")
                    
                    df_clean[col] = df_clean[col].replace(unknown_values, encoder.classes_[0])
                
                df_encoded[col] = encoder.transform(df_clean[col])
    
    
    feature_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    
    return df_encoded, (le_gender, le_race, le_education, le_lunch, le_prep), feature_columns
@api_view(['POST'])
def train_model(request):
    """Train optimized prediction models with real data"""
    try:
        
        selected_font = setup_arabic_matplotlib()
        
        
        df = load_real_data()
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Unique races: {df['race/ethnicity'].unique()}")
        print(f"Unique education levels: {df['parental level of education'].unique()}")
        
        
        df_encoded, encoders, feature_columns = preprocess_data(df)
        
        
        
        demographic_features = [
            'gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'
        ]
        
        
        X = df_encoded[demographic_features].copy()
        
        print(f"Training features: {demographic_features}")
        print(f"Training data shape: {X.shape}")
        
        
        X['gender_race'] = X['gender'] * X['race/ethnicity']
        X['education_prep'] = X['parental level of education'] * X['test preparation course']
        X['lunch_prep'] = X['lunch'] * X['test preparation course']
        X['gender_education'] = X['gender'] * X['parental level of education']
        X['race_education'] = X['race/ethnicity'] * X['parental level of education']
        
        
        feature_columns = X.columns.tolist()
        
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        
        models = {}
        scores = {}
        performance_metrics = {}
        
        
        X_train, X_test, y_train_all, y_test_all = train_test_split(
            X_scaled, df_encoded[['math score', 'reading score', 'writing score']], 
            test_size=0.15, random_state=42, shuffle=True
        )
        
        for i, subject in enumerate(['math score', 'reading score', 'writing score']):
            y_train = y_train_all.iloc[:, i]
            y_test = y_test_all.iloc[:, i]
            
            
            
            
            gb_model = GradientBoostingRegressor(
                n_estimators=500,     
                learning_rate=0.03,   
                max_depth=6,          
                min_samples_split=15, 
                min_samples_leaf=8,   
                subsample=0.85,       
                max_features='sqrt',  
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=20   
            )
            
            
            rf_model = RandomForestRegressor(
                n_estimators=300,     
                max_depth=12,         
                min_samples_split=8,  
                min_samples_leaf=3,   
                max_features='sqrt',  
                bootstrap=True,
                oob_score=True,       
                random_state=42,
                n_jobs=-1
            )
            
            
            from sklearn.linear_model import Ridge
            ridge_model = Ridge(alpha=1.0, random_state=42)
            
            
            from sklearn.ensemble import ExtraTreesRegressor
            et_model = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            
            model = VotingRegressor([
                ('gb', gb_model),
                ('rf', rf_model),
                ('ridge', ridge_model),
                ('et', et_model)
            ])
            
            model.fit(X_train, y_train)
            
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            
            train_accuracy_10 = np.mean(np.abs(y_train - y_pred_train) <= 10) * 100
            test_accuracy_10 = np.mean(np.abs(y_test - y_pred_test) <= 10) * 100
            
            
            train_accuracy_5 = np.mean(np.abs(y_train - y_pred_train) <= 5) * 100
            test_accuracy_5 = np.mean(np.abs(y_test - y_pred_test) <= 5) * 100
            
            
            def categorize_scores(scores):
                return np.where(scores >= 80, 'ممتاز (80+)',
                       np.where(scores >= 70, 'جيد جداً (70-79)',
                       np.where(scores >= 60, 'جيد (60-69)',
                       np.where(scores >= 50, 'مقبول (50-59)', 'ضعيف (<50)'))))
            
            y_test_categories = categorize_scores(y_test)
            y_pred_categories = categorize_scores(y_pred_test)
            
            
            cm = confusion_matrix(y_test_categories, y_pred_categories)
            
            
            plt.figure(figsize=(12, 10))
            
            
            arabic_labels = ['ممتاز (80+)', 'جيد جداً (70-79)', 'جيد (60-69)', 'مقبول (50-59)', 'ضعيف (<50)']
            reshaped_labels = []
            for label in arabic_labels:
                safe_label = safe_arabic_text(label)
                reshaped_labels.append(safe_label)
            
            
            sns.heatmap(cm, 
                       annot=True,           
                       fmt='d',              
                       cmap='Blues',         
                       annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'},  
                       cbar_kws={'shrink': 0.8},  
                       square=True,          
                       linewidths=2,         
                       linecolor='white',    
                       xticklabels=reshaped_labels,
                       yticklabels=reshaped_labels,
                       vmin=0,               
                       robust=True)          
            
            
            for i in range(len(cm)):
                for j in range(len(cm[0])):
                    plt.text(j + 0.5, i + 0.5, str(cm[i][j]), 
                            ha='center', va='center', 
                            fontsize=18, fontweight='bold', 
                            color='black' if cm[i][j] < cm.max()/2 else 'white')
            
            
            title_text = safe_arabic_text(f'مصفوفة الخلط - {subject}')
            plt.title(title_text, fontsize=14, pad=20)
            
            xlabel_text = safe_arabic_text('التنبؤ')
            plt.xlabel(xlabel_text, fontsize=10)
            
            ylabel_text = safe_arabic_text('القيمة الفعلية')
            plt.ylabel(ylabel_text, fontsize=10)
            
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            confusion_matrix_img = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            
            plt.figure(figsize=(10, 6))
            
            residuals = y_test - y_pred_test
            plt.scatter(y_pred_test, residuals, alpha=0.6, color='#B85C57')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            
            
            xlabel_text = safe_arabic_text('القيم المتنبأ بها')
            plt.xlabel(xlabel_text, fontsize=10)
            
            ylabel_text = safe_arabic_text('البواقي (الفعلي - المتنبأ)')
            plt.ylabel(ylabel_text, fontsize=10)
            
            title_text = safe_arabic_text(f'مخطط البواقي - {subject}')
            plt.title(title_text, fontsize=12)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            
            buffer2 = BytesIO()
            plt.savefig(buffer2, format='png', dpi=150, bbox_inches='tight')
            buffer2.seek(0)
            residuals_img = base64.b64encode(buffer2.getvalue()).decode()
            plt.close()
            
            
            plt.figure(figsize=(10, 8))
            
            plt.scatter(y_test, y_pred_test, alpha=0.6, color='#B85C57')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            
            
            xlabel_text = safe_arabic_text('الدرجات الفعلية')
            plt.xlabel(xlabel_text, fontsize=10)
            
            ylabel_text = safe_arabic_text('الدرجات المتنبأ بها')
            plt.ylabel(ylabel_text, fontsize=10)
            
            title_text = safe_arabic_text(f'الفعلي مقابل المتنبأ - {subject}')
            plt.title(title_text, fontsize=12)
            
            plt.grid(True, alpha=0.3)
            
            
            plt.text(0.05, 0.95, f'R² = {test_r2:.3f}', transform=plt.gca().transAxes, 
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.tight_layout()
            
            
            buffer3 = BytesIO()
            plt.savefig(buffer3, format='png', dpi=150, bbox_inches='tight')
            buffer3.seek(0)
            actual_vs_pred_img = base64.b64encode(buffer3.getvalue()).decode()
            plt.close()
            
            models[subject] = model
            scores[subject] = test_r2
            
            performance_metrics[subject] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_accuracy_10': train_accuracy_10,
                'test_accuracy_10': test_accuracy_10,
                'train_accuracy_5': train_accuracy_5,
                'test_accuracy_5': test_accuracy_5,
                'confusion_matrix': cm.tolist(),
                'confusion_matrix_img': confusion_matrix_img,
                'residuals_img': residuals_img,
                'actual_vs_pred_img': actual_vs_pred_img,
                'feature_importance': dict(zip(feature_columns, 
                    
                    (model.named_estimators_['gb'].feature_importances_ + 
                     model.named_estimators_['rf'].feature_importances_ +
                     model.named_estimators_['et'].feature_importances_) / 3
                )),
                'classification_report': classification_report(y_test_categories, y_pred_categories, output_dict=True)
            }
            
            
            model_path = f'models/{subject.replace(" ", "_")}_model.joblib'
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, model_path)
            
            print(f"{subject} - R²: {test_r2:.3f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")
        
        
        joblib.dump(encoders, 'models/encoders.joblib')
        joblib.dump(scaler, 'models/scaler.joblib')
        joblib.dump(feature_columns, 'models/feature_columns.joblib')
        
        
        joblib.dump(performance_metrics, 'models/performance_metrics.joblib')
        
        return Response({
            'message': 'Optimized models trained successfully with real data',
            'scores': scores,
            'performance_metrics': performance_metrics,
            'data_info': {
                'total_records': len(df),
                'features_used': feature_columns,
                'unique_races': df['race/ethnicity'].unique().tolist(),
                'unique_education': df['parental level of education'].unique().tolist()
            }
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Training error: {error_details}")
        return Response({
            'error': f'خطأ في التدريب: {str(e)}',
            'details': error_details,
            'message': 'حدث خطأ أثناء تدريب النموذج'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
@api_view(['POST', 'GET'])
def predict_scores(request):
    """Predict student scores with enhanced models support"""
    try:
        
        if request.method == 'POST':
            data = request.data
        else:  
            data = {
                'gender': request.GET.get('gender', ''),
                'race_ethnicity': request.GET.get('race_ethnicity', ''),
                'parental_education': request.GET.get('parental_education', ''),
                'lunch': request.GET.get('lunch', ''),
                'test_preparation': request.GET.get('test_preparation', '')
            }
        
        print(f"Received prediction request ({request.method}): {data}")
        
        
        required_fields = ['gender', 'race_ethnicity', 'parental_education', 'lunch', 'test_preparation']
        for field in required_fields:
            if field not in data or not data[field]:
                return Response({
                    'error': f'يرجى ملء جميع الحقول المطلوبة: {field}',
                    'message': 'جميع البيانات مطلوبة للتنبؤ الدقيق'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        
        regular_model_files = [
            'models/math_score_model.joblib',
            'models/reading_score_model.joblib', 
            'models/writing_score_model.joblib',
            'models/encoders.joblib',
            'models/scaler.joblib',
            'models/feature_columns.joblib'
        ]
        
        
        use_regular = all(os.path.exists(f) for f in regular_model_files)
        
        if not use_regular:
            return Response({
                'error': 'يجب تدريب النموذج أولاً قبل التنبؤ',
                'message': 'اضغط على زر "تدريب النموذج" أولاً لإنشاء النماذج المطلوبة',
                'action_required': 'TRAIN_MODEL_FIRST'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        
        try:
            print("📊 استخدام النماذج العادية")
            math_model = joblib.load('models/math_score_model.joblib')
            reading_model = joblib.load('models/reading_score_model.joblib')
            writing_model = joblib.load('models/writing_score_model.joblib')
            encoders = joblib.load('models/encoders.joblib')
            scaler = joblib.load('models/scaler.joblib')
            feature_columns = joblib.load('models/feature_columns.joblib')
            scalers = {'default': scaler}
            
            
            try:
                performance_metrics = joblib.load('models/performance_metrics.joblib')
            except FileNotFoundError:
                performance_metrics = None
                
        except Exception as e:
            return Response({
                'error': 'خطأ في تحميل النماذج المدربة',
                'message': 'يرجى إعادة تدريب النموذج',
                'details': str(e),
                'action_required': 'RETRAIN_MODEL'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        
        
        le_gender, le_race, le_education, le_lunch, le_prep = encoders
        
        
        clean_data = {
            'gender': str(data.get('gender', '')).strip().lower(),
            'race/ethnicity': str(data.get('race_ethnicity', '')).strip().lower().replace('_', ' ').replace('+', ' '),
            'parental level of education': str(data.get('parental_education', '')).strip().lower().replace('%27', "'").replace('+', ' ').replace('associate%27s', "associate's").replace('bachelor%27s', "bachelor's").replace('master%27s', "master's"),
            'lunch': str(data.get('lunch', '')).strip().lower(),
            'test preparation course': str(data.get('test_preparation', '')).strip().lower()
        }
        
        
        if clean_data.get('lunch', '') == 'free':
            clean_data['lunch'] = 'free/reduced'
        elif clean_data.get('lunch', '') == 'reduced':
            clean_data['lunch'] = 'free/reduced'
        
        
        if 'associate\'s' in clean_data.get('parental level of education', ''):
            clean_data['parental level of education'] = 'associate\'s degree'
        elif 'bachelor\'s' in clean_data.get('parental level of education', ''):
            clean_data['parental level of education'] = 'bachelor\'s degree'
        elif 'master\'s' in clean_data.get('parental level of education', ''):
            clean_data['parental level of education'] = 'master\'s degree'
            
        print(f"Cleaned data: {clean_data}")
        
        
        validation_errors = []
        
        if clean_data.get('gender', '') not in le_gender.classes_:
            validation_errors.append(f"جنس غير صحيح: {data.get('gender', '')}. القيم المسموحة: {', '.join(le_gender.classes_)}")
            
        if clean_data.get('race/ethnicity', '') not in le_race.classes_:
            validation_errors.append(f"مجموعة عرقية غير صحيحة: {data.get('race_ethnicity', '')}. القيم المسموحة: {', '.join(le_race.classes_)}")
            
        if clean_data.get('parental level of education', '') not in le_education.classes_:
            validation_errors.append(f"مستوى تعليم غير صحيح: {data.get('parental_education', '')}. القيم المسموحة: {', '.join(le_education.classes_)}")
            
        if clean_data.get('lunch', '') not in le_lunch.classes_:
            validation_errors.append(f"نوع وجبة غير صحيح: {data.get('lunch', '')}. القيم المسموحة: {', '.join(le_lunch.classes_)}")
            
        if clean_data.get('test preparation course', '') not in le_prep.classes_:
            validation_errors.append(f"حالة التحضير غير صحيحة: {data.get('test_preparation', '')}. القيم المسموحة: {', '.join(le_prep.classes_)}")
        
        if validation_errors:
            return Response({
                'error': 'بيانات الإدخال غير صحيحة',
                'validation_errors': validation_errors,
                'available_options': {
                    'gender': le_gender.classes_.tolist(),
                    'race_ethnicity': le_race.classes_.tolist(),
                    'parental_education': le_education.classes_.tolist(),
                    'lunch': le_lunch.classes_.tolist(),
                    'test_preparation': le_prep.classes_.tolist()
                }
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            
            encoded_data = [
                le_gender.transform([clean_data.get('gender', '')])[0],
                le_race.transform([clean_data.get('race/ethnicity', '')])[0],
                le_education.transform([clean_data.get('parental level of education', '')])[0],
                le_lunch.transform([clean_data.get('lunch', '')])[0],
                le_prep.transform([clean_data.get('test preparation course', '')])[0]
            ]
            
            print(f"Encoded data: {encoded_data}")
            
            
            input_df = pd.DataFrame([encoded_data], columns=[
                'gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'
            ])
            
            
            input_df['gender_race'] = input_df['gender'] * input_df['race/ethnicity']
            input_df['education_prep'] = input_df['parental level of education'] * input_df['test preparation course']
            input_df['lunch_prep'] = input_df['lunch'] * input_df['test preparation course']
            input_df['gender_education'] = input_df['gender'] * input_df['parental level of education']
            input_df['race_education'] = input_df['race/ethnicity'] * input_df['parental level of education']
            
            
            expected_columns = [
                'gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course',
                'gender_race', 'education_prep', 'lunch_prep', 'gender_education', 'race_education'
            ]
            
            
            input_df = input_df.reindex(columns=expected_columns, fill_value=0)
            
            print(f"Final feature vector shape: {input_df.shape}")
            print(f"Feature columns: {len(feature_columns)}")
            
        except Exception as encoding_error:
            return Response({
                'error': f'خطأ في معالجة البيانات: {str(encoding_error)}',
                'message': 'حدث خطأ أثناء تحضير البيانات للتنبؤ'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        
        input_array = input_df.values
        
        print(f"Input array shape: {input_array.shape}")
        
        
        
        input_scaled = scalers['default'].transform(input_array)
        print(f"Scaled input shape: {input_scaled.shape}")
        
        math_pred = float(math_model.predict(input_scaled)[0])
        reading_pred = float(reading_model.predict(input_scaled)[0])
        writing_pred = float(writing_model.predict(input_scaled)[0])
        
        print(f"Raw predictions - Math: {math_pred:.2f}, Reading: {reading_pred:.2f}, Writing: {writing_pred:.2f}")
        
        
        math_pred = max(0, min(100, round(math_pred, 1)))
        reading_pred = max(0, min(100, round(reading_pred, 1)))
        writing_pred = max(0, min(100, round(writing_pred, 1)))
        
        
        raw_math = float(math_model.predict(input_scaled)[0])
        raw_reading = float(reading_model.predict(input_scaled)[0])
        raw_writing = float(writing_model.predict(input_scaled)[0])
        
        if raw_math > 100 or raw_reading > 100 or raw_writing > 100:
            print(f"⚠️ Warning: Raw predictions exceeded 100 - Math: {raw_math:.1f}, Reading: {raw_reading:.1f}, Writing: {raw_writing:.1f}")
            print("Predictions have been capped at 100")
        
        
        total_score = math_pred + reading_pred + writing_pred
        average_score = total_score / 3
        
        
        if average_score >= 85:
            performance_level = "ممتاز"
            performance_color = "#28a745"
        elif average_score >= 75:
            performance_level = "جيد جداً"
            performance_color = "#17a2b8"
        elif average_score >= 65:
            performance_level = "جيد"
            performance_color = "#ffc107"
        elif average_score >= 50:
            performance_level = "مقبول"
            performance_color = "#fd7e14"
        else:
            performance_level = "يحتاج تحسين"
            performance_color = "#dc3545"
        
        
        scores = {'الرياضيات': math_pred, 'القراءة': reading_pred, 'الكتابة': writing_pred}
        strongest_subject = max(scores, key=scores.get)
        weakest_subject = min(scores, key=scores.get)
        
        
        recommendations = []
        
        if math_pred < 60:
            recommendations.append("يُنصح بالتركيز على تقوية مهارات الرياضيات من خلال حل المزيد من التمارين")
        if reading_pred < 60:
            recommendations.append("يُنصح بزيادة وقت القراءة اليومية لتحسين مهارات الفهم")
        if writing_pred < 60:
            recommendations.append("يُنصح بممارسة الكتابة بانتظام وتعلم قواعد اللغة")
            
        if clean_data.get('test preparation course', '') == 'none' and average_score < 75:
            recommendations.append("يُنصح بشدة بالالتحاق بدورة تحضيرية للامتحانات")
            
        if not recommendations:
            recommendations.append("أداء ممتاز! استمر في المحافظة على هذا المستوى")
        
        predictions = {
            'math_score': math_pred,
            'reading_score': reading_pred,
            'writing_score': writing_pred,
            'total_score': round(total_score, 1),
            'average_score': round(average_score, 1),
            'performance_level': performance_level,
            'performance_color': performance_color,
            'strongest_subject': strongest_subject,
            'weakest_subject': weakest_subject,
            'recommendations': recommendations
        }
        
        
        response_data = {
            'success': True,
            'predictions': predictions,
            'input_data': data
        }
        
        if performance_metrics:
            response_data['model_performance'] = {
                'math_accuracy': performance_metrics['math score']['test_r2'],
                'reading_accuracy': performance_metrics['reading score']['test_r2'],
                'writing_accuracy': performance_metrics['writing score']['test_r2'],
                'math_mae': performance_metrics['math score']['test_mae'],
                'reading_mae': performance_metrics['reading score']['test_mae'],
                'writing_mae': performance_metrics['writing score']['test_mae'],
                'math_r2': performance_metrics['math score']['test_r2'],
                'reading_r2': performance_metrics['reading score']['test_r2'],
                'writing_r2': performance_metrics['writing score']['test_r2']
            }
        
        return Response(response_data)
    
    except KeyError as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        missing_key = str(e).strip("'")
        return Response({
            'error': f'بيانات ناقصة: يرجى التأكد من ملء جميع الحقول المطلوبة ({missing_key})',
            'message': 'حدث خطأ في معالجة البيانات. يرجى التأكد من ملء جميع الحقول بشكل صحيح وإعادة المحاولة.'
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        error_msg = str(e)
        if 'KeyError' in error_msg or 'parental_education' in error_msg:
            error_msg = 'حدث خطأ في معالجة البيانات. يرجى التأكد من ملء جميع الحقول بشكل صحيح وإعادة المحاولة.'
        return Response({
            'error': error_msg,
            'message': 'حدث خطأ أثناء التنبؤ. يرجى المحاولة مرة أخرى.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
@api_view(['GET'])
def get_model_performance(request):
    """Get detailed model performance metrics and visualizations"""
    try:
        
        performance_metrics = None
        metrics_type = 'regular'
        
        try:
            performance_metrics = joblib.load('models/performance_metrics.joblib')
            print("📊 تم تحميل مقاييس الأداء")
        except FileNotFoundError:
            return Response({
                'error': 'لم يتم العثور على مقاييس الأداء. يرجى تدريب النموذج أولاً.',
                'message': 'قم بتدريب النموذج لإنشاء مقاييس الأداء'
            }, status=status.HTTP_404_NOT_FOUND)
        
        
        if performance_metrics:
            summary_stats = {
                'average_r2': np.mean([metrics['test_r2'] for metrics in performance_metrics.values()]),
                'average_mae': np.mean([metrics['test_mae'] for metrics in performance_metrics.values()]),
                'average_rmse': np.mean([metrics['test_rmse'] for metrics in performance_metrics.values()]),
                'best_subject': max(performance_metrics.keys(), key=lambda k: performance_metrics[k]['test_r2']),
                'worst_subject': min(performance_metrics.keys(), key=lambda k: performance_metrics[k]['test_r2']),
                'metrics_type': metrics_type
            }
        
        return Response({
            'performance_metrics': performance_metrics,
            'summary_stats': summary_stats,
            'metrics_type': metrics_type,
            'message': 'تم تحميل مقاييس الأداء بنجاح'
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Performance metrics error: {error_details}")
        
        return Response({
            'error': str(e),
            'details': error_details
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def check_model_status(request):
    """Check if models are trained and ready for prediction"""
    try:
        
        regular_model_files = [
            'models/math_score_model.joblib',
            'models/reading_score_model.joblib', 
            'models/writing_score_model.joblib',
            'models/encoders.joblib',
            'models/scaler.joblib',
            'models/feature_columns.joblib'
        ]
        
        regular_ready = all(os.path.exists(f) for f in regular_model_files)
        
        
        file_status = {}
        for file_path in regular_model_files:
            file_status[file_path] = {
                'exists': os.path.exists(file_path),
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        
        if regular_ready:
            
            try:
                test_encoders = joblib.load('models/encoders.joblib')
                test_scaler = joblib.load('models/scaler.joblib')
                
                return Response({
                    'models_ready': True,
                    'model_type': 'regular',
                    'message': 'النماذج جاهزة للتنبؤ',
                    'file_status': file_status,
                    'encoders_classes': {
                        'gender': test_encoders[0].classes_.tolist(),
                        'race_ethnicity': test_encoders[1].classes_.tolist(),
                        'parental_education': test_encoders[2].classes_.tolist(),
                        'lunch': test_encoders[3].classes_.tolist(),
                        'test_preparation': test_encoders[4].classes_.tolist()
                    }
                })
            except Exception as load_error:
                return Response({
                    'models_ready': False,
                    'message': f'النماذج موجودة لكن لا يمكن تحميلها: {str(load_error)}',
                    'file_status': file_status,
                    'recommendation': 'قم بإعادة تدريب النماذج'
                })
        else:
            missing_regular = [f for f in regular_model_files if not os.path.exists(f)]
            
            return Response({
                'models_ready': False,
                'message': 'لا توجد نماذج مدربة',
                'missing_regular_files': missing_regular,
                'file_status': file_status,
                'recommendation': 'قم بتدريب النماذج للحصول على التنبؤات'
            })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Model status check error: {error_details}")
        
        return Response({
            'models_ready': False,
            'error': str(e),
            'details': error_details
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def prediction_dashboard(request):
    """Render the prediction dashboard"""
    return render(request, 'prediction/dashboard.html')

@api_view(['GET'])
def download_performance_report(request):
    """Generate and download PDF performance report"""
    try:
        
        performance_metrics = joblib.load('models/performance_metrics.joblib')
        
        
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="performance_report.pdf"'
        
        
        doc = SimpleDocTemplate(response, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        
        elements = []
        
        
        styles = getSampleStyleSheet()
        
        
        arabic_style = ParagraphStyle(
            'ArabicStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=12,
            alignment=2,  
            spaceAfter=12,
        )
        
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontName='Helvetica-Bold',
            fontSize=16,
            alignment=1,  
            spaceAfter=20,
        )
        
        
        title = Paragraph("AI Education System - Performance Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        
        arabic_title_text = "تقرير أداء نظام الذكاء الاصطناعي التعليمي"
        safe_title = safe_arabic_text(arabic_title_text)
        arabic_title = Paragraph(safe_title, arabic_style)
        elements.append(arabic_title)
        elements.append(Spacer(1, 20))
        
        
        summary_data = [['Subject', 'R² Score', 'MAE', 'Accuracy ±10', 'Accuracy ±5']]
        
        for subject in performance_metrics:
            metric = performance_metrics[subject]
            summary_data.append([
                subject,
                f"{metric['test_r2']:.3f} ({metric['test_r2']*100:.1f}%)",
                f"{metric['test_mae']:.2f}",
                f"{metric['test_accuracy_10']:.1f}%",
                f"{metric['test_accuracy_5']:.1f}%"
            ])
        
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        
        for subject in performance_metrics:
            metric = performance_metrics[subject]
            if 'confusion_matrix_img' in metric:
                
                subject_header = Paragraph(f"Confusion Matrix - {subject}", styles['Heading2'])
                elements.append(subject_header)
                elements.append(Spacer(1, 12))
                
                
                img_data = base64.b64decode(metric['confusion_matrix_img'])
                img_buffer = BytesIO(img_data)
                
                
                img = Image(img_buffer, width=4*inch, height=3*inch)
                elements.append(img)
                elements.append(Spacer(1, 20))
        
        
        importance_header = Paragraph("Feature Importance Analysis", styles['Heading2'])
        elements.append(importance_header)
        elements.append(Spacer(1, 12))
        
        for subject in performance_metrics:
            metric = performance_metrics[subject]
            if 'feature_importance' in metric:
                
                subject_subheader = Paragraph(f"{subject}", styles['Heading3'])
                elements.append(subject_subheader)
                
                
                importance_data = [['Feature', 'Importance']]
                for feature, importance in metric['feature_importance'].items():
                    
                    feature_translations = {
                        'gender': 'الجنس',
                        'race/ethnicity': 'العرق/الإثنية',
                        'parental level of education': 'مستوى تعليم الوالدين',
                        'lunch': 'نوع الوجبة',
                        'test preparation course': 'دورة التحضير للاختبار',
                        'gender_race': 'تفاعل الجنس والعرق',
                        'education_prep': 'تفاعل التعليم والتحضير',
                        'lunch_prep': 'تفاعل الوجبة والتحضير'
                    }
                    
                    arabic_feature = feature_translations.get(feature, feature)
                    safe_feature = safe_arabic_text(arabic_feature)
                    
                    importance_data.append([
                        safe_feature,
                        f"{importance*100:.1f}%"
                    ])
                
                importance_table = Table(importance_data)
                importance_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(importance_table)
                elements.append(Spacer(1, 15))
        
        
        doc.build(elements)
        
        return response
        
    except FileNotFoundError:
        return Response({
            'error': 'لم يتم العثور على مقاييس الأداء. يرجى تدريب النموذج أولاً.',
            'message': 'قم بتدريب النموذج لإنشاء مقاييس الأداء'
        }, status=status.HTTP_404_NOT_FOUND)
    
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)