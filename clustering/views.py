from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import json
import plotly

def load_real_data():
    """Load real student performance data for clustering"""
    try:
        
        df = pd.read_csv('Data/StudentsPerformance.csv')
        
        
        df.columns = df.columns.str.strip()
        
        
        score_columns = ['math score', 'reading score', 'writing score']
        for col in score_columns:
            if col in df.columns:
                max_score = df[col].max()
                min_score = df[col].min()
                print(f"Clustering data validation - {col}: min={min_score}, max={max_score}")
                
                
                if max_score > 100 or min_score < 0:
                    print(f"⚠️ Data issue in {col}, applying bounds [0, 100]")
                    df[col] = df[col].clip(0, 100)
        
        
        df['student_id'] = range(1, len(df) + 1)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
        
        return df
        
    except FileNotFoundError:
        print("CSV file not found, using sample data")
        
        np.random.seed(42)
        n_students = 1000
        
        data = {
            'student_id': range(1, n_students + 1),
            'math score': np.random.randint(0, 100, n_students),
            'reading score': np.random.randint(0, 100, n_students),
            'writing score': np.random.randint(0, 100, n_students),
            'gender_encoded': np.random.choice([0, 1], n_students),
            'race/ethnicity_encoded': np.random.choice([0, 1, 2, 3, 4], n_students),
            'parental level of education_encoded': np.random.choice([0, 1, 2, 3, 4, 5], n_students),
            'lunch_encoded': np.random.choice([0, 1], n_students),
            'test preparation course_encoded': np.random.choice([0, 1], n_students)
        }
        
        return pd.DataFrame(data)

@api_view(['POST'])
def perform_clustering(request):
    """Perform K-means clustering on student data"""
    try:
        
        n_clusters = request.data.get('n_clusters', 4)
        
        
        df = load_real_data()
        
        
        features = ['math score', 'reading score', 'writing score', 
                   'gender_encoded', 'race/ethnicity_encoded', 
                   'parental level of education_encoded', 'lunch_encoded', 
                   'test preparation course_encoded']
        X = df[features]
        
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        
        df['cluster'] = cluster_labels
        
        
        cluster_stats = []
        
        
        temp_stats = []
        for i in range(n_clusters):
            cluster_data = df[df['cluster'] == i]
            avg_math = float(cluster_data['math score'].mean())
            avg_reading = float(cluster_data['reading score'].mean())
            avg_writing = float(cluster_data['writing score'].mean())
            avg_total = (avg_math + avg_reading + avg_writing) / 3
            
            temp_stats.append({
                'cluster_id': i,
                'avg_total': avg_total,
                'avg_math_score': avg_math,
                'avg_reading_score': avg_reading,
                'avg_writing_score': avg_writing,
                'student_count': len(cluster_data),
                'cluster_data': cluster_data
            })
        
        
        temp_stats.sort(key=lambda x: x['avg_total'], reverse=True)
        
        
        performance_names = [
            "المتفوقون",           
            "الجيدون",             
            "المتوسطون",           
            "المحتاجون للتحسين",    
            "المحتاجون للدعم"       
        ]
        
        for rank, cluster_info in enumerate(temp_stats):
            cluster_name = performance_names[rank] if rank < len(performance_names) else f"المجموعة {rank+1}"
            
            stats = {
                'cluster_id': cluster_info['cluster_id'],
                'cluster_name': cluster_name,
                'student_count': cluster_info['student_count'],
                'avg_math_score': cluster_info['avg_math_score'],
                'avg_reading_score': cluster_info['avg_reading_score'],
                'avg_writing_score': cluster_info['avg_writing_score'],
                'avg_total': cluster_info['avg_total'],
                'performance_rank': rank + 1,
                'description': generate_cluster_description_by_performance(cluster_info['avg_total'], rank),
                'gender_distribution': get_gender_distribution(cluster_info['cluster_data']),
                'race_distribution': get_race_distribution(cluster_info['cluster_data']),
                'education_distribution': get_education_distribution(cluster_info['cluster_data']),
                'lunch_distribution': get_lunch_distribution(cluster_info['cluster_data']),
                'prep_distribution': get_prep_distribution(cluster_info['cluster_data'])
            }
            cluster_stats.append(stats)
        
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        visualization_data = {
            'x': X_pca[:, 0].tolist(),
            'y': X_pca[:, 1].tolist(),
            'cluster': cluster_labels.tolist(),
            'math_scores': df['math score'].tolist(),
            'reading_scores': df['reading score'].tolist(),
            'writing_scores': df['writing score'].tolist()
        }
        
        
        correlation_data = generate_correlation_analysis(df)
        distribution_data = generate_score_distributions(df)
        demographic_data = generate_demographic_analysis(df)
        
        return Response({
            'cluster_stats': cluster_stats,
            'visualization_data': visualization_data,
            'correlation_data': correlation_data,
            'distribution_data': distribution_data,
            'demographic_data': demographic_data,
            'total_students': len(df)
        })
    
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def generate_cluster_description(cluster_id, cluster_data):
    """Generate Arabic description for each cluster"""
    avg_math = cluster_data['math score'].mean()
    avg_reading = cluster_data['reading score'].mean()
    avg_writing = cluster_data['writing score'].mean()
    avg_total = (avg_math + avg_reading + avg_writing) / 3
    
    if avg_total > 85:
        return "طلاب متفوقون بدرجات عالية في جميع المواد ويحتاجون لتحديات إضافية"
    elif avg_total > 70:
        return "طلاب بأداء جيد يمكن تطوير مهاراتهم أكثر بالتوجيه المناسب"
    elif avg_total > 55:
        return "طلاب بأداء متوسط يحتاجون لدعم إضافي في بعض المواد"
    else:
        return "طلاب يحتاجون لتدخل فوري ودعم مكثف لتحسين أدائهم الأكاديمي"

def generate_cluster_description_by_performance(avg_total, rank):
    """Generate description based on actual performance and ranking"""
    if rank == 0:  
        if avg_total > 85:
            return "طلاب متفوقون بدرجات ممتازة في جميع المواد - يحتاجون لأنشطة إثرائية وتحديات متقدمة"
        elif avg_total > 75:
            return "طلاب بأداء جيد جداً - يمكن تطويرهم ليصبحوا متفوقين بالتوجيه المناسب"
        else:
            return "طلاب بأداء جيد نسبياً - الأفضل في مجموعتهم ولكن يحتاجون لتحسين"
    
    elif rank == 1:  
        if avg_total > 70:
            return "طلاب بأداء جيد - لديهم إمكانات للوصول للتفوق بالدعم المناسب"
        else:
            return "طلاب بأداء متوسط إلى جيد - يحتاجون لتعزيز نقاط القوة"
    
    elif rank == 2:  
        if avg_total > 60:
            return "طلاب بأداء متوسط - يحتاجون لدعم متوازن في جميع المواد"
        else:
            return "طلاب بأداء أقل من المتوسط - يحتاجون لخطة تحسين شاملة"
    
    elif rank == 3:  
        return "طلاب يحتاجون لتحسين كبير - يتطلبون دعماً مكثفاً ومتابعة دقيقة"
    
    else:  
        return "طلاب يحتاجون لتدخل فوري ودعم مكثف - أولوية قصوى للمساعدة الأكاديمية"

def get_gender_distribution(cluster_data):
    """Get gender distribution for cluster"""
    if 'gender' in cluster_data.columns:
        return cluster_data['gender'].value_counts().to_dict()
    return {}

def get_race_distribution(cluster_data):
    """Get race/ethnicity distribution for cluster"""
    if 'race/ethnicity' in cluster_data.columns:
        return cluster_data['race/ethnicity'].value_counts().to_dict()
    return {}

def get_education_distribution(cluster_data):
    """Get parental education distribution for cluster"""
    if 'parental level of education' in cluster_data.columns:
        return cluster_data['parental level of education'].value_counts().to_dict()
    return {}

def get_lunch_distribution(cluster_data):
    """Get lunch type distribution for cluster"""
    if 'lunch' in cluster_data.columns:
        return cluster_data['lunch'].value_counts().to_dict()
    return {}

def get_prep_distribution(cluster_data):
    """Get test preparation distribution for cluster"""
    if 'test preparation course' in cluster_data.columns:
        return cluster_data['test preparation course'].value_counts().to_dict()
    return {}

def generate_correlation_analysis(df):
    """Generate correlation analysis between scores"""
    score_columns = ['math score', 'reading score', 'writing score']
    correlation_matrix = df[score_columns].corr()
    
    return {
        'correlation_matrix': correlation_matrix.to_dict(),
        'strongest_correlation': {
            'subjects': ['reading score', 'writing score'],
            'value': correlation_matrix.loc['reading score', 'writing score']
        }
    }

def generate_score_distributions(df):
    """Generate score distribution data"""
    score_columns = ['math score', 'reading score', 'writing score']
    distributions = {}
    
    for subject in score_columns:
        distributions[subject] = {
            'mean': float(df[subject].mean()),
            'std': float(df[subject].std()),
            'min': float(df[subject].min()),
            'max': float(df[subject].max()),
            'q25': float(df[subject].quantile(0.25)),
            'q50': float(df[subject].quantile(0.50)),
            'q75': float(df[subject].quantile(0.75)),
            'histogram': df[subject].value_counts().sort_index().to_dict()
        }
    
    return distributions

def generate_demographic_analysis(df):
    """Generate demographic analysis"""
    demographic_data = {}
    
    if 'gender' in df.columns:
        demographic_data['gender'] = df['gender'].value_counts().to_dict()
    
    if 'race/ethnicity' in df.columns:
        demographic_data['race'] = df['race/ethnicity'].value_counts().to_dict()
    
    if 'parental level of education' in df.columns:
        demographic_data['education'] = df['parental level of education'].value_counts().to_dict()
    
    if 'lunch' in df.columns:
        demographic_data['lunch'] = df['lunch'].value_counts().to_dict()
    
    if 'test preparation course' in df.columns:
        demographic_data['prep'] = df['test preparation course'].value_counts().to_dict()
    
    return demographic_data

@api_view(['GET'])
def get_cluster_visualization(request):
    """Generate cluster visualization"""
    try:
        
        df = load_real_data()
        features = ['math score', 'reading score', 'writing score', 
                   'gender_encoded', 'race/ethnicity_encoded', 
                   'parental level of education_encoded', 'lunch_encoded', 
                   'test preparation course_encoded']
        X = df[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        
        fig = px.scatter(
            x=X_pca[:, 0], 
            y=X_pca[:, 1], 
            color=cluster_labels,
            title="Student Clustering Visualization",
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
            color_discrete_sequence=['#000000', '#CC0000', '#E6E6E6', '#D6CDC5'] 
        )
        
        fig.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#EFE6DE',
            font_color='#000000'
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return Response({'plot': graphJSON})
    
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def clustering_dashboard(request):
    """Render the clustering dashboard"""
    return render(request, 'clustering/dashboard.html')