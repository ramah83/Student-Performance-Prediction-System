# Education AI - Student Performance Prediction System

A comprehensive system for analyzing and predicting student performance using artificial intelligence and machine learning techniques.

## Key Features

- **Grade Prediction**: Predict student scores in math, reading, and writing
- **Data Analysis**: Comprehensive analysis of student performance and influencing factors
- **Smart Clustering**: Group students based on performance and shared characteristics
- **Interactive Dashboard**: Modern user interface with interactive charts and visualizations
- **Detailed Reports**: Generate comprehensive PDF reports on student performance

## Technologies Used

- **Backend**: Django 4.2, Django REST Framework
- **Machine Learning**: Scikit-learn, TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Database**: SQLite (scalable to PostgreSQL)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Requirements

- Python 3.8+
- Django 4.2+
- Libraries listed in `requirements.txt`

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/education-ai.git
cd education-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate 
# or
source venv/bin/activate  
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Setup database
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Load sample data (optional)
```bash
python manage.py load_sample_data
```

### 6. Train models
```bash
python manage.py train_models
```

### 7. Run the server
```bash
python manage.py runserver
```

## Project Structure

```
education_ai/
├── clustering/          # Smart clustering application
├── dashboard/           # Main dashboard
├── prediction/          # Prediction and analysis application
├── models/             # Trained ML models
├── Data/               # Student data
├── static/             # Static files (CSS, JS)
├── templates/          # HTML templates
└── requirements.txt    # Project requirements
```

## Usage

1. **Home Page**: Display general statistics about student performance
2. **Grade Prediction**: Input new student data to get grade predictions
3. **Clustering Analysis**: View different student groups and their characteristics
4. **Reports**: Generate detailed PDF reports

## Contact

For any questions or suggestions, please open an Issue in the project.
**Gmail** : rammohamed962@gmail.com
**linkedin** : [rammohamed962@gmail.com](https://www.linkedin.com/in/mohamed-khaled-16547233a/)
