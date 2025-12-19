<h1 align="center" style="color:#2c3e50;">Education AI</h1>
<h3 align="center" style="color:#34495e;">Student Performance Prediction System</h3>

<p align="center">
A comprehensive system for analyzing and predicting student performance using
<b>Artificial Intelligence</b> and <b>Machine Learning</b>.
</p>

<hr>

<h2 style="color:#2c3e50;">Overview</h2>

<p>
<b>Education AI</b> is designed to help educators and data analysts understand student performance through prediction, analysis, clustering, and visualization.
The system integrates <b>Django</b> with modern <b>Machine Learning models</b> to provide accurate insights and interactive dashboards.
</p>

<hr>

<h2 style="color:#2c3e50;">Key Features</h2>

<ul>
  <li><b>Grade Prediction</b><br>
      Predict student scores in math, reading, and writing.
  </li>
  <br>
  <li><b>Data Analysis</b><br>
      Comprehensive analysis of student performance and influencing factors.
  </li>
  <br>
  <li><b>Smart Clustering</b><br>
      Group students based on performance and shared characteristics.
  </li>
  <br>
  <li><b>Interactive Dashboard</b><br>
      Modern user interface with interactive charts and visualizations.
  </li>
  <br>
  <li><b>Detailed Reports</b><br>
      Generate comprehensive <b>PDF reports</b> on student performance.
  </li>
</ul>

<hr>

<h2 style="color:#2c3e50;">Technologies Used</h2>

<h3 style="color:#34495e;">Backend</h3>
<ul>
  <li>Django 4.2</li>
  <li>Django REST Framework</li>
</ul>

<h3 style="color:#34495e;">Machine Learning</h3>
<ul>
  <li>Scikit-learn</li>
  <li>TensorFlow / Keras</li>
</ul>

<h3 style="color:#34495e;">Frontend</h3>
<ul>
  <li>HTML5</li>
  <li>CSS3</li>
  <li>JavaScript</li>
  <li>Chart.js</li>
</ul>

<h3 style="color:#34495e;">Data Processing & Visualization</h3>
<ul>
  <li>Pandas</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
</ul>

<h3 style="color:#34495e;">Database</h3>
<ul>
  <li>SQLite (Scalable to PostgreSQL)</li>
</ul>

<hr>

<h2 style="color:#2c3e50;">Requirements</h2>

<ul>
  <li>Python <b>3.8+</b></li>
  <li>Django <b>4.2+</b></li>
  <li>Libraries listed in <code>requirements.txt</code></li>
</ul>

<hr>

<h2 style="color:#2c3e50;">Installation and Setup</h2>

<h3>1. Clone the Repository</h3>
<pre><code>git clone https://github.com/yourusername/education-ai.git
cd education-ai</code></pre>

<h3>2. Create Virtual Environment</h3>
<pre><code>python -m venv venv
venv\Scripts\activate
# or
source venv/bin/activate</code></pre>

<h3>3. Install Requirements</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>4. Setup Database</h3>
<pre><code>python manage.py makemigrations
python manage.py migrate</code></pre>

<h3>5. Load Sample Data (Optional)</h3>
<pre><code>python manage.py load_sample_data</code></pre>

<h3>6. Train Machine Learning Models</h3>
<pre><code>python manage.py train_models</code></pre>

<h3>7. Run Development Server</h3>
<pre><code>python manage.py runserver</code></pre>

<hr>

<h2 style="color:#2c3e50;">Project Structure</h2>

<pre><code>education_ai/
├── clustering/          # Smart clustering application
├── dashboard/           # Main dashboard
├── prediction/          # Prediction and analysis application
├── models/              # Trained ML models
├── Data/                # Student data
├── static/              # Static files (CSS, JS)
├── templates/           # HTML templates
└── requirements.txt     # Project requirements</code></pre>

<hr>

<h2 style="color:#2c3e50;">Usage</h2>

<ul>
  <li><b>Home Page</b><br>
      Displays general statistics about student performance.
  </li>
  <br>
  <li><b>Grade Prediction</b><br>
      Input new student data to get predicted scores.
  </li>
  <br>
  <li><b>Clustering Analysis</b><br>
      View different student groups and their characteristics.
  </li>
  <br>
  <li><b>Reports</b><br>
      Generate detailed PDF performance reports.
  </li>
</ul>

<hr>

<h2 style="color:#2c3e50;">Contact</h2>

<p>
For any questions or suggestions, please open an <b>Issue</b> in the project.
</p>

<p>
<b>Email:</b> rammohamed962@gmail.com<br>
<b>LinkedIn:</b> https://www.linkedin.com/in/mohamed-khaled-16547233a/
</p>
