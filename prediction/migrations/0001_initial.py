

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='StudentData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('gender', models.CharField(choices=[('male', 'Male'), ('female', 'Female')], max_length=10)),
                ('race_ethnicity', models.CharField(choices=[('group A', 'Group A'), ('group B', 'Group B'), ('group C', 'Group C'), ('group D', 'Group D'), ('group E', 'Group E')], max_length=20)),
                ('parental_education', models.CharField(choices=[('some high school', 'Some High School'), ('high school', 'High School'), ('some college', 'Some College'), ("associate's degree", "Associate's Degree"), ("bachelor's degree", "Bachelor's Degree"), ("master's degree", "Master's Degree")], max_length=30)),
                ('lunch', models.CharField(choices=[('standard', 'Standard'), ('free/reduced', 'Free/Reduced')], max_length=15)),
                ('test_preparation', models.CharField(choices=[('none', 'None'), ('completed', 'Completed')], max_length=15)),
                ('math_score', models.IntegerField()),
                ('reading_score', models.IntegerField()),
                ('writing_score', models.IntegerField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='PredictionResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted_math_score', models.FloatField(blank=True, null=True)),
                ('predicted_reading_score', models.FloatField(blank=True, null=True)),
                ('predicted_writing_score', models.FloatField(blank=True, null=True)),
                ('model_used', models.CharField(max_length=50)),
                ('accuracy_score', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('student_data', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='prediction.studentdata')),
            ],
        ),
    ]
