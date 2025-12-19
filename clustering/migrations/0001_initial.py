

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ClusterResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cluster_id', models.IntegerField()),
                ('cluster_name', models.CharField(max_length=100)),
                ('description', models.TextField()),
                ('student_count', models.IntegerField()),
                ('avg_math_score', models.FloatField()),
                ('avg_reading_score', models.FloatField()),
                ('avg_writing_score', models.FloatField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='StudentCluster',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('student_id', models.IntegerField()),
                ('distance_to_center', models.FloatField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('cluster_result', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='clustering.clusterresult')),
            ],
        ),
    ]
