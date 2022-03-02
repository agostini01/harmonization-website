# Generated by Django 3.0.7 on 2022-02-05 01:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0006_auto_20220110_1924'),
    ]

    operations = [
        migrations.CreateModel(
            name='recommendations',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('var1', models.CharField(blank=True, default='-9.0', max_length=50, null=True)),
                ('var2', models.CharField(blank=True, default='-9.0', max_length=50, null=True)),
                ('desc1', models.CharField(blank=True, default='-9.0', max_length=5000, null=True)),
                ('desc2', models.CharField(blank=True, default='-9.0', max_length=5000, null=True)),
                ('dist1', models.FloatField(blank=True, default=-9.0, null=True)),
                ('dist2', models.FloatField(blank=True, default=-9.0, null=True)),
                ('dist3', models.FloatField(blank=True, default=-9.0, null=True)),
                ('dist4', models.FloatField(blank=True, default=-9.0, null=True)),
            ],
        ),
    ]
