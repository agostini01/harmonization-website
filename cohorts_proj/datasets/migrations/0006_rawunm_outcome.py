# Generated by Django 3.0.7 on 2020-08-21 13:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0005_rawdar'),
    ]

    operations = [
        migrations.AddField(
            model_name='rawunm',
            name='Outcome',
            field=models.CharField(blank=True, choices=[('0', 'term'), ('1', 'preterm')], max_length=1),
        ),
    ]
