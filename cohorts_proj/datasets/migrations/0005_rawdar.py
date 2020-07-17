# Generated by Django 3.0.6 on 2020-06-15 21:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0004_rawunm'),
    ]

    operations = [
        migrations.CreateModel(
            name='RawDAR',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('unq_id', models.CharField(max_length=100)),
                ('assay', models.CharField(max_length=100)),
                ('lab', models.CharField(max_length=100)),
                ('participant_type', models.CharField(choices=[('maternal', 'mother'), ('child', 'child')], max_length=15)),
                ('time_period', models.CharField(choices=[('12G', 'week 12 gestational'), ('24G', 'week 24 gestational'), ('6WP', 'week 6 portpartun'), ('6MP', 'month 6 postpartum'), ('1YP', 'year 1 postpartum'), ('2YP', 'year 2 postpartum'), ('3YP', 'year 3 postpartum'), ('5YP', 'year 5 postpartum')], max_length=3)),
                ('batch', models.IntegerField()),
                ('squid', models.CharField(max_length=100)),
                ('sample_gestage_days', models.IntegerField()),
                ('Ag', models.FloatField()),
                ('Ag_IDL', models.FloatField()),
                ('Ag_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Al', models.FloatField()),
                ('Al_IDL', models.FloatField()),
                ('Al_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('As', models.FloatField()),
                ('As_IDL', models.FloatField()),
                ('As_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Be', models.FloatField()),
                ('Be_IDL', models.FloatField()),
                ('Be_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Cd', models.FloatField()),
                ('Cd_IDL', models.FloatField()),
                ('Cd_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Co', models.FloatField()),
                ('Co_IDL', models.FloatField()),
                ('Co_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Cr', models.FloatField()),
                ('Cr_IDL', models.FloatField()),
                ('Cr_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Cu', models.FloatField()),
                ('Cu_IDL', models.FloatField()),
                ('Cu_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Fe', models.FloatField()),
                ('Fe_IDL', models.FloatField()),
                ('Fe_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Hg', models.FloatField()),
                ('Hg_IDL', models.FloatField()),
                ('Hg_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Mn', models.FloatField()),
                ('Mn_IDL', models.FloatField()),
                ('Mn_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Mo', models.FloatField()),
                ('Mo_IDL', models.FloatField()),
                ('Mo_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Ni', models.FloatField()),
                ('Ni_IDL', models.FloatField()),
                ('Ni_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Pb', models.FloatField()),
                ('Pb_IDL', models.FloatField()),
                ('Pb_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Sb', models.FloatField()),
                ('Sb_IDL', models.FloatField()),
                ('Sb_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Se', models.FloatField()),
                ('Se_IDL', models.FloatField()),
                ('Se_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Sn', models.FloatField()),
                ('Sn_IDL', models.FloatField()),
                ('Sn_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Tl', models.FloatField()),
                ('Tl_IDL', models.FloatField()),
                ('Tl_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('U', models.FloatField()),
                ('U_IDL', models.FloatField()),
                ('U_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('W', models.FloatField()),
                ('W_IDL', models.FloatField()),
                ('W_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('Zn', models.FloatField()),
                ('Zn_IDL', models.FloatField()),
                ('Zn_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
                ('V', models.FloatField()),
                ('V_IDL', models.FloatField()),
                ('V_BDL', models.CharField(choices=[('1', 'below detection level'), ('0', 'above detection level'), ('nan', 'invalid')], max_length=3)),
            ],
        ),
    ]