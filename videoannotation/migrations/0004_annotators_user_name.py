# Generated by Django 2.0.2 on 2018-04-18 10:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videoannotation', '0003_auto_20180418_1023'),
    ]

    operations = [
        migrations.AddField(
            model_name='annotators',
            name='user_name',
            field=models.CharField(default='none', max_length=100),
        ),
    ]