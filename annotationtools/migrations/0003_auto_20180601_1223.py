# Generated by Django 2.0.2 on 2018-06-01 12:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('annotationtools', '0002_auto_20180528_1035'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datasets',
            name='dataset_type',
            field=models.IntegerField(choices=[(0, 'VideoAnnotation'), (1, 'ImageAnnotation'), (2, 'ObjectDetection')]),
        ),
    ]