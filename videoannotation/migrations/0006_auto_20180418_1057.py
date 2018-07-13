# Generated by Django 2.0.2 on 2018-04-18 10:57

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('videoannotation', '0005_auto_20180418_1034'),
    ]

    operations = [
        migrations.CreateModel(
            name='Annotation_details',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('validation', models.BooleanField()),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='annotations', to=settings.AUTH_USER_MODEL)),
                ('video_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='annotations', to='videoannotation.Annotated_videos')),
            ],
        ),
        migrations.DeleteModel(
            name='Annotators',
        ),
    ]
