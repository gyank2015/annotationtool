from django.db import models
from django.contrib.auth.models import User
# Create your models here.


class Video(models.Model):

    video_name = models.CharField(max_length=100)
    video_id = models.CharField(max_length=100, primary_key=True)
    source = models.CharField(max_length=100)
    duration = models.IntegerField()

    def __str__(self):
        return self.video_id


class Annotated_videos(models.Model):

    video_id = models.CharField(max_length=100, primary_key=True)
    label = models.CharField(max_length=100)
    annotated = models.BooleanField(default=False)
    validation = models.BooleanField(default=False)
    source = models.CharField(max_length=100)

    def __str__(self):
        return self.video_id


class Annotation_details(models.Model):

    user = models.ForeignKey(User, related_name='annotations', on_delete=models.CASCADE)
    validation = models.BooleanField()
    video = models.ForeignKey(Annotated_videos, on_delete=models.CASCADE, related_name='annotations')
