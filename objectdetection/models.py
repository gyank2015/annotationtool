from django.db import models

from django.contrib.auth.models import User
from annotationtools.models import Datasets

annotation_states = (
    (0, 'NotAnnotated'),
    (1, 'BeingAnnotated'),
    (2, 'Annotated'),
)


class Objectdatas(models.Model):
    dataset = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    data_ID = models.CharField(max_length=100)
    annotation_state = models.IntegerField(choices=annotation_states, default=0)
    annotated_by = models.ForeignKey(User, on_delete=models.PROTECT, null=True, default=None)
