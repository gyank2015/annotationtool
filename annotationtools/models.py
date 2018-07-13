from django.db import models
from django.contrib.auth.models import User

dataset_type = (
    (0, 'VideoAnnotation'),
    (1, 'ImageAnnotation'),
    (2, 'ObjectDetection'),
)
dataset_type_mapping = {
    'VideoAnnotation': 0,
    'ImageAnnotation': 1,
    'ObjectDetection': 2,
}
response_choices = (
    (0, 'Skip'),
    (1, 'Yes'),
    (2, 'No'),
)
response_mapping = {
    'Skip': 0,
    'Yes': 1,
    'No': 2,
}


class Datasets(models.Model):
    dataset_type = models.IntegerField(choices=dataset_type)
    dataset_ID = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    source = models.CharField(max_length=100)
    path = models.CharField(max_length=200)


class Labels(models.Model):
    dataset = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    label_ID = models.IntegerField()
    name = models.CharField(max_length=100)

    class Meta:
        unique_together = ('dataset', 'label_ID')

    def calculate_labelID(self, ds):
        present_labels = Labels.objects.filter(dataset__dataset_ID=ds.dataset_ID).order_by('-label_ID').values_list('label_ID', flat=True)
        if present_labels:
            return present_labels[0] + 1
        else:
            return 1

    def save(self, *args, **kwargs):
        label_ID = self.calculate_labelID(self.dataset)
        self.label_ID = label_ID
        super(Labels, self).save(*args, **kwargs)


class Data(models.Model):
    label = models.ForeignKey(Labels, on_delete=models.CASCADE)
    data_ID = models.CharField(max_length=100)
    yes_count = models.IntegerField(default=0)
    no_count = models.IntegerField(default=0)
    skip_count = models.IntegerField(default=0)


class User_state(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    label = models.ForeignKey(Labels, on_delete=models.CASCADE)
    cur_data_id = models.CharField(max_length=100, default='')
    skip_count = models.IntegerField(default=0)
    yes_count = models.IntegerField(default=0)
    no_count = models.IntegerField(default=0)

    class Meta:
        unique_together = ('user', 'label')


class Annotation_history(models.Model):
    user = models.ForeignKey(User, on_delete=models.PROTECT)
    data = models.ForeignKey(Data, on_delete=models.CASCADE)
    response = models.IntegerField(choices=response_choices)

    class Meta:
        unique_together = ('user', 'data')
