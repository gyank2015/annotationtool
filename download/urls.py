from django.urls import path
from . import views

app_name = 'download'

urlpatterns = [
    path('', views.datasets, name='datasets'),
    path('<str:dataset_ID>', views.labels, name='labels'),

]
