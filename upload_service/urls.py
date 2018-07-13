from django.urls import path
from . import views

app_name = 'upload_service'

urlpatterns = [
    path('', views.details, name='details'),
    path('upload/', views.upload, name='upload'),
    path('uploaded/datasets', views.uploaded_datasets, name='uploaded_datasets'),
    path('uploaded/<str:dataset_ID>/addlabels', views.addlabels, name='addlabels'), 
    path('uploaded/<str:dataset_ID>/viewlabels', views.viewlabels, name='viewlabels'),
    path('uploaded/<str:dataset_ID>/addimages', views.addimages, name='addimages'),
    path('uploaded/addimages', views.addimagesfile, name='addimagesfile'),


]
