from django.urls import path
from . import views

app_name = 'imageannotation'

urlpatterns = [
    path('', views.datasets, name='datasets'),
    path('<str:datasetID>/labels/', views.labels, name='labels'),
    path('<str:datasetID>/<str:labelID>/<str:page>/', views.imagedetail, name='detail'),
    path('annotate/', views.annotate, name='annotate'),
    path('results/', views.results, name='results'),
]
