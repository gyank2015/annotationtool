from django.urls import path
from . import views

app_name = 'videoannotation'

urlpatterns = [
    path('', views.datasets, name='datasets'),
    path('results/', views.results, name='results'),
    path('annotate/', views.annotate, name='annotate'),
    path('<str:datasetID>/labels/', views.labels, name='labels'),
    path('<str:datasetID>/<str:labelID>/<str:page>/', views.videodetail, name='detail')
]
