from django.urls import path
from . import views


app_name = 'objectdetection'

urlpatterns = [
    path('', views.datasets, name='datasets'),
    # path('results/', views.results, name='results'),
    path('detail/<int:datasetID>/<str:page>/', views.test, name='test'),
    path('annotate/', views.annotate, name='annotate'),
    path('getxml/<int:datasetID>/<str:dataID>', views.getxml, name='getxml'),
    path('putxml/<int:datasetID>/<str:dataID>', views.putxml, name='putxml'),
    path('detail/<int:datasetID>/<str:page>/get_next_id/', views.get_next_id, name='get_next_id'),
    # path('get_next_id/', views.get_next_id, name='get_next_id'),
    # path('<str:datasetID>/labels/', views.labels, name='labels'),
    # path('<str:datasetID>/<str:labelID>/<str:page>/', views.videodetail, name='detail')
]
