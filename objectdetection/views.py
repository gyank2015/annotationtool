from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse
import json
from django.db import transaction
from .models import Objectdatas
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from annotationtools.models import Datasets, Labels, dataset_type_mapping ,User_state
from django.conf import settings
import os
DEFAULT_LABEL=""
# ITEMS_PER_PAGE = settings.ITEMS_PER_PAGE
per_page_data = settings.PER_PAGE_DATA
from src.json2xml import Json2xml
import xml.etree.ElementTree as ET
from django.http import JsonResponse
from django.core import serializers
import xml.etree.cElementTree as ET

# Create your views here.


@login_required()
def datasets(request):
    request = setupSession(request)
    request, datasets = setupDatasetType(request, dataset_type=dataset_type_mapping['ObjectDetection'])
    return render(request, 'objectdetection/datasets.html', {'datasets': datasets})


def setupDatasetType(request, dataset_type):
    datasets = Datasets.objects.filter(dataset_type=dataset_type)
    request.session['dataset_type'] = dataset_type
    return request, datasets


def setupSession(request):
    request.session['dataset_type'] = 2
    request.session['dataset_ID'] = 0
    request.session['last_viewed_data_id'] = ''

    return request

@login_required
def test(request, datasetID, page='current'):

    show_next = True
    show_prev = True
    show_restart = True
    name = Datasets.objects.get(dataset_ID=datasetID).name
    user = User.objects.filter(username=request.user.username).get()
    label = Labels.objects.filter(dataset__dataset_ID=datasetID)[0]
    user_state = User_state.objects.filter(user=user, label=label)
    if not user_state:
        User_state.objects.create(user=user, label=label)
        user_state = User_state.objects.filter(user=user, label=label)
    user_state = user_state[0]

    last_viewed_data_id = user_state.cur_data_id
    _last_viewed_dataID = last_viewed_data_id

    if page == 'restart':
        data_list = Objectdatas.objects.filter(dataset__dataset_ID=datasetID).order_by('data_ID')[:per_page_data]
        show_prev = False
        _last_viewed_dataID = data_list[0].data_ID
    elif page == 'next':
        data_list = Objectdatas.objects.filter(dataset__dataset_ID=datasetID, data_ID__gte=last_viewed_data_id).order_by('data_ID')[:2 * per_page_data]
        if len(data_list) < per_page_data + 1:
            data_list = []
            _last_viewed_dataID = last_viewed_data_id
            show_next = False
        else:
            _last_viewed_dataID = data_list[per_page_data].data_ID
            data_list = data_list[per_page_data:]
    elif page == 'prev':
        data_list = Objectdatas.objects.filter(dataset__dataset_ID=datasetID, data_ID__lt=last_viewed_data_id).order_by('-data_ID')[:per_page_data][::-1]
        if(data_list):
            _last_viewed_dataID = data_list[0].data_ID
        else:
            _last_viewed_dataID = ''
            show_prev = False
    else:
        last_viewed_data_id = '0'
        data_list = Objectdatas.objects.filter(dataset__dataset_ID=datasetID, data_ID__gte=last_viewed_data_id).order_by('data_ID')[:per_page_data]
        # data_list = Objectdatas.objects.filter(dataset__dataset_ID=datasetID)
        # print('data_list', data_list)
    user_state.cur_data_id = _last_viewed_dataID
    user_state.save()

    data_ID_list = [{'data_ID': data.data_ID, 'annotation_state': data.annotation_state} for data in data_list]
    # data_ID_list = ['test1.jpg', 'test2.jpg', 'test3.jpg']
    labels = Labels.objects.filter(dataset__dataset_ID=datasetID)
    label_names = [label.name for label in labels]
    # label_names = ['label1', 'label12', 'label3', 'label4']
    data_ID = ''  # data_ID_list[0]
    cur_data_ID = data_ID
    return render(request, 'objectdetection/test.html', {'dataset_name':name,'cur_data_ID': cur_data_ID, 'data_ID_list': data_ID_list, 'show_next':show_next,'show_prev':show_prev,'show_restart':show_restart,'label_names': label_names, 'dataset_ID': datasetID, 'data_ID': data_ID})


def annotate(request):
    if request.method == "POST":
        annotationData = json.loads(request.POST["annotationData"])
        return redirect(reverse('objectdetection:test'))
    elif request.method == 'GET':
        print('path', request.GET["path"])

@transaction.atomic
def getxml(request, datasetID, dataID):
    if request.method == 'GET':
        xmlfile = dataID.split('.')[0]
        filepath = settings.MEDIA_ROOT + '/' + str(datasetID) + '/xml/' + xmlfile + '.xml'
        annotation_state = 0
        # Make below transaction atomic
        data = Objectdatas.objects.filter(dataset__dataset_ID=datasetID, data_ID=dataID).get()
        annotation_state = data.annotation_state
        if annotation_state == 0:
            data.annotation_state = 1
            data.save()
        return JsonResponse({'annotation_state': annotation_state})
        # Also return the xml data when annotation_state == 2


@transaction.atomic
def putxml_savexml(jsondata, filepath, datasetID, dataID):
    data = Objectdatas.objects.filter(dataset__dataset_ID=datasetID, data_ID=dataID).get()
    jsondata = json.loads(jsondata)
    print (jsondata)
    annotation_state = data.annotation_state
    if(annotation_state == 2):
        return annotation_state
    else:
        root = ET.Element("image")
        for annotation_dict in jsondata:
            box=ET.SubElement(root, "box", x1= str(annotation_dict['x1']),y1=str(annotation_dict['y1']),x2=str(annotation_dict['x2']),y2=str(annotation_dict['y2']))
            ET.SubElement(box,"label").text = annotation_dict['label']
        tree = ET.ElementTree(root)
        tree.write(filepath)
        data.annotation_state=2
        data.save()
    return annotation_state


def putxml(request, datasetID, dataID):
    if request.method == 'POST':
        annotateData = request.POST["annotateData"]
        print((annotateData))
        filepath = settings.MEDIA_ROOT + '/' + str(datasetID) + '/xml/' + dataID.split('.')[0] + '.xml'
        # Make below transaction atomic
        data = Objectdatas.objects.filter(dataset__dataset_ID=datasetID, data_ID=dataID).get()
        annotation_state = data.annotation_state
        if annotation_state != 2:
            annotation_state = putxml_savexml(annotateData, filepath, datasetID, dataID)
        return JsonResponse({'annotation_state': annotation_state})


def get_next_id(request,datasetID, page='current'):
    print("in get_next_id")
    cur_data_ID = request.GET.get('cur_data_ID', None)
    cur_data = Objectdatas.objects.get(dataset__dataset_ID=datasetID,data_ID=cur_data_ID)
    print(cur_data)
    datas=Objectdatas.objects.filter(dataset__dataset_ID=datasetID,data_ID__gt=cur_data_ID)
    print(datas)
    data=Objectdatas.objects.filter(dataset__dataset_ID=datasetID,data_ID__gt=cur_data_ID)[0]
    json_data = {
        'next_id': data.data_ID
    }
    # print(json_data)
    return JsonResponse(json_data)