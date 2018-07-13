from django.shortcuts import render
from django.http import HttpResponse
from annotationtools.models import Datasets, Labels, Data, User_state, dataset_type_mapping,dataset_type
from django.urls import reverse
from objectdetection.models import Objectdatas
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.db import transaction
from django.conf import settings
import logging
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
PATH = settings.MEDIA_ROOT
import os
from django.views.decorators.csrf import csrf_exempt
import zipfile
import shutil

# Create your views here.
@login_required(login_url='/login/')
def details(request):
	if request.method=='GET':
		return render(request,'upload_service/upload_details.html',{})
	else:
		DatasetName=request.POST['DatasetName']
		DatasetType= request.POST['Dataset Type']
		DatasetType=(dataset_type_mapping[DatasetType])
		b=Datasets.objects.create(dataset_type=DatasetType,name=DatasetName,source=request.user.username,path="/")
		b.path=PATH+'/'+str(b.dataset_ID)
		b.save()
		if not os.path.exists(b.path):
			os.makedirs(b.path)
		xml_path = b.path+'/xml'
		if DatasetType==2:
			if not os.path.exists(xml_path):
				os.makedirs(xml_path)
		labels = request.POST['labels'].split(",")
		labelcluster = request.POST['labels']
		for label in labels:
			Labels.objects.create(dataset=b,name=label)
		return render(request,'upload_service/upload_files.html',{'dataset_ID':b.dataset_ID,'DatasetType':b.dataset_type})



# @login_required(login_url='/login/')
@csrf_exempt
def upload (request):
	if request.method=='POST':
		dataset_ID=(request.POST['dataset_ID'])
		dataset_type = (request.POST['dataset_type'])
		dataset_type = int(dataset_type[:-1])
		dataset_id=dataset_ID[:-1]
		dataset_name = Datasets.objects.get(dataset_ID=int(dataset_id)).name
		images=(request.FILES.getlist('images'))
		for image in images:
			uri = dataset_ID+str(image)
			path = default_storage.save(uri, ContentFile(image.read()))
		zippy = images[0]
		folder_name = str(zippy).split(".")[0]
		zipuri ='/'+str(images[0])
		with zipfile.ZipFile(zippy) as zf:
			images = zf.namelist()
		labels=Labels.objects.filter(dataset__dataset_ID=int(dataset_id))
		if dataset_type!=2:
			for image in images:
				for label in labels:
					Data.objects.create(label=label,data_ID=str(image.split('/')[1]))
		else:
			dataset = Datasets.objects.get(dataset_ID=int(dataset_id))
			for image in images:
				Objectdatas.objects.create(dataset=dataset,data_ID=str(image.split('/')[1]))
		with zipfile.ZipFile(zippy) as zf:
			uri = os.path.abspath(os.path.join(settings.MEDIA_ROOT,dataset_ID))
			zf.extractall(uri)
		source = uri+'/'+folder_name
		dest = uri +'/'+dataset_name
		uri+=zipuri
		os.rename(source,dest)
		os.remove(uri)
		return HttpResponse('<h1> in upload </h1>')


@login_required(login_url='/login/')
def uploaded_datasets(request):
	datasets = Datasets.objects.filter(source=request.user.username)
	return render(request,'upload_service/uploaded_datasets.html',{'datasets':datasets})


@login_required(login_url='/login/')
def viewlabels(request,dataset_ID='none'):
	dataset_ID = dataset_ID
	labels = Labels.objects.filter(dataset__dataset_ID=dataset_ID)
	return render(request , 'upload_service/viewlabels.html',{'labels':labels})

@login_required(login_url='/login/')
def addlabels(request,dataset_ID='none'):
	if request.method=='GET':
		dataset_ID = dataset_ID
		dataset_name = Datasets.objects.get(dataset_ID=dataset_ID).name
		dataset_type = Datasets.objects.get(dataset_ID=dataset_ID).dataset_type
		return render(request,'upload_service/addlabels.html',{'dataset_ID':dataset_ID , 'dataset_name' : dataset_name, 'dataset_type':dataset_type})


	else :
		labels = request.POST['labels'].split(",")
		dataset_ID=(request.POST['dataset_ID'])
		dataset_id=int(dataset_ID[:-1])
		dataset = Datasets.objects.get(dataset_ID=dataset_id)
		dataset_type = (request.POST['dataset_type'])
		dataset_name = request.POST['dataset_name']
		dataset_type = int(dataset_type[:-1])
		existing_label = Labels.objects.filter(dataset__dataset_ID = dataset_id)[0]
		new_labels=[]
		for label in labels:
			new_labels.append(Labels.objects.create(dataset=dataset,name=label))
		labels = new_labels
		if dataset_type!=2:
			Datas = Data.objects.filter(label=existing_label)
			images = []
			for data in Datas:
				images.append(data.data_ID)
			# print(images,labels,existing_label)
			for image in images:
				for label in labels:
					Data.objects.create(label=label,data_ID=image)
		else:
			Labels.objects.create(dataset=dataset)
		return redirect(reverse('upload_service:uploaded_datasets'))


@login_required(login_url='/login/')
def addimages(request,dataset_ID='none'):
	if request.method=='GET':
		return render(request,'upload_service/addimages.html',{'dataset_ID':dataset_ID})

@login_required(login_url='/login/')
def addimagesfile(request):
	if request.method=='POST':
		dataset_ID=(request.POST['dataset_ID'])
		dataset_id=dataset_ID[:-1]
		dataset_name = Datasets.objects.get(dataset_ID=int(dataset_id)).name
		dataset_type = Datasets.objects.get(dataset_ID=int(dataset_id)).dataset_type
		images=(request.FILES.getlist('images'))
		for image in images:
			uri = dataset_ID+str(image)
			path = default_storage.save(uri, ContentFile(image.read()))
		zippy = images[0]
		folder_name = str(zippy).split(".")[0]
		zipuri ='/'+str(images[0])
		with zipfile.ZipFile(zippy) as zf:
			images = zf.namelist()
		print('images',images)
		new_images=[]
		existing_images=[]
		uri = os.path.abspath(os.path.join(settings.MEDIA_ROOT,dataset_ID+'/'+dataset_name))
		for image in images:
			if os.path.exists(uri+'/'+image.split('/')[1]):
				existing_images.append(image)
			else:
				new_images.append(image)
		labels=Labels.objects.filter(dataset__dataset_ID=int(dataset_id))
		print(new_images,"************")
		print(dataset_type)
		if dataset_type!=2:
			for image in new_images:
				for label in labels:
					Data.objects.create(label=label,data_ID=str(image.split('/')[1]))
		else:
			print(new_images,"************")
			dataset = Datasets.objects.get(dataset_ID=int(dataset_id))
			for image in new_images:
				print(Objectdatas.objects.create(dataset=dataset,data_ID=str(image.split('/')[1])))
		with zipfile.ZipFile(zippy) as zf:
			uri = os.path.abspath(os.path.join(settings.MEDIA_ROOT,dataset_ID))
			zf.extractall(uri)
		source = uri+'/'+folder_name
		dest = uri +'/'+dataset_name
		for image in new_images:
			shutil.move(source+'/'+image.split('/')[1],dest)
		uri+=zipuri
		os.remove(uri)
		shutil.rmtree(source)
		return HttpResponse('<h1> in upload </h1>')


