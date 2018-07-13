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
from zipfile import ZipFile
import shutil
import mimetypes
from django.http import StreamingHttpResponse
from wsgiref.util import FileWrapper


# Create your views here.
@login_required(login_url='/login/')
def datasets(request):
	datasets = Datasets.objects.all()
	return render(request,'download/datasets.html',{'datasets':datasets})

@login_required(login_url='/login/')
def labels(request,dataset_ID='none'):
	if request.method =='GET':
		labels = Labels.objects.filter(dataset__dataset_ID=dataset_ID)
		dataset_type = Datasets.objects.get(dataset_ID=dataset_ID).dataset_type
		print(dataset_type)
		return render(request,'download/labels.html',{'dataset_ID':dataset_ID,'labels':labels,'dataset_type':dataset_type})
	else:
		print(request.POST)
		dataset_ID = int(request.POST['dataset_ID'])
		dataset_name = Datasets.objects.get(dataset_ID=(dataset_ID)).name
		dataset_type = Datasets.objects.get(dataset_ID=(dataset_ID)).dataset_type
		Data_list= []
		labels= []
		if dataset_type!=2:
			labels_ID = request.POST.getlist('labels')
			for label in labels_ID:
				labels.append(Labels.objects.get(dataset__dataset_ID=dataset_ID, label_ID=int(label)).name)
				Data_list.append(Data.objects.filter(label__dataset__dataset_ID=dataset_ID, label__label_ID=int(label)))
			images = []
			zip_file = os.path.abspath(os.path.join(settings.MEDIA_ROOT,str(dataset_ID)+'/'+dataset_name+'_result'))
			if not os.path.exists(zip_file):
				os.makedirs(zip_file)
			count=0
			for data in Data_list:
				for image in data:
					if image.yes_count>image.no_count:
						images.append(image.data_ID)
				uri = os.path.abspath(os.path.join(settings.MEDIA_ROOT,str(dataset_ID)))
				dest=uri+'/'+dataset_name+'_result'+'/'+labels[count]
				uri+='/'+dataset_name
				if not os.path.exists(dest):
					os.makedirs(dest)
				for image in images:
					file = os.path.join(uri, image)
					if (os.path.isfile(file)):
						shutil.copy(file,dest)
				images = []
				count+=1
			shutil.make_archive(zip_file, 'zip', zip_file)
			file_path = zip_file+'.zip'
			filename = os.path.basename(file_path)
			chunk_size = 8192
			response = StreamingHttpResponse(FileWrapper(open(file_path, 'rb'), chunk_size),content_type=mimetypes.guess_type(file_path)[0])
			response['Content-Length'] = os.path.getsize(file_path)    
			response['Content-Disposition'] = "attachment; filename=%s" % filename
			shutil.rmtree(zip_file)
			os.remove(file_path)
			return response
		else:
			Data_list.append(Objectdatas.objects.filter(dataset__dataset_ID=dataset_ID,annotation_state=2))
			zip_file = os.path.abspath(os.path.join(settings.MEDIA_ROOT,str(dataset_ID)+'/'+dataset_name+'_result'))
			uri = os.path.abspath(os.path.join(settings.MEDIA_ROOT,str(dataset_ID)))
			dest = uri+'/'+dataset_name+'_result'+'/images'
			xmluri = uri +'/xml'
			xmldest = uri +'/'+dataset_name+'_result'+'/xml'
			uri +='/'+dataset_name
			if not os.path.exists(dest):
				os.makedirs(dest)
			if not os.path.exists(xmldest):
				os.makedirs(xmldest)
			for data in Data_list[0]:
				file = os.path.join(uri, data.data_ID)
				if (os.path.isfile(file)):
					shutil.copy(file,dest)
				xmlfile = os.path.join(xmluri,data.data_ID.split(".")[0]+'.xml')
				if (os.path.isfile(xmlfile)):
					shutil.copy(xmlfile,xmldest)
			shutil.make_archive(zip_file, 'zip', zip_file)
			file_path = zip_file+'.zip'
			filename = os.path.basename(file_path)
			chunk_size = 8192
			response = StreamingHttpResponse(FileWrapper(open(file_path, 'rb'), chunk_size),content_type=mimetypes.guess_type(file_path)[0])
			response['Content-Length'] = os.path.getsize(file_path)    
			response['Content-Disposition'] = "attachment; filename=%s" % filename
			shutil.rmtree(zip_file)
			os.remove(file_path)
			return response
			# return HttpResponse('<h1>labels selected!</h1>')
