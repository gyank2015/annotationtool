from django.shortcuts import render
from django.http import HttpResponse
from annotationtools.models import Datasets, Labels, Data, User_state, dataset_type_mapping
from django.urls import reverse

from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.db import transaction
from django.conf import settings
import logging
logger = logging.getLogger(__name__)

ITEMS_PER_PAGE = settings.ITEMS_PER_PAGE


# print(User.objects.all())
@login_required(login_url='/login/')
def datasets(request):
    request = setupSession(request)
    request, datasets = setupDatasetType(request, dataset_type=dataset_type_mapping['ImageAnnotation'])
    # datasets = [{ 'dataset_ID' : 'sasdsd',  'dataset_name' : 'sasdsd'}]
    return render(request, 'imageannotation/datasets.html', {'datasets': datasets})


def setupSession(request):
    request.session['dataset_type'] = 1
    request.session['label_ID'] = 0
    request.session['dataset_ID'] = 0
    request.session['last_viewed_data_id'] = ''

    return request


def setupDatasetType(request, dataset_type):
    datasets = Datasets.objects.filter(dataset_type=dataset_type)
    request.session['dataset_type'] = dataset_type
    return request, datasets


@login_required
def labels(request, datasetID=0):
    request, chosen_dataset_ID = setupDataset_labels(request, datasetID)
    if chosen_dataset_ID == 0:
        return redirect(reverse('imageannotation:datasets'))  # Show some error message to user

    labels = []
    dataset_labels = Labels.objects.filter(dataset__dataset_ID=request.session['dataset_ID'])
    labels = [{'label_ID': row.label_ID, 'label_name': row.name} for row in dataset_labels]
    return render(request, 'imageannotation/labels.html', {'dataset_name': request.session['dataset_name'], 'datasetID': request.session['dataset_ID'], 'labels': labels})


def setupDataset_labels(request, datasetID):
    '''
    If datasetID exists	: sets session variabels
            otherwise		: returns datasetID = 0
    '''
    request.session['dataset_type'] = dataset_type_mapping['ImageAnnotation']
    chosen_dataset = Datasets.objects.filter(dataset_type=request.session['dataset_type'], dataset_ID=datasetID)
    if chosen_dataset:
        request.session['dataset_ID'] = datasetID
        request.session['dataset_name'] = chosen_dataset[0].name
        request.session['dataset_path'] = chosen_dataset[0].path
        request.session['dataset_source'] = chosen_dataset[0].source
    else:
        return request, 0
    return request, chosen_dataset[0]


def getUserMetrics(request, dataset_ID, label_ID):
    user = User.objects.filter(username=request.user.username)[:1]
    user_state = User_state.objects.filter(user=user, label__dataset__dataset_ID=dataset_ID, label__label_ID=label_ID)
    totalAnnotated = user_state[0].skip_count + user_state[0].yes_count + user_state[0].no_count
    userMetrics = {'skip_count': user_state[0].skip_count, 'yes_count': user_state[
        0].yes_count, 'no_count': user_state[0].no_count, 'totalAnnotated': totalAnnotated}
    return userMetrics


@login_required
def imagedetail(request, datasetID='none', labelID='none', page='current'):
    ''' TODO : change default values to in type and take care of conditions

    '''
    name = Datasets.objects.get(dataset_ID=datasetID).name
    label_ID = labelID
    dataset_ID = datasetID

    request, chosen_dataset_ID, chosen_label_ID, _ditch = setupSessionData(request, datasetID, labelID)
    # print(chosen_dataset)
    if chosen_dataset_ID == 0:
        return redirect(reverse('imageannotation:datasets'))
    if chosen_label_ID == 0:
        return redirect(reverse('imageannotation:labels', kwargs={'datasetID': request.session['dataset_ID']}))

    userMetrics = getUserMetrics(request, request.session['dataset_ID'], request.session['label_ID'])

    # request, chosen_label = setupLabel(request, labelID)
    # # print(chosen_label)
    # if chosen_label == 'none':
    # 	return redirect(reverse('imageannotation/'+request.session['dataset_ID']+'/labels'))

    _last_viewed_image_id = request.session['last_viewed_data_id']

    data_list = Data.objects.filter(label__label_ID=label_ID, label__dataset__dataset_ID=dataset_ID,
                                    data_ID__gt=_last_viewed_image_id).order_by('data_ID')[:ITEMS_PER_PAGE]
    image_list = data_list

    show_next = 1
    if len(image_list) < 1:
        show_next = 0
    print('dadfdf', dataset_ID, name)
    response = render(request, 'imageannotation/imagedetail.html', {'dataset_name':name,'images': image_list, 'show_next': show_next, 'dataset_ID': request.session[
                      'dataset_ID'], 'label_ID': request.session['label_ID'], 'label_name': request.session['label_name'], 'userMetrics': userMetrics})

    if len(image_list) > 0:
        request.session['reset_cur_data_id'] = image_list[len(image_list) - 1].data_ID

    return response


def setupSessionData(request, datasetID, labelID):
    '''
    Helper function to set session data if datasetID and labelID is valid otherwise return 0 for corresponding invalid value.
    Redirection is taken care of by videodetail()
    '''
    chosen_dataset_ID, chosen_label_ID, last_viewed_data_id = 0, 0, ''
    if datasetID == request.session.get('dataset_ID', 0) and labelID == request.session.get('label_ID', 0):
        last_viewed_data_id = request.session['last_viewed_data_id']
        return request, request.session['dataset_ID'], request.session['label_ID'], last_viewed_data_id
    elif datasetID != request.session.get('dataset_ID', 0):
        request, chosen_dataset_ID = setupDataset_detail(request, datasetID)
    else:
        chosen_dataset_ID = request.session.get('dataset_ID', 0)

    request, chosen_label_ID = setupLabel_detail(request, labelID)
    if chosen_dataset_ID != 0 and chosen_label_ID != 0:
        request, last_viewed_data_id = get_dataID_from_user_state(request)
    return request, chosen_dataset_ID, chosen_label_ID, last_viewed_data_id


def setupDataset_detail(request, datasetID):
    dataset_ID = datasetID
    # print("setupDataset_detail")
    if request.session.get('dataset_ID', 0) != dataset_ID:
        request, chosen_dataset = setupDataset_labels(request, dataset_ID)
        # print(chosen_dataset, "setupDataset_detail")
        if not chosen_dataset:
            return request, 0

    return request, request.session['dataset_ID']


# print(request.user.username)
def setupLabel_detail(request, labelID):
    '''
    Helper function
    '''
    chosen_label = Labels.objects.filter(dataset__dataset_ID=request.session['dataset_ID'], label_ID=labelID)
    # chosen_label=[]
    if request.session.get('label_ID', 0) != labelID:
        if not chosen_label:
            return request, 0

    request.session['label_ID'] = labelID
    request.session['label_name'] = chosen_label[0].name
    return request, request.session['label_ID']


def get_dataID_from_user_state(request):
    '''
    If user_state has entry for the dataset_ID and label_ID : get state from the corresponding row
    otherwise												: create an entry and set it to lowest possible id i.e. ''
    '''
    cur_data_id = ''
    user = User.objects.filter(username=request.user.username)[:1]
    user_state = User_state.objects.filter(user=user, label__label_ID=request.session['label_ID'], label__dataset__dataset_ID=request.session['dataset_ID'])
    if len(user_state) < 1:
        label = Labels.objects.get(dataset__dataset_ID=request.session['dataset_ID'], label_ID=request.session['label_ID'])
        User_state.objects.create(user=user[0], label=label)
    else:
        cur_data_id = user_state[0].cur_data_id

    request.session['last_viewed_data_id'] = cur_data_id  # TODO: Doubt

    return request, request.session['last_viewed_data_id']


@login_required
def annotate(request):
    # selected_choice = {}

    annotation_response = {}
    for i in range(ITEMS_PER_PAGE):
        if 'data_id{}'.format(i + 1) in request.POST and 'choice{}'.format(i + 1) in request.POST:
            annotation_response[request.POST['data_id{}'.format(i + 1)]] = int(request.POST['choice{}'.format(i + 1)])
        else:
            break
    data_id_list = []
    for key, _val in annotation_response.items():
        data_id_list.append(key)

    dataRows = Data.objects.filter(label__label_ID=request.session['label_ID'], label__dataset__dataset_ID=request.session[
                                   'dataset_ID'], data_ID__in=data_id_list).order_by('data_ID')
    record_anotation(request, dataRows, annotation_response)
    # print(type(request.session['dataset_ID']))
    return redirect(reverse('imageannotation:detail', kwargs={'datasetID': request.session['dataset_ID'], 'labelID': request.session['label_ID'], 'page': 'next'}))


@transaction.atomic
def record_anotation(request, dataRows, annotation_response):

    dataset_ID = request.session['dataset_ID']
    label_ID = request.session['label_ID']
    skip_count = 0
    yes_count = 0
    no_count = 0
    for data in dataRows:
        ann = annotation_response[data.data_ID]
        if(ann == 0):
            data.skip_count += 1
            skip_count += 1
        elif ann == 1:
            data.yes_count += 1
            yes_count += 1
        elif ann == 2:
            data.no_count += 1
            no_count += 1
        data.save()
    # Update User_state
    dataRows_count = len(dataRows)
    if dataRows_count:
        request.session['last_viewed_data_id'] = request.session['reset_cur_data_id']
        user = User.objects.filter(username=request.user.username)[:1]
        user_state = User_state.objects.filter(
            user=user, label__dataset__dataset_ID=dataset_ID, label__label_ID=label_ID).get()
        user_state.cur_data_id = request.session['last_viewed_data_id']
        user_state.skip_count = user_state.skip_count + skip_count
        user_state.yes_count = user_state.yes_count + yes_count
        user_state.no_count = user_state.no_count + no_count
        user_state.save()
    user_state = User_state.objects.filter(
        user__username=request.user.username, label__dataset__dataset_ID=dataset_ID, label__label_ID=label_ID).get()

# @transaction.atomic
# def db_annotate(images, selected_choice):
# 	choice_dict = {'1' : True, '2' : False}
# 	for image in images:
# 		image.annotated = True
# 		image.validation = choice_dict[selected_choice[image.image_id]]
# 		image.save(update_fields = ['annotated', 'validation'])


@login_required
def results(request, image_id):
    # image = get_object_or_404(Annotated_images, pk=image_id)
    return HttpResponse('<h1> Annotation done !</h1>')
