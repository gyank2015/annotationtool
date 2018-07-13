from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UserRegistrationForm
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from .models import dataset_type


def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            userObj = form.cleaned_data
            username = userObj['username']
            email = userObj['email']
            password = userObj['password']
            User.objects.create_user(username, email, password)
            user = authenticate(username=username, password=password)
            login(request, user)
            return HttpResponseRedirect(reverse('landing'))

    else:
        form = UserRegistrationForm()
    return render(request, 'annotationtools/register.html', {'form': form})


@login_required
def landing(request):
    annotation_types = [{'choice': item[1], 'link': str.lower('/' + item[1])} for item in dataset_type]

    return render(request, 'annotationtools/landing.html', {'annotation_types': annotation_types})


def home(request):
    return render(request, 'annotationtools/home.html')
