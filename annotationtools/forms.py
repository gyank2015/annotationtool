from django import forms
from django.contrib.auth.models import User
from django.core.validators import validate_email


class UserRegistrationForm(forms.Form):
    username = forms.CharField(
        required=True,
        label='Username',
        max_length=32
    )
    email = forms.CharField(
        required=True,
        label='Email',
        max_length=32,
    )
    password = forms.CharField(
        required=True,
        label='Password',
        max_length=32,
        widget=forms.PasswordInput()
    )
    password_repeat = forms.CharField(
        required=True,
        label='Confirm Password',
        max_length=32,
        widget=forms.PasswordInput()
    )

    def clean_email(self):
        # Check for email already registered
        email = self.cleaned_data['email']
        if (User.objects.filter(email=email).exists()):
            raise forms.ValidationError('Email already exists')
        validate_email(email)
        return self.cleaned_data['email']

    def clean_username(self):
        # Check for username already registered
        username = self.cleaned_data['username']
        if (User.objects.filter(username=username).exists()):
            raise forms.ValidationError('Username already exists')
        return self.cleaned_data['username']

    def clean_password_repeat(self):
        # Check for password and password_repeat match
        password = self.cleaned_data['password']
        password_repeat = self.cleaned_data['password_repeat']
        if password != password_repeat:
            raise forms.ValidationError(u'Passwords do not match')
        return self.cleaned_data['password_repeat']
