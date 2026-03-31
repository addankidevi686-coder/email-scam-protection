from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class SignupForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'class': 'form-control',
        'placeholder': 'Enter your email'
    }))
    first_name = forms.CharField(max_length=50, required=False, widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'First name (optional)'
    }))

    class Meta:
        model = User
        fields = ('username', 'first_name', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            if 'class' not in field.widget.attrs:
                field.widget.attrs['class'] = 'form-control'
            placeholders = {
                'username': 'Choose a username',
                'password1': 'Create a password',
                'password2': 'Confirm your password',
            }
            if field_name in placeholders:
                field.widget.attrs['placeholder'] = placeholders[field_name]


class EmailCheckForm(forms.Form):
    email_text = forms.CharField(
        label='Email Content',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 8,
            'placeholder': 'Paste your email content here to check if it is spam or safe...',
            'id': 'emailText'
        })
    )
    email_file = forms.FileField(
        required=False,
        label='Or Upload Email File (.txt)',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.txt'
        })
    )
