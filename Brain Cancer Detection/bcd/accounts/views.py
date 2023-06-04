from . import forms
from django.views.generic import CreateView
from django.urls import reverse_lazy

# Create your views here.

class SignUp(CreateView):
    form_class = forms.UserCreateForm
    template_name = 'accounts/signup.html'
    success_url = reverse_lazy('login')