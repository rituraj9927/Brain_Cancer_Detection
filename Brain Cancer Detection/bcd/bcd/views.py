from django.shortcuts import render, redirect
# from .models import *
# from django.contrib.auth.models import User
# from django.contrib.auth import authenticate, login, logout
# from datetime import date

import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image

import keras
from keras.models import load_model
import numpy as np
import math
#from keras.preprocessing import image

from .functions import handle_uploaded_file
from PIL import Image

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

model = load_model("trained_model.h5")

# Create your views here.

class HomePage(TemplateView):
    template_name = 'index.html'

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return HttpResponseRedirect(reverse("welcome"))
        return super().get(request, *args, **kwargs)

class WelcomePage(LoginRequiredMixin, TemplateView):
    template_name = 'welcome.html'

class ThanksPage(TemplateView):
    template_name = 'thanks.html'


@csrf_exempt
def cancer_detection(request):
    d={}
    color = ""
    if request.method == 'POST':
        mri_image = request.FILES['logo']
        handle_uploaded_file(request.FILES['logo'])
        #model = load_model("trained_model.h5")
        folder = os.path.join(BASE_DIR,"static")
        path = os.path.join(folder, str(mri_image))
        img = image.load_img(path, target_size=(224,224))

        im = Image.open(path)
        rgb_im = im.convert("RGB")
        rgb_im.save("static/input.jpg")

        i = image.img_to_array(img)
        input_arr = np.array([i])
        img_preprocessed = preprocess_input(input_arr)
        prediction = model.predict(img_preprocessed)

        pred = model.predict(img_preprocessed)

        if pred[0][0] == 1:
            p = 'MRI is NOT having a Tumor'
            color = 'green'
        else:
            p = 'MRI is having a Tumor'
            color = 'red'
        print(p)
        d = {'result': p, 'res_color': color,'data':99}
    return render(request, "output.html", d)


# def prediction(request):
#     if not request.user.is_authenticated:
#         return redirect('user_login')
#     user = request.user
#     student = StudentUser.objects.get(user=user)
#     mri = MRI.objects.filter(user=student)
#     d = {'mri':mri}
#     return render(request, "prediction.html",d)
