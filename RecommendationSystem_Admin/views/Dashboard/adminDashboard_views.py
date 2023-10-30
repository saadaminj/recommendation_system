from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout
from RecommendationSystem_Admin.models import *
from RecommendationSystem_App.models import *

def AdminDashboardView(request):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {}
            return render(request,'./AdminDashboard.html',context=context)
        else:
            if request.method == 'POST':
                return redirect('AdminDashboardView')
            else:
                return redirect('AdminDashboardView')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')
