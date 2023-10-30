from django.http.response import JsonResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout
from RecommendationSystem_Admin.models import *
from RecommendationSystem_App.models import *

def appUserList(request):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'appUserObj':User.objects.all()
            }
            return render(request,'./AppUser/AppUserList.html',context=context)
        else:
            if request.method == 'POST':
                return redirect('appUserList')
            else:
                return redirect('appUserList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')
    
def addNewAppUser(request):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                
            }
            return render(request,'./AppUser/AddNewAppUser.html',context=context)
        else:
            if request.method == 'POST':
                first_name = request.POST.get('first_name')
                last_name = request.POST.get('last_name')
                email = request.POST.get('email')
                phone_no = request.POST.get('phone_no')                                  
                if User.objects.filter(first_name = first_name,last_name = last_name,email = email,phone_no = phone_no).exists():
                    messages.error(request,'User Already Exists.')
                    return redirect('appUserList')
                else:
                    User.objects.create(
                        first_name = first_name,
                        last_name = last_name,
                        email = email,
                        phone_no = phone_no
                    )
                    messages.success(request,'New User Added Successfully.')
                    return redirect('appUserList')
            else:
                return redirect('appUserList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def appUserInfo(request,id):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'appuserObj': User.objects.filter(pk = id)
            }
            return render(request,'./AppUser/AppUserDetails.html',context=context)
        else:
            if request.method == 'POST':
                return redirect('appUserList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def editAppUserDetails(request,id):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'appuserObj':User.objects.filter(pk = id)
            }
            return render(request,'./AppUser/EditAppUser.html',context=context)
        else:
            if request.method == 'POST':
                first_name = request.POST.get('first_name')
                last_name = request.POST.get('last_name')
                email = request.POST.get('email')
                phone_no = request.POST.get('phone_no')                                  
                if User.objects.filter(first_name = first_name,last_name = last_name,email = email,phone_no = phone_no).exists():
                    messages.error(request,'User Already Exists.')
                    return redirect('appUserList')
                else:
                    User.objects.filter(pk = id).update(
                        first_name = first_name,
                        last_name = last_name,
                        email = email,
                        phone_no = phone_no
                    )
                    messages.success(request,'User Details Updated Successfully.')
                    return redirect('appUserList')
            else:
                return redirect('appUserList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def deleteAppUser(request,id):
    count = 0
    lst = []
    if User.objects.filter(pk=id).exists():
        if count == 0:
            User.objects.filter(pk=id).delete()
            new = {
                'Flag':'deleted'
            }
            lst.append(new)
            return JsonResponse(lst,safe=False)
        else:
            new = {
                'Flag':'count'
            }
            lst.append(new)
            return JsonResponse(lst,safe=False)
    else:            
        new = {
            'Flag':'recordNot'
        }
        lst.append(new)
        return JsonResponse(lst,safe=False)