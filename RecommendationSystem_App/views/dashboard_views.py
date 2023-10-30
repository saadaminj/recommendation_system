import os
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout
from RecommendationSystem_Admin.models import *
from RecommendationSystem_App.models import *
import subprocess


def HomeView(request):
    if request.session.get('user_id'):        
        user_id = request.session.get('user_id')
        try :
            result = subprocess.run(["python", "NCF_Model.py", str(user_id)], stdout=subprocess.PIPE)
            print(result.stdout)
            print(result)   
            print(result.stdout.decode('utf-8'))
        except Exception as e:
            print(str(e))
        try :
            result = subprocess.run(["python", "xdeepfm_new.py", str(user_id)], stdout=subprocess.PIPE)
            print('Xdeep FM : ',result.stdout)
            print('Xdeep FM : ',result)   
            print('Xdeep FM : ',result.stdout.decode('utf-8'))
        except Exception as e:
            print(str(e))
        try :
            result = subprocess.run(["python", "DQN.py", str(user_id)], stdout=subprocess.PIPE)
            print('DQN : ',result.stdout)
            print('DQN : ',result)   
            print('DQN : ',result.stdout.decode('utf-8'))
        except Exception as e:
            print(str(e))
        context = {}
    else:
        try :
            result = subprocess.run(["python", "NCF_Model.py", str(1)], stdout=subprocess.PIPE)
            # print(result.stdout)
            # print(result)   
            # print(result.stdout.decode('utf-8'))
        except Exception as e:
            print(str(e))
        context = {}
    return render(request,'Home.html',context=context)


def Applogin(request):
    if request.method == 'GET':
        return render(request,'App_Login.html')
    else:
        if request.method == 'POST':
            email = request.POST.get('email')
            password = request.POST.get('password')
            if User.objects.filter(email = email).exists():
                userObj = User.objects.get(email = email)
                if userObj.password == password:
                    request.session['user_id'] = userObj.pk
                    request.session['first_name'] = userObj.first_name
                    request.session['last_name'] = userObj.last_name
                    request.session['email'] = userObj.email
                    messages.success(request,'Welcome to Nosh Ecomm.')
                    return redirect('HomePage')
                else:
                    messages.error(request,'Invalid Password')
                    return redirect('Applogin')  
            else:
                messages.error(request,'No User Found. Register Here First.')
                return redirect('AppSignup')
    
def AppSignUp(request):
    if request.method == 'GET':
        return render(request,'appSignUp.html')
    else:
        if request.method == 'POST':
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            contact_no = request.POST.get('contact_no')
            email = request.POST.get('email')
            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm_password')
            if password == confirm_password:
                if not User.objects.filter(first_name = first_name,last_name = last_name,email = email).exists():
                    User.objects.create(
                        first_name = first_name,
                        last_name = last_name,
                        phone_no = contact_no,
                        email = email,
                        password = password,
                    )
                    messages.success(request,'You Are Registered Successfully.')
                    return redirect('Applogin')
                else:
                    messages.error(request,'User Already Registered. Please Log In First.')
                    return redirect('Applogin')
            else:
                messages.error(request,'Password and Confirm Password Must be same.')
                return redirect('AppSignup')
            
def appLogout(request):
    del request.session['user_id']
    del request.session['first_name']
    del request.session['last_name']
    del request.session['email']
    logout(request)  
    return redirect('HomePage')
