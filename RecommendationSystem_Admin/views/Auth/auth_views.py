from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout
from RecommendationSystem_Admin.models import *
from RecommendationSystem_App.models import *

def login(request):
    if request.method == 'GET':
        if request.session.get('email'):
            return redirect("logout_view")
        else:
            return render(request, "./Auth/login.html") 
    else:
        if request.method == 'POST':
            email = request.POST.get('email')
            password = request.POST.get('password')
            if AdminUser.objects.filter(email=email).exists():
                user = AdminUser.objects.get(email=email)
                if user:
                    if email == user.email and password == user.password:
                        request.session['user_id'] = user.pk
                        request.session['first_name'] = user.first_name
                        request.session['last_name'] = user.last_name
                        request.session['email'] = user.email
                        messages.success(request, "Welcome to the Admin Pannel.")
                        return redirect('AdminDashboardView')
                    else:
                        print("Email and Password Khota Che.")
                        messages.error(request, "Invalid Email Or Password.")
                        return redirect("Login")
                else:
                    messages.error(request, "Invalid Email Or Password , Please Check Once")
                    return redirect("Login")
            else:
                messages.error(request, "Invalid Email Or Password , Please Check Once")
                return redirect("Login")

def signUp(request):
    if request.method == 'GET':
        return render(request,'./Auth/signUp.html')
    else:
        if request.method == 'POST':
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            email = request.POST.get('email')
            password = request.POST.get('password')
            cpassword = request.POST.get('cpassword')
            contact_no = request.POST.get('contact_no')
            if password == cpassword:
                if not AdminUser.objects.filter(first_name = first_name,last_name = last_name,email = email,password = password,contact_no = contact_no).exists():
                    AdminUser.objects.create(
                        first_name = first_name,
                        last_name = last_name,
                        email = email,
                        password = password,
                        contact_no = contact_no
                    )
                    messages.success(request,'User Registered Successfully. Please Login First.')
                    return redirect("Login")
                else:
                    messages.error(request,'User Already Existed. Please Login.')
                    return redirect('Login')
            else:
                messages.error(request,'Confim Password Isnot Valid.')
                return redirect('signUp')
    

def forgetPassword(request):
    return render(request,'./Auth/ForgetPassword.html')

def logout_view(request):
    del request.session['user_id']
    del request.session['first_name']
    del request.session['last_name']
    del request.session['email']
    logout(request)  
    return redirect('Login')
