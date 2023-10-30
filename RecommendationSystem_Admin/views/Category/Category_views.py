from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout
from RecommendationSystem_Admin.models import *
from RecommendationSystem_App.models import *
from django.http.response import JsonResponse

def CategoryList(request):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'categoryObj':Category.objects.all().order_by('category_name')
            }
            return render(request,'./Category/CategoryList.html',context=context)
        else:
            if request.method == 'POST':
                return redirect('CategoryList')
            else:
                return redirect('CategoryList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def addNewCategory(request):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
            }
            return render(request,'./Category/AddNewCategory.html',context=context)
        else:
            if request.method == 'POST':
                category_name = request.POST.get('category_name')
                if not Category.objects.filter(category_name = category_name).exists():
                    Category.objects.create(
                        category_name = category_name
                    )
                    messages.success(request,'New Category Added Successfully.')
                    return redirect('CategoryList')
                else:
                    messages.error(request,'Category Already Existed.')
                    return redirect('CategoryList')
            else:
                return redirect('CategoryList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def editCategoryDetails(request,id):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'categoryObj':Category.objects.filter(pk = id)
            }
            return render(request,'./Category/EditCategoryInfo.html',context=context)
        else:
            if request.method == 'POST':
                category_name = request.POST.get('category_name')
                if not Category.objects.filter(category_name = category_name).exists():
                    Category.objects.filter(pk = id).update(
                        category_name = category_name
                    )
                    messages.success(request,'Category Details Updated Successfully.')
                    return redirect('CategoryList')
                else:
                    messages.error(request,'Category Already Existed.')
                    return redirect('CategoryList')
            else:
                return redirect('CategoryList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')


def deleteCategory(request,id):
    count = 0
    lst = []
    if Category.objects.filter(pk=id).exists():
        categoryObj = Category.objects.get(pk=id)
        count = int(len(Subcategory.objects.filter(category = categoryObj)))
        count = count + int(len(Product.objects.filter(product_category = categoryObj)))
        if count == 0:
            Category.objects.filter(pk=id).delete()
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