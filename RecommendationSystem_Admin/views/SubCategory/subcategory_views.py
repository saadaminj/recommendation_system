from django.shortcuts import render, redirect
from django.contrib import messages
from django.http.response import JsonResponse
from RecommendationSystem_Admin.models import *
from RecommendationSystem_App.models import *

def SubCategoryList(request):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'subcategoryObj':Subcategory.objects.all().order_by('subcategory_name')
            }
            return render(request,'./SubCategory/SubCategoryList.html',context=context)
        else:
            if request.method == 'POST':
                return redirect('SubCategoryList')
            else:
                return redirect('SubCategoryList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def addNewSubCategoryList(request):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'categoryObj':Category.objects.all()
            }
            return render(request,'./SubCategory/AddNewSubCategory.html',context=context)
        else:
            if request.method == 'POST':
                subcategory_name = request.POST.get('subcategory_name')
                category = request.POST.get('category')
                if not Subcategory.objects.filter(subcategory_name = subcategory_name,category = Category.objects.get(pk = category)).exists():
                    Subcategory.objects.create(
                        subcategory_name = subcategory_name,
                        category = Category.objects.get(pk = category)
                    )
                    messages.success(request,'New Subcategory Added Successfully.')
                    return redirect('SubCategoryList')
                else:
                    messages.error(request,'Subcategory Already Exists.')
                    return redirect('SubCategoryList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def editSubCategoryList(request,id):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'categoryObj':Category.objects.all(),
                'subcategoryObj':Subcategory.objects.filter(pk = id)
            }
            return render(request,'./SubCategory/EditSubCategoryInfo.html',context=context)
        else:
            if request.method == 'POST':
                subcategory_name = request.POST.get('subcategory_name')
                category = request.POST.get('category')
                if not Subcategory.objects.filter(subcategory_name = subcategory_name,category = Category.objects.get(pk = category)).exists():
                    Subcategory.objects.filter(pk = id).update(
                        subcategory_name = subcategory_name,
                        category = Category.objects.get(pk = category)
                    )
                    messages.success(request,'Subcategory Updated Successfully.')
                    return redirect('SubCategoryList')
                else:
                    messages.error(request,'Subcategory Already Exists.')
                    return redirect('SubCategoryList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def subcategoryInfo(request,id):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {        
                'subcategoryObj':Subcategory.objects.filter(pk = id),
                'productObj':Product.objects.filter(product_subcategory_id = id)
            }
            return render(request,'./SubCategory/SubCategoryInfo.html',context=context)
        else:
            if request.method == 'POST':
                return redirect('SubCategoryList')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')
    
def deleteSubcategory(request,id):
    count = 0
    lst = []
    if Subcategory.objects.filter(pk=id).exists():
        subcategoryObj = Subcategory.objects.get(pk=id)
        count = int(len(Product.objects.filter(product_subcategory = subcategoryObj)))
        if count == 0:
            Subcategory.objects.filter(pk=id).delete()
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