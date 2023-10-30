from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout
from RecommendationSystem_Admin.models import *
from RecommendationSystem_App.models import *

def productDetails(request):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'productObj':Product.objects.all()
            }
            return render(request,'./Product/ProductList.html',context=context)
        else:
            if request.method == 'POST':
                return redirect('productDetails')
            else:
                return redirect('productDetails')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')


def addNewProduct(request):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'categoryObj':Category.objects.all(),
                'subcategoryObj':Subcategory.objects.all(),   
            }
            return render(request,'./Product/AddNewProduct.html',context=context)
        else:
            if request.method == 'POST':
                product_name = request.POST.get('product_name')
                product_code = request.POST.get('product_code')
                product_price = request.POST.get('product_price')
                poster = request.POST.get('poster')
                product_category = request.POST.get('product_category')
                product_subcategory = request.POST.get('product_subcategory')
                print(
                    'product_name = ',product_name,
                    'product_code = ',product_code,
                    'product_price = ',product_price,
                    'poster = ',poster,
                    'product_category = ',Category.objects.get(pk = product_category),
                    'product_subcategory = ',Subcategory.objects.get(pk = product_subcategory),
                )
                if Product.objects.filter(product_name = product_name,product_code = product_code,product_price = product_price,product_category = product_category,product_subcategory = product_subcategory).exists():
                    messages.error(request,'Product Already Exists.')
                    return redirect('productDetails')
                else:
                    Product.objects.create(
                        product_name = product_name,
                        product_code = product_code,
                        product_price = product_price,
                        poster = poster,
                        product_category = Category.objects.get(pk = product_category),
                        product_subcategory = Subcategory.objects.get(pk = product_subcategory),
                    )
                    messages.success(request,'Product Added Successfully.')
                    return redirect('productDetails')
            else:
                return redirect('productDetails')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def productInfo(request,id):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'productObj':Product.objects.filter(pk = id)
            }
            return render(request,'./Product/ProductInfo.html',context=context)
        else:
            if request.method == 'POST':
                product_name = request.POST.get('product_name')
                product_code = request.POST.get('product_code')
                product_price = request.POST.get('product_price')
                poster = request.POST.get('poster')
                product_category = request.POST.get('product_category')
                product_subcategory = request.POST.get('product_subcategory')
                if Product.objects.filter(product_name = product_name,product_code = product_code,product_price = product_price,product_category = product_category,product_subcategory = product_subcategory).exists():
                    messages.error(request,'Product Already Exists.')
                    return redirect('productDetails')
                else:
                    Product.objects.filter(pk = id).update(
                        product_name = product_name,
                        product_code = product_code,
                        product_price = product_price,
                        poster = poster,
                        product_category = product_category,
                        product_subcategory = product_subcategory
                    )
                    messages.success(request,'Product Updated Successfully.')
                    return redirect('productDetails')
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')

def editProductDetails(request,id):
    if request.session.get('email'):
        if request.method == 'GET':
            context = {
                'categoryObj':Category.objects.all(),
                'subcategoryObj':Subcategory.objects.all(),
                'productObj':Product.objects.filter(pk = id)
            }
            return render(request,'./Product/EditProductInfo.html',context=context)
        else:
            if request.method == 'POST':
                return redirect('productInfo',id)
            else:
                return redirect('productInfo',id)
    else:
        messages.error(request,'Please Login First.')
        return redirect('Login')