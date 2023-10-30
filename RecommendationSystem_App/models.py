from django.db import models

# Create your models here.

Choices = (('active','active'),('inactive','inactive'))

class User(models.Model):
    first_name = models.CharField(max_length=100,null=False,blank=False,default='')
    last_name = models.CharField(max_length=100,null=False,blank=False,default='')
    email = models.EmailField(max_length=254,null=False,blank=False,default='')
    phone_no = models.CharField(max_length=20,null=True,blank=True,default='')
    password = models.CharField(max_length=200,null=False,blank=False,default='') 

class Category(models.Model):
    category_name = models.CharField(max_length=100,null=False,blank=False,default='')

class Subcategory(models.Model):
    subcategory_name = models.CharField(max_length=100,null=False,blank=False,default='')
    category = models.ForeignKey(Category, on_delete=models.CASCADE,related_name='Category',null=True,blank=True,default='')
    hlink = models.CharField(max_length=255,null=True,blank=True)

class Product(models.Model):
    product_name = models.CharField(max_length=255,null=True,blank=True,default='')
    product_code = models.CharField(max_length=255,null=True,blank=True,default='')
    product_price = models.CharField(max_length=255,null=True,blank=True,default='')
    poster = models.ImageField(upload_to ='images/',null=True,blank=True,default='')
    product_category = models.ForeignKey(Category, on_delete=models.CASCADE,related_name='Product_Category',null=True,blank=True,default='')
    product_subcategory = models.ForeignKey(Subcategory, on_delete=models.CASCADE,related_name='Product_SubCategory',null=True,blank=True,default='')
