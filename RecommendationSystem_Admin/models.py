from django.db import models

# Create your models here.
class AdminUser(models.Model):
    first_name = models.CharField(max_length=100,null=False,blank=False,default='')
    last_name = models.CharField(max_length=100,null=False,blank=False,default='')
    email = models.EmailField(max_length=254,null=False,blank=False,default='')
    phone_no = models.CharField(max_length=20,null=True,blank=True,default='')
    password = models.CharField(max_length=200,null=False,blank=False,default='') 
