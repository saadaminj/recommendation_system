from django.contrib import admin
from .models import *

# Register your models here.
class AdminUserAdmin(admin.ModelAdmin):
    list_display = [
    'first_name',
    'last_name',
    'email',
    'phone_no',
    'password',
    ]
admin.site.register(AdminUser,AdminUserAdmin)