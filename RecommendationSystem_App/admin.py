from django.contrib import admin
from .models import *

class UserAdmin(admin.ModelAdmin):
    list_display = [
        'first_name',
        'last_name',
        'email',
        'phone_no',
        'password',
    ]
admin.site.register(User,UserAdmin)


class CategoryAdmin(admin.ModelAdmin):
    list_display = [
        'category_name'
    ]
admin.site.register(Category,CategoryAdmin)
    

class SubcategoryAdmin(admin.ModelAdmin):
    list_display = [
        'subcategory_name',
        'category',
        'hlink',
    ]
admin.site.register(Subcategory,SubcategoryAdmin)

class ProductAdmin(admin.ModelAdmin):
    list_display = [
        'product_name',
        'product_code',
        'product_price',
        'poster',
        'product_category',
        'product_subcategory',
    ]
admin.site.register(Product,ProductAdmin)
