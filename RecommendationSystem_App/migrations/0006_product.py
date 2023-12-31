# Generated by Django 4.1.5 on 2023-03-10 15:32

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('RecommendationSystem_App', '0005_delete_product'),
    ]

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('poster', models.ImageField(upload_to='uploads/')),
                ('product_category', models.ForeignKey(blank=True, default='', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='Product_Category', to='RecommendationSystem_App.category')),
                ('product_subcategory', models.ForeignKey(blank=True, default='', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='Product_SubCategory', to='RecommendationSystem_App.subcategory')),
            ],
        ),
    ]
