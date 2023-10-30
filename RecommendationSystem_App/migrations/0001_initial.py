# Generated by Django 4.1.5 on 2023-03-10 14:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first_name', models.CharField(default='', max_length=100)),
                ('last_name', models.CharField(default='', max_length=100)),
                ('email', models.EmailField(default='', max_length=254)),
                ('phone_no', models.CharField(blank=True, default='', max_length=20, null=True)),
                ('password', models.CharField(default='', max_length=200)),
            ],
        ),
    ]
