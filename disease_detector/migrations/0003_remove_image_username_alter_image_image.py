# Generated by Django 5.0.3 on 2024-04-24 17:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("disease_detector", "0002_image"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="image",
            name="username",
        ),
        migrations.AlterField(
            model_name="image",
            name="image",
            field=models.ImageField(upload_to=""),
        ),
    ]
