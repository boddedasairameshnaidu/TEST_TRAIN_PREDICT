# Generated by Django 3.1.3 on 2024-05-24 11:01

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('TTP', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='user',
            old_name='password',
            new_name='pwd',
        ),
    ]