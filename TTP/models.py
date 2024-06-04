from django.db import models

# Create your models here.

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    pwd = models.CharField(max_length=100)
    
class Res(models.Model):
    alg_name = models.CharField(max_length=100)
    acc = models.FloatField()
    pre = models.FloatField()
    rec = models.FloatField()
    f1 = models.FloatField()