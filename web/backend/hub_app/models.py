from django.db import models
from django.contrib.auth.models import User

class ModelWeight(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    file = models.FileField(upload_to='model_weights/')
    uploader = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    downloads = models.IntegerField(default=0)
    metadata = models.JSONField(default=dict)

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    file = models.FileField(upload_to='datasets/')
    uploader = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    downloads = models.IntegerField(default=0)
    metadata = models.JSONField(default=dict)
