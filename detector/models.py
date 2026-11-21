from django.conf import settings
from django.db import models


class Prediction(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    text = models.TextField(max_length=2000)
    label = models.CharField(max_length=16)
    confidence = models.FloatField(null=True, blank=True)
    probabilities = models.JSONField(default=dict)
    explanation = models.TextField(blank=True, default='')
    model_version = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.label if self.confidence is None else f'{self.label} ({self.confidence:.2%})'
