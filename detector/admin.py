from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'owner', 'label', 'confidence', 'model_version', 'created_at']
    list_filter = ['label', 'created_at']
    readonly_fields = ['created_at']
    search_fields = ['text', 'owner__username']
