from django.contrib import admin
from .models import Video, Annotation_details, Annotated_videos

admin.site.register(Video)
admin.site.register(Annotated_videos)
admin.site.register(Annotation_details)
# Register your models here.
