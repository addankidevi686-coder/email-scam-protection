from django.contrib import admin
from .models import EmailCheck


@admin.register(EmailCheck)
class EmailCheckAdmin(admin.ModelAdmin):
    list_display = ('user', 'prediction_result', 'confidence_score', 'email_preview', 'date_checked')
    list_filter = ('prediction_result', 'date_checked', 'user')
    search_fields = ('user__username', 'email_text')
    readonly_fields = ('date_checked',)
    ordering = ('-date_checked',)

    def email_preview(self, obj):
        return obj.email_preview()
    email_preview.short_description = 'Email Preview'
