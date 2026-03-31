from django.db import models
from django.contrib.auth.models import User


class EmailCheck(models.Model):
    RESULT_CHOICES = [
        ('spam', 'Spam'),
        ('safe', 'Safe'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='email_checks')
    email_text = models.TextField()
    prediction_result = models.CharField(max_length=10, choices=RESULT_CHOICES)
    confidence_score = models.FloatField()
    date_checked = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date_checked']

    def __str__(self):
        return f"{self.user.username} - {self.prediction_result} ({self.date_checked.strftime('%Y-%m-%d %H:%M')})"

    def email_preview(self):
        return self.email_text[:120] + '...' if len(self.email_text) > 120 else self.email_text
