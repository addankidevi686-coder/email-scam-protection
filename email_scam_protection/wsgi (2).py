import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'email_scam_protection.settings')
application = get_wsgi_application()
