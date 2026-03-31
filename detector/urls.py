from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('check/', views.check_email, name='check_email'),
    path('history/', views.history, name='history'),
    path('history/delete/<int:pk>/', views.delete_history, name='delete_history'),
    path('history/delete-all/', views.delete_all_history, name='delete_all_history'),
    path('ajax/check/', views.ajax_check_email, name='ajax_check_email'),
    path('api/check/', views.api_check_email, name='api_check_email'),
    path('dashboard/', views.dashboard, name='dashboard'),
]
