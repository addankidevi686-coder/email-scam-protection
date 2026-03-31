import json
from datetime import date, timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.models import User
from django.db.models import Count, Q

from .forms import SignupForm, EmailCheckForm
from .models import EmailCheck
from .predictor import predict_email, highlight_suspicious_words


def home(request):
    stats = {}
    if request.user.is_authenticated:
        checks = EmailCheck.objects.filter(user=request.user)
        stats = {
            'total': checks.count(),
            'spam': checks.filter(prediction_result='spam').count(),
            'safe': checks.filter(prediction_result='safe').count(),
        }
    return render(request, 'detector/home.html', {'stats': stats})


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.username}! Your account has been created.')
            return redirect('home')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = SignupForm()
    return render(request, 'detector/signup.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            next_url = request.GET.get('next', 'home')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    return render(request, 'detector/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')


@login_required
def check_email(request):
    result = None
    confidence = None
    highlighted_text = None
    email_text = ''

    if request.method == 'POST':
        form = EmailCheckForm(request.POST, request.FILES)
        if form.is_valid():
            email_text = form.cleaned_data.get('email_text', '').strip()
            email_file = form.cleaned_data.get('email_file')

            if email_file:
                try:
                    email_text = email_file.read().decode('utf-8')
                except Exception:
                    messages.error(request, 'Could not read the uploaded file.')
                    return render(request, 'detector/check_email.html', {'form': form})

            if not email_text:
                messages.error(request, 'Please enter or upload email content.')
            else:
                result, confidence = predict_email(email_text)
                highlighted_text = highlight_suspicious_words(email_text)

                EmailCheck.objects.create(
                    user=request.user,
                    email_text=email_text,
                    prediction_result=result,
                    confidence_score=confidence,
                )
    else:
        form = EmailCheckForm()

    return render(request, 'detector/check_email.html', {
        'form': form,
        'result': result,
        'confidence': confidence,
        'highlighted_text': highlighted_text,
        'email_text': email_text,
    })


@login_required
def history(request):
    checks = EmailCheck.objects.filter(user=request.user)
    filter_by = request.GET.get('filter', 'all')
    if filter_by == 'spam':
        checks = checks.filter(prediction_result='spam')
    elif filter_by == 'safe':
        checks = checks.filter(prediction_result='safe')

    return render(request, 'detector/history.html', {
        'checks': checks,
        'filter_by': filter_by,
    })


@login_required
def delete_history(request, pk):
    check = get_object_or_404(EmailCheck, pk=pk, user=request.user)
    check.delete()
    messages.success(request, 'Record deleted successfully.')
    return redirect('history')


@login_required
def delete_all_history(request):
    if request.method == 'POST':
        EmailCheck.objects.filter(user=request.user).delete()
        messages.success(request, 'All history deleted successfully.')
    return redirect('history')


@login_required
def ajax_check_email(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email_text = data.get('email_text', '').strip()
            if not email_text:
                return JsonResponse({'error': 'No email text provided.'}, status=400)

            result, confidence = predict_email(email_text)
            highlighted = highlight_suspicious_words(email_text)

            EmailCheck.objects.create(
                user=request.user,
                email_text=email_text,
                prediction_result=result,
                confidence_score=confidence,
            )

            return JsonResponse({
                'result': result,
                'confidence': confidence,
                'highlighted_text': highlighted,
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid method.'}, status=405)


def api_check_email(request):
    """Public API endpoint for spam detection."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email_text = data.get('email_text', '').strip()
            if not email_text:
                return JsonResponse({'error': 'No email text provided.'}, status=400)
            result, confidence = predict_email(email_text)
            return JsonResponse({
                'result': result,
                'confidence': confidence,
                'is_spam': result == 'spam',
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'POST method required.'}, status=405)

@login_required
def dashboard(request):
    from datetime import date, timedelta
    total_checks = EmailCheck.objects.count()
    total_spam   = EmailCheck.objects.filter(prediction_result='spam').count()
    total_safe   = EmailCheck.objects.filter(prediction_result='safe').count()
    total_users  = User.objects.count()

    spam_pct = round(total_spam / total_checks * 100) if total_checks else 0
    safe_pct = 100 - spam_pct

    top_users_qs = (
        User.objects
        .annotate(
            total=Count('email_checks'),
            spam=Count('email_checks', filter=Q(email_checks__prediction_result='spam')),
            safe=Count('email_checks', filter=Q(email_checks__prediction_result='safe')),
        )
        .filter(total__gt=0)
        .order_by('-total')[:5]
    )
    top_users = [
        {'username': u.username, 'total': u.total, 'spam': u.spam, 'safe': u.safe}
        for u in top_users_qs
    ]

    daily = []
    for i in range(6, -1, -1):
        day = date.today() - timedelta(days=i)
        s = EmailCheck.objects.filter(date_checked__date=day, prediction_result='spam').count()
        f = EmailCheck.objects.filter(date_checked__date=day, prediction_result='safe').count()
        daily.append({'date': day.strftime('%b %d'), 'spam': s, 'safe': f})

    stats = {
        'total_users':  total_users,
        'total_checks': total_checks,
        'total_spam':   total_spam,
        'total_safe':   total_safe,
        'spam_pct':     spam_pct,
        'safe_pct':     safe_pct,
        'top_users':    top_users,
        'daily_json':   json.dumps(daily),
    }
    return render(request, 'detector/dashboard.html', {'stats': stats})
