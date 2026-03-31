from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='EmailCheck',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email_text', models.TextField()),
                ('prediction_result', models.CharField(choices=[('spam', 'Spam'), ('safe', 'Safe')], max_length=10)),
                ('confidence_score', models.FloatField()),
                ('date_checked', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='email_checks', to='auth.user')),
            ],
            options={
                'ordering': ['-date_checked'],
            },
        ),
    ]
