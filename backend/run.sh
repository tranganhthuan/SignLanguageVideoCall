python manage.py flush --no-input
python manage.py migrate
export DJANGO_SETTINGS_MODULE=backend.settings
daphne -v2 -b 127.0.0.1 -p 8000 backend.asgi:application