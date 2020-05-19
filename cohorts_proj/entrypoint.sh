#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
  echo "Waiting for postgres db $SQL_HOST $SQL_PORT ..."

  while ! nc -z $SQL_HOST $SQL_PORT; do
    sleep 0.1
  done

  echo "PostgreSQL started"
fi

# Next line erases the database at every docker run
#python manage.py flush --no-input
python manage.py migrate
python manage.py collectstatic --no-input --clear

exec "$@"
