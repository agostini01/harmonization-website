# Data Harmonization

This project implements the data harmonization website by 
Dockerizing Django with Postgres, Gunicorn, and Nginx.

## How to use

### Development

Uses the default Django development server.

1. Rename *config/.env.dev-sample* to *config/.env.dev*.
1. Update the environment variables in the *docker-compose.yml* and *config/.env.dev* files.
1. Build the images and run the containers:

    ```sh
    $ cp config/.env.dev-sample config/.env.dev
    $ docker-compose up -d --build
    ```

    Test it out at [http://localhost:8000](http://localhost:8000). The "app" folder is mounted into the container and your code changes apply automatically.

### Production

Uses gunicorn + nginx.

1. Rename *config/.env.prod-sample* to *config/.env.prod* and *config/.env.prod.db-sample* to *config/.env.prod.db*. Update the environment variables.
1. Build the images and run the containers:

    ```sh
    $ cp config/.env.prod-sample config/.env.prod
    $ cp config/.env.prod.db-sample config/.env.prod.db
    $ docker-compose -f docker-compose.prod.yml up -d --build
    ```

    Test it out at [http://localhost:1337](http://localhost:1337). No mounted folders. To apply changes, the image must be re-built.


# Special Remarks

This project was build on top of different amazing tutorials:

* https://github.com/testdrivenio/django-on-docker
* https://github.com/wsvincent/djangoforprofessionals
