# Generated by Django 2.2.4 on 2019-08-20 15:14

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='detected',
            name='time_stamp',
            field=models.DateTimeField(default=datetime.datetime.now),
        ),
    ]
