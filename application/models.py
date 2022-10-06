from typing import Any

import dateutil.utils
import django
# Create your models here.
from cloudinary_storage import storage

from django.db import models

import uuid

from django.db import models
from django.urls.base import reverse


class DataBt(models.Model):
    idDataBt = models.CharField(max_length=50)
    dateOrigine = models.DateField()
    sniNote = models.IntegerField()
    pd = models.FloatField()
    # IdStat = models.ForeignKey(application.models.statImport)


class utilisateur(models.Model):
    idEmpl = models.IntegerField()
    nom = models.CharField(max_length=30)
    prenom = models.CharField(max_length=30)


class facteurMacro(models.Model):
    idFM = models.IntegerField()
    denominationFm = models.CharField(max_length=30)
    valeurFM = models.FloatField()
    anneeFM = models.IntegerField()


class dataSimulee(models.Model):
    idSim = models.IntegerField()
    anneeSim = models.IntegerField()
    sniNote = models.IntegerField()
    pdSim = models.FloatField()
    facteurMacro1 = models.FloatField()
    facteurMacro2 = models.FloatField()


class scenario(models.Model):
    idScenario = models.IntegerField()
    dateScenario = models.DateField(default=django.utils.timezone.now)
    typeScenario = models.CharField(max_length=20)
    sniNote = models.IntegerField()
    sniNoteStr = models.CharField(max_length=25, default='nonNoté')
    pdScenario = models.FloatField()
    mo = models.CharField(max_length=25, default='nonIndique')
    an = models.IntegerField(default=9999)


class statImport(models.Model):
    nbrLigneImporte = models.IntegerField()
    pdMoyenne = models.FloatField(default=99)
    dateImport = models.DateField(default=django.utils.timezone.now)


class historiqueScenario(models.Model):
    dateScen = models.DateField(default=django.utils.timezone.now)
    chomInf = models.FloatField()
    chomSup = models.FloatField()
    pibInf = models.FloatField()
    pibSup = models.FloatField()


class note(models.Model):
    noteA = models.FloatField(default=99)
    noteB = models.FloatField(default=99)
    noteCplus = models.FloatField(default=99)
    noteC = models.FloatField(default=99)
    noteCmoins = models.FloatField(default=99)
    noteD = models.FloatField(default=99)
    noteE = models.FloatField(default=99)
    noteF = models.FloatField(default=99)


class metrique(models.Model):
    nomDuModele = models.CharField(max_length=10)
    test_score = models.FloatField(default=99)
    R2 = models.FloatField(default=99)
    MAE = models.FloatField(default=99)
    RMSE = models.FloatField(default=99)
    MAE = models.FloatField(default=99)
    Median = models.FloatField(default=99)
    dateCreationModele = models.DateField(default=django.utils.timezone.now)


class historiqueFWL(models.Model):
    dateScenFWL = models.DateField(default=django.utils.timezone.now)
    chomInfFWL = models.FloatField()
    chomSupFWL = models.FloatField()
    pibInfFWL = models.FloatField()
    pibSupFWL = models.FloatField()


class metriques(models.Model):
    Sensitivity = models.FloatField()
    Specificity = models.FloatField()
    f1_score = models.FloatField()
    acc_test = models.FloatField()
    acc_train = models.FloatField()
    acc_val = models.FloatField()
    nomMod = models.CharField(max_length=25, default='nonIndique')


class Record(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    voice_record = models.FileField(
        upload_to="records", storage=storage.RawMediaCloudinaryStorage())
    language = models.CharField(max_length=50, null=True, blank=True)
