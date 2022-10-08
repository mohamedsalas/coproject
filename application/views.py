import pickle
import sqlite3
import sys
from datetime import datetime
from random import random, randint
import django.utils.timezone
import numpy as np
from django import views
from django.contrib import messages
from django.contrib.messages import SUCCESS, ERROR
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from keras.applications import ResNet50
from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

import application
from application.models import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

##################################### RECORD################################
import wave
import pyaudio
def start_stream(request):
    try:
        if request.method == "GET":
            print('debut')
            var = request.GET.get('statut')
            print(var)
            frames = []
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=22050, input=True, frames_per_buffer=1024)
            for i in range(int(var)):
                data = stream.read(1024)
                frames.append(data)
                print('streaming')
            stream.stop_stream()
            print('close')
            stream.close()
            audio.terminate()
            sound_file = wave.open("enregistrement/aaa.wav", "wb")
            sound_file.setnchannels(1)
            sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            sound_file.setframerate(22050)
            sound_file.writeframes(b''.join(frames))
            sound_file.close()
            print("finished streaming")
            messages.add_message(request, SUCCESS, 'Enregistrement réussi')
        return render(request, 'backoffice/simulation.html')
    except:
        print("erreur")
        return render(request, 'backoffice/simulation.html')
##################################################################################
# Create your views here.
def simulation(request):
    sim = dataSimulee.objects.all()[:5]
    return render(request, 'backoffice/simulation.html', context={"sim": sim})

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


###########################  RETRAIN ##########################


def reTree(request):
    try:
        if request.method == "GET":
            url = request.GET.get('imported')
            urlValid = request.GET.get('importedValidation')
            x = pd.read_csv(url, sep=',')
            y = pd.read_csv(urlValid)
            df = pd.DataFrame(x)
            dfValidation = pd.DataFrame(y)
            print("le dataset de validation est", dfValidation.head())
            del df['Unnamed: 0']
            del dfValidation['Unnamed: 0']
            print(df)
            X1 = df.copy()
            Y1 = df['class']
            To_drop = ['class', 'audio']
            X1 = X1.drop(To_drop, axis=1)

            X1Validation = dfValidation.copy()
            Y1Validation = dfValidation['class']
            To_drop = ['class', 'audio']
            X1Validation = X1Validation.drop(To_drop, axis=1)

            X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, train_size=0.8, stratify=Y1)
            final_model = DecisionTreeClassifier(criterion='gini',
                                                 splitter='best',
                                                 random_state=1)
            final_model.fit(X_train, Y_train)
            dt_y_pred = final_model.predict(X1Validation)
            cm = confusion_matrix(Y1Validation, dt_y_pred)
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            # print('Sensitivity : ', sensitivity)

            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            # print('Specificity : ', specificity)
            # print('f1_score', f1_score(Y1Validation, dt_y_pred))
            # print("acc_test", final_model.score(X_test, Y_test))
            # print("acc_train", final_model.score(X_train, Y_train))
            # print("acc_val", final_model.score(X1Validation, Y1Validation))
            a = sensitivity
            b = specificity
            c = f1_score(Y1Validation, dt_y_pred)
            d = final_model.score(X_test, Y_test)
            e = final_model.score(X_train, Y_train)
            f = final_model.score(X1Validation, Y1Validation)

        cnxn = sqlite3.connect('covid19')
        cnxn.commit()

        cursor2 = cnxn.cursor()
        dff = pd.read_sql("SELECT * FROM main.application_metriques WHERE nomMod  LIKE 'DecisionTree'", con=cnxn)
        dff = pd.DataFrame(dff)
        del dff['id']
        # print(dff)

        cursor2.close()
        modelProd = list(dff.iloc[-1])
        # print(modelProd)
        l = []
        l.append(a)
        l.append(b)
        l.append(c)
        l.append(d)
        l.append(e)
        l.append(f)
        l.append("DecisionTree")
        L = ["Sensitivity", "Specificity", "f1_score", "acc_test", "acc_train", "acc_val", "nomMod"]
        L = np.vstack((L, modelProd))
        L = np.vstack((L, l))
        L = pd.DataFrame(L)
        L = L.rename(columns=L.iloc[0])
        L = L.iloc[1:, :]
        print(L)

        cursor3 = cnxn.cursor()
        cursor3.execute("DELETE  FROM  main.application_metriques WHERE nomMod LIKE 'DecisionTree'")
        cnxn.commit()
        cursor3.close()

        cnxn.commit()
        cursor = cnxn.cursor()
        for index, row in L.iterrows():
            cursor.execute("INSERT INTO  main.application_metriques(Sensitivity,Specificity, f1_score, acc_test,"
                           "acc_train,acc_val,nomMod) values(?,?,?,?,?,?,?)",
                           (row.Sensitivity, row.Specificity, row.f1_score, row.acc_test, row.acc_train, row.acc_val,
                            "DecisionTree"))
        cnxn.commit()
        cursor.close()
        cnxn.commit()
        metriqueDT = metriques.objects.all().filter(nomMod="DecisionTree")
        pickle.dump(final_model, open('model_non_traite/Dtree.pkl', 'wb'))
        return render(request, 'backoffice/DecesionTree.html', context={'metriqueDT': metriqueDT})


    except:
        print("erreur")
    return render(request, 'backoffice/DecesionTree.html')


import shutil


def remplacerDTREE(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'DecisionTree'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_utilise/Dtree.pkl")
        source = 'model_non_traite/Dtree.pkl'
        shutil.copyfile(source, 'model_utilise/Dtree.pkl')
        os.remove("model_non_traite/Dtree.pkl")
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT min(id) FROM main.application_metriques LIMIT2 "
            "WHERE nomMod LIKE 'DecisionTree'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        messages.add_message(request, SUCCESS, 'jerdani')
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('reTree/')


def annulerDtree(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'DecisionTree'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_non_traite/Dtree.pkl")
        cnxn = sqlite3.connect('covid19')
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT id FROM main.application_metriques LIMIT1 "
            "WHERE nomMod LIKE 'DecisionTree'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        messages.add_message(request, SUCCESS, 'jerdani')
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('reTree/')



#from xgboost import XGBClassifier


def reboost(request):
    try:
        if request.method == "GET":
            url = request.GET.get('imported')
            urlValid = request.GET.get('importedValidation')
            x = pd.read_csv(url, sep=',')
            y = pd.read_csv(urlValid, sep=',')
            df = pd.DataFrame(x)
            dfValidation = pd.DataFrame(y)
            print(df.columns)
            print(dfValidation.columns)
            # print("le dataset de validation est", dfValidation.head())
            del df['Unnamed: 0']
            del dfValidation['Unnamed: 0']
            X1 = df.copy()
            Y1 = df['class']
            To_drop = ['class', 'audio']
            X1 = X1.drop(To_drop, axis=1)
            liste_columns = dfValidation.columns
            for i in liste_columns:
                dfValidation[i] = dfValidation[i].astype(float)
            X1Validation = dfValidation.copy()
            Y1Validation = dfValidation['class']
            To_drop = ['class', 'audio']
            X1Validation = X1Validation.drop(To_drop, axis=1)

            X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, train_size=0.8, stratify=Y1)
            #xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=7, min_child_weight=1, gamma=0,
            #                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005,
            #                    objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
            xgb.fit(X_train, Y_train)
            dt_y_pred = xgb.predict(X1Validation)
            cm = confusion_matrix(Y1Validation, dt_y_pred)
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            # print('Sensitivity : ', sensitivity)

            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            # print('Specificity : ', specificity)
            # print('f1_score', f1_score(Y1Validation, dt_y_pred))
            # print("acc_test", final_model.score(X_test, Y_test))
            # print("acc_train", final_model.score(X_train, Y_train))
            # print("acc_val", final_model.score(X1Validation, Y1Validation))
            a = sensitivity
            b = specificity
            c = f1_score(Y1Validation, dt_y_pred)
            d = xgb.score(X_test, Y_test)
            e = xgb.score(X_train, Y_train)
            f = xgb.score(X1Validation, Y1Validation)

        cnxn = sqlite3.connect('covid19')
        cnxn.commit()

        cursor2 = cnxn.cursor()
        dff = pd.read_sql("SELECT * FROM main.application_metriques WHERE nomMod  LIKE 'Xgboost'", con=cnxn)
        dff = pd.DataFrame(dff)
        del dff['id']
        # print(dff)

        cursor2.close()
        modelProd = list(dff.iloc[-1])
        # print(modelProd)
        l = []
        l.append(a)
        l.append(b)
        l.append(c)
        l.append(d)
        l.append(e)
        l.append(f)
        l.append("Xgboost")
        L = ["Sensitivity", "Specificity", "f1_score", "acc_test", "acc_train", "acc_val", "nomMod"]
        L = np.vstack((L, modelProd))
        L = np.vstack((L, l))
        L = pd.DataFrame(L)
        L = L.rename(columns=L.iloc[0])
        L = L.iloc[1:, :]
        print(L)

        cursor3 = cnxn.cursor()
        cursor3.execute("DELETE  FROM  main.application_metriques WHERE nomMod LIKE 'Xgboost'")
        cnxn.commit()
        cursor3.close()

        cursor = cnxn.cursor()
        for index, row in L.iterrows():
            cursor.execute("INSERT INTO  main.application_metriques(Sensitivity,Specificity, f1_score, acc_test,"
                           "acc_train,acc_val,nomMod) values(?,?,?,?,?,?,?)",
                           (row.Sensitivity, row.Specificity, row.f1_score, row.acc_test, row.acc_train, row.acc_val,
                            "Xgboost"))
        cnxn.commit()
        cursor.close()
        cnxn.commit()
        metriqueDT = metriques.objects.all().filter(nomMod="Xgboost")
        pickle.dump(xgb, open('model_non_traite/xgb.pkl', 'wb'))
        return render(request, 'backoffice/metriqueRidge.html', context={'metriqueDT': metriqueDT})
    except:
        print("erreur")
        return render(request, 'backoffice/metriqueRidge.html')


def remplacerxgb(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'Xgboost'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_utilise/xgb.pkl")
        source = 'model_non_traite/xgb.pkl'
        shutil.copyfile(source, 'model_utilise/xgb.pkl')
        os.remove("model_non_traite/xgb.pkl")
        cnxn = sqlite3.connect('covid19')
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT min(id) FROM main.application_metriques LIMIT2 "
            "WHERE nomMod LIKE 'Xgboost'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        messages.add_message(request, SUCCESS, 'jerdani')
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('reboost/')


def annulerxgb(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'Xgboost'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_non_traite/xgb.pkl")
        cnxn = sqlite3.connect('covid19')
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT id FROM main.application_metriques LIMIT1 "
            "WHERE nomMod LIKE 'Xgboost'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('reboost/')


def resvm(request):
    try:
        if request.method == "GET":
            url = request.GET.get('imported')
            urlValid = request.GET.get('importedValidation')
            x = pd.read_csv(url, sep=',')
            y = pd.read_csv(urlValid, sep=',')
            df = pd.DataFrame(x)
            dfValidation = pd.DataFrame(y)
            print(df.columns)
            print(dfValidation.columns)
            # print("le dataset de validation est", dfValidation.head())
            del df['Unnamed: 0']
            del dfValidation['Unnamed: 0']
            X1 = df.copy()
            Y1 = df['class']
            To_drop = ['class', 'audio']
            X1 = X1.drop(To_drop, axis=1)
            liste_columns = dfValidation.columns
            for i in liste_columns:
                dfValidation[i] = dfValidation[i].astype(float)
            X1Validation = dfValidation.copy()
            Y1Validation = dfValidation['class']
            To_drop = ['class', 'audio']
            X1Validation = X1Validation.drop(To_drop, axis=1)

            X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, train_size=0.8, stratify=Y1)
            model = svm.SVC(kernel='rbf')
            model.fit(X_train, Y_train)
            dt_y_pred = model.predict(X1Validation)
            cm = confusion_matrix(Y1Validation, dt_y_pred)
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            # print('Sensitivity : ', sensitivity)

            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            # print('Specificity : ', specificity)
            # print('f1_score', f1_score(Y1Validation, dt_y_pred))
            # print("acc_test", final_model.score(X_test, Y_test))
            # print("acc_train", final_model.score(X_train, Y_train))
            # print("acc_val", final_model.score(X1Validation, Y1Validation))
            a = sensitivity
            b = specificity
            c = f1_score(Y1Validation, dt_y_pred)
            d = model.score(X_test, Y_test)
            e = model.score(X_train, Y_train)
            f = model.score(X1Validation, Y1Validation)

        cnxn = sqlite3.connect('covid19')
        cnxn.commit()

        cursor2 = cnxn.cursor()
        dff = pd.read_sql("SELECT * FROM main.application_metriques WHERE nomMod  LIKE 'SVM'", con=cnxn)
        dff = pd.DataFrame(dff)
        del dff['id']
        # print(dff)

        cursor2.close()
        modelProd = list(dff.iloc[-1])
        # print(modelProd)
        l = []
        l.append(a)
        l.append(b)
        l.append(c)
        l.append(d)
        l.append(e)
        l.append(f)
        l.append("SVM")
        L = ["Sensitivity", "Specificity", "f1_score", "acc_test", "acc_train", "acc_val", "nomMod"]
        L = np.vstack((L, modelProd))
        L = np.vstack((L, l))
        L = pd.DataFrame(L)
        L = L.rename(columns=L.iloc[0])
        L = L.iloc[1:, :]
        print(L)

        cursor3 = cnxn.cursor()
        cursor3.execute("DELETE  FROM  main.application_metriques WHERE nomMod LIKE 'SVM'")
        cnxn.commit()
        cursor3.close()
        cursor = cnxn.cursor()

        for index, row in L.iterrows():
            cursor.execute("INSERT INTO  main.application_metriques(Sensitivity,Specificity, f1_score, acc_test,"
                           "acc_train,acc_val,nomMod) values(?,?,?,?,?,?,?)",
                           (row.Sensitivity, row.Specificity, row.f1_score, row.acc_test, row.acc_train, row.acc_val,
                            "SVM"))
        cnxn.commit()
        cursor.close()
        cnxn.commit()
        metriqueDT = metriques.objects.all().filter(nomMod="SVM")
        pickle.dump(model, open('model_non_traite/svm.pkl', 'wb'))
        return render(request, 'backoffice/SVM.html', context={'metriqueDT': metriqueDT})
    except:
        print("erreur")
        return render(request, 'backoffice/SVM.html')


def remplacersvm(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'SVM'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_utilise/svm.pkl")
        source = 'model_non_traite/svm.pkl'
        shutil.copyfile(source, 'model_utilise/svm.pkl')
        os.remove("model_non_traite/svm.pkl")
        cnxn = sqlite3.connect('covid19')
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT min(id) FROM main.application_metriques LIMIT2 "
            "WHERE nomMod LIKE 'SVM'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('resvm/')


def annulersvm(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'SVM'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_non_traite/svm.pkl")
        cnxn = sqlite3.connect('covid19')
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT id FROM main.application_metriques LIMIT1 "
            "WHERE nomMod LIKE 'SVM'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('resvm/')


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Convolution1D, MaxPool1D, Input, \
    GlobalAveragePooling2D
from keras import losses, models, optimizers
from keras.activations import relu, softmax
#import tensorflow as tf
import os
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import save_model


def recnn(request):
    try:
        if request.method == "GET":

            url = request.GET.get('imported')
            urlValid = request.GET.get('importedValidation')
            print('aaaaaaaa')
            liste_mfcc = os.listdir(url)
            k = 0
            y = []
            mfcc_tenseur = np.empty((len(liste_mfcc), 198, 151, 1), dtype=np.float64)
            for li in liste_mfcc:
                mfcc_df = pd.read_csv(url + '\\' + li)

                mfcc_df = mfcc_df.iloc[:, 1:]
                y.append(mfcc_df['class'][0])
                mfcc_df.drop(['class'], axis=1, inplace=True)
                mfcc_df.drop(['audio'], axis=1, inplace=True)
                # print(mfcc_df.shape)
                mfcc_reshape = mfcc_df.to_numpy().reshape(198, 151, 1)
                # rgb_batch = np.repeat(mfcc_reshape, 3, -1)
                mfcc_tenseur[k, :, :, :] = mfcc_reshape
                k += 1
            print('bbbbbbb')

            liste_mfcc1 = os.listdir(urlValid)
            k = 0
            y1 = []
            mfcc_tenseur1 = np.empty((len(liste_mfcc1), 198, 151, 1), dtype=np.float64)
            for li in liste_mfcc1:
                mfcc_df = pd.read_csv(urlValid + '\\' + li)
                mfcc_df = mfcc_df.iloc[:, 1:]
                y1.append(mfcc_df['class'][0])
                mfcc_df.drop(['class'], axis=1, inplace=True)
                mfcc_df.drop(['audio'], axis=1, inplace=True)
                # print(mfcc_df.shape)
                mfcc_reshape = mfcc_df.to_numpy().reshape(198, 151, 1)
                # rgb_batch = np.repeat(mfcc_reshape, 3, -1)
                mfcc_tenseur1[k, :, :, :] = mfcc_reshape
                k += 1
            print('ccccccccccccc')
            num_rows = 198
            num_columns = 151
            num_channels = 1
            learning_rate = 0.001
            model = Sequential()
            model.add(
                Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
            model.add(MaxPooling2D(pool_size=2))
            model.add(Dropout(0.2))

            model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
            model.add(MaxPooling2D(pool_size=1))
            model.add(Dropout(0.2))

            model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
            model.add(MaxPooling2D(pool_size=1))
            model.add(Dropout(0.2))

            model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
            model.add(MaxPooling2D(pool_size=1))
            model.add(Dropout(0.2))
            model.add(GlobalAveragePooling2D())

            model.add(Dense(2, activation='softmax'))
            #optimizer = tf.optimizers.Adam(learning_rate)
            model.compile(optimizer=optimizer, loss=losses.binary_crossentropy, metrics=['acc'])
            EPOCHES = 2
            trainY = to_categorical(y)
            valY = to_categorical(y1)
            model.fit(mfcc_tenseur, trainY, epochs=EPOCHES, batch_size=1, validation_data=(mfcc_tenseur1, valY))
            rounded_predictions = model.predict(mfcc_tenseur1)
            r = pd.DataFrame(rounded_predictions)
            rounded_predictions1 = model.predict(mfcc_tenseur)
            r1 = pd.DataFrame(rounded_predictions1)
            cm = confusion_matrix(y1, round(r[1]))
            cm1 = confusion_matrix(y, round(r1[1]))
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            a = sensitivity
            b = specificity
            c = f1_score(y1, round(r[1]))
            d = (cm1[0][0] + cm1[1][1]) / np.sum(cm1)
            e = (cm1[0][0] + cm1[1][1]) / np.sum(cm1)
            f = (cm[0][0] + cm[1][1]) / np.sum(cm)
        cnxn = sqlite3.connect('covid19')
        cnxn.commit()

        cursor2 = cnxn.cursor()
        dff = pd.read_sql("SELECT * FROM main.application_metriques WHERE nomMod  LIKE 'CNN'", con=cnxn)
        dff = pd.DataFrame(dff)
        del dff['id']
        # print(dff)

        cursor2.close()
        print('dddddddd')

        # print(dff)

        modelProd = list(dff.iloc[-1])
        # print(modelProd)
        l = []
        l.append(a)
        l.append(b)
        l.append(c)
        l.append(d)
        l.append(e)
        l.append(f)
        l.append("CNN")
        L = ["Sensitivity", "Specificity", "f1_score", "acc_test", "acc_train", "acc_val", "nomMod"]
        L = np.vstack((L, modelProd))
        L = np.vstack((L, l))
        L = pd.DataFrame(L)
        L = L.rename(columns=L.iloc[0])
        L = L.iloc[1:, :]
        print(L)

        cursor3 = cnxn.cursor()
        cursor3.execute("DELETE  FROM  main.application_metriques WHERE nomMod LIKE 'CNN'")
        cnxn.commit()
        cursor3.close()
        cursor = cnxn.cursor()

        for index, row in L.iterrows():
            cursor.execute("INSERT INTO  main.application_metriques(Sensitivity,Specificity, f1_score, acc_test,"
                           "acc_train,acc_val,nomMod) values(?,?,?,?,?,?,?)",
                           (row.Sensitivity, row.Specificity, row.f1_score, row.acc_test, row.acc_train, row.acc_val,
                            "CNN"))
        cnxn.commit()
        cursor.close()
        cnxn.commit()
        metriqueDT = metriques.objects.all().filter(nomMod="CNN")
        save_model(model, "model_non_traite/cnn.h5")
        return render(request, 'backoffice/CNN.html', context={'metriqueDT': metriqueDT})
    except:
        return render(request, 'backoffice/CNN.html')


def remplacercnn(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'CNN'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_utilise/cnn.H5")
        source = 'model_non_traite/cnn.H5'
        shutil.copyfile(source, 'model_utilise/cnn.H5')
        os.remove("model_non_traite/cnn.H5")
        cnxn = sqlite3.connect('covid19')
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT min(id) FROM main.application_metriques LIMIT2 "
            "WHERE nomMod LIKE 'CNN'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('recnn/')


def annulercnn(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'CNN'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_non_traite/cnn.H5")
        cnxn = sqlite3.connect('covid19')
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT id FROM main.application_metriques LIMIT1 "
            "WHERE nomMod LIKE 'CNN'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('recnn/')


from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers


def reresnet50(request):
    try:
        if request.method == "GET":

            url = request.GET.get('imported')
            urlValid = request.GET.get('importedValidation')
            print('aaaaaaaa')
            liste_mfcc = os.listdir(url)
            k = 0
            y = []
            mfcc_tenseur = np.empty((len(liste_mfcc), 198, 151, 3), dtype=np.float64)
            for li in liste_mfcc:
                mfcc_df = pd.read_csv(url + '\\' + li)

                mfcc_df = mfcc_df.iloc[:, 1:]
                y.append(mfcc_df['class'][0])
                mfcc_df.drop(['class'], axis=1, inplace=True)
                mfcc_df.drop(['audio'], axis=1, inplace=True)
                # print(mfcc_df.shape)
                mfcc_reshape = mfcc_df.to_numpy().reshape(198, 151, 1)
                rgb_batch = np.repeat(mfcc_reshape, 3, -1)
                mfcc_tenseur[k, :, :, :] = rgb_batch
                k += 1
            print('bbbbbbb')

            liste_mfcc1 = os.listdir(urlValid)
            k = 0
            y1 = []
            mfcc_tenseur1 = np.empty((len(liste_mfcc1), 198, 151, 3), dtype=np.float64)
            for li in liste_mfcc1:
                mfcc_df = pd.read_csv(urlValid + '\\' + li)
                mfcc_df = mfcc_df.iloc[:, 1:]
                y1.append(mfcc_df['class'][0])
                mfcc_df.drop(['class'], axis=1, inplace=True)
                mfcc_df.drop(['audio'], axis=1, inplace=True)
                # print(mfcc_df.shape)
                mfcc_reshape = mfcc_df.to_numpy().reshape(198, 151, 1)
                rgb_batch = np.repeat(mfcc_reshape, 3, -1)
                mfcc_tenseur1[k, :, :, :] = rgb_batch
                k += 1
            print('ccccccccccccc')
            restnet = ResNet50(include_top=False, input_shape=(198, 151, 3))
            output = restnet.layers[-1].output
            output = keras.layers.Flatten()(output)
            restnet = Model(restnet.input, outputs=output)
            for layer in restnet.layers:
                layer.trainable = True
            restnet.summary()
            model = Sequential()
            model.add(restnet)
            model.add(Dense(2, activation='softmax'))
            model.compile(loss='binary_crossentropy',
                          optimizer=optimizers.RMSprop(lr=2e-5),
                          metrics=['accuracy'])
            EPOCHES = 1
            trainY = to_categorical(y)
            valY = to_categorical(y1)
            H = model.fit(mfcc_tenseur, trainY, epochs=EPOCHES, batch_size=1, validation_data=(mfcc_tenseur1, valY))
            rounded_predictions = model.predict(mfcc_tenseur1)
            r = pd.DataFrame(rounded_predictions)
            rounded_predictions1 = model.predict(mfcc_tenseur)
            r1 = pd.DataFrame(rounded_predictions1)
            cm = confusion_matrix(y1, round(r[1]))
            cm1 = confusion_matrix(y, round(r1[1]))
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            a = sensitivity
            b = specificity
            c = f1_score(y1, round(r[1]))
            d = (cm1[0][0] + cm1[1][1]) / np.sum(cm1)
            e = (cm1[0][0] + cm1[1][1]) / np.sum(cm1)
            f = (cm[0][0] + cm[1][1]) / np.sum(cm)
        cnxn = sqlite3.connect('covid19')
        cnxn.commit()

        cursor2 = cnxn.cursor()
        dff = pd.read_sql("SELECT * FROM main.application_metriques WHERE nomMod  LIKE 'RESNET50'", con=cnxn)
        dff = pd.DataFrame(dff)
        del dff['id']
        # print(dff)

        cursor2.close()
        print('dddddddd')

        # print(dff)

        modelProd = list(dff.iloc[-1])
        # print(modelProd)
        l = []
        l.append(a)
        l.append(b)
        l.append(c)
        l.append(d)
        l.append(e)
        l.append(f)
        l.append("RESNET50")
        L = ["Sensitivity", "Specificity", "f1_score", "acc_test", "acc_train", "acc_val", "nomMod"]
        L = np.vstack((L, modelProd))
        L = np.vstack((L, l))
        L = pd.DataFrame(L)
        L = L.rename(columns=L.iloc[0])
        L = L.iloc[1:, :]
        print(L)

        cursor3 = cnxn.cursor()
        cursor3.execute("DELETE  FROM  main.application_metriques WHERE nomMod LIKE 'RESNET50'")
        cnxn.commit()
        cursor3.close()
        cursor = cnxn.cursor()

        for index, row in L.iterrows():
            cursor.execute("INSERT INTO  main.application_metriques(Sensitivity,Specificity, f1_score, acc_test,"
                           "acc_train,acc_val,nomMod) values(?,?,?,?,?,?,?)",
                           (row.Sensitivity, row.Specificity, row.f1_score, row.acc_test, row.acc_train, row.acc_val,
                            "RESNET50"))
        cnxn.commit()
        cursor.close()
        cnxn.commit()
        metriqueDT = metriques.objects.all().filter(nomMod="RESNET50")
        save_model(model, "model_non_traite/resnet50.h5")
        return render(request, 'backoffice/RESNET50.html', context={'metriqueDT': metriqueDT})
    except:
        return render(request, 'backoffice/RESNET50.html')


def remplacerresnet50(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'RESNET50'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_utilise/resnet50.H5")
        source = 'model_non_traite/resnet50.H5'
        shutil.copyfile(source, 'model_utilise/resnet50.H5')
        os.remove("model_non_traite/resnet50.H5")
        cnxn = sqlite3.connect('covid19')
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT min(id) FROM main.application_metriques LIMIT2 "
            "WHERE nomMod LIKE 'RESNET50'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('reresnet50/')


def annulerresnet50(request):
    cnxn = sqlite3.connect('covid19')
    cursor4 = cnxn.cursor()
    cursor4.execute(
        "SELECT  count(*)  FROM main.application_metriques "
        "WHERE nomMod LIKE 'RESNET50'")
    k = cursor4.fetchone()
    cnxn.commit()
    cursor4.close()
    if k == (2,):
        os.remove("model_non_traite/resnet50.H5")
        cnxn = sqlite3.connect('covid19')
        cursor3 = cnxn.cursor()
        cursor3.execute(
            "DELETE  FROM main.application_metriques WHERE id=(SELECT id FROM main.application_metriques LIMIT1 "
            "WHERE nomMod LIKE 'RESNET50'  ORDER BY id DESC)")
        cnxn.commit()
        cursor3.close()
        return render(request, 'backoffice/models.html')
    else:
        messages.add_message(request, SUCCESS, 'tu dois lancer le reapprentissage')
        return redirect('reresnet50/')


############ AFFICHER  LES METRIQUES #########################
def DecisionTree(request):
    metriqueDT = metriques.objects.all().filter(nomMod="DecisionTree")
    return render(request, 'backoffice/DecesionTree.html', context={'metriqueDT': metriqueDT})
def Dd(request):
    return render(request, 'backoffice/DecesionTree.html')

def SVM(request):
    return render(request, 'backoffice/SVM.html')


def CNN(request):
    return render(request, 'backoffice/CNN.html')


def Resnet50(request):
    return render(request, 'backoffice/RESNET50.html')


def Xgboost(request):
    metR = metrique.objects.all()
    return render(request, 'backoffice/metriqueRidge.html', context={'metR': metR})


from pydub import AudioSegment


######################## LES PREDICTIONS ##########################
def DecisionTreePred(request):
    try:
        if request.method == "GET":
            if request.GET.get('imported'):
                url = request.GET.get('imported')
            else:
                url = request.GET.get('import')
            x, fs = librosa.load(url)
            audio_segment = AudioSegment.from_file(url)
            feature_names = []
            feature_names += ["mfcc_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_secon_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            # feature_names.append('loudness')
            feature_names += ["mel_{0:d}".format(mfcc_i) for mfcc_i in range(1, 13 + 1)]
            arr = np.array(feature_names)
            hop_length = 1024
            mfccs = librosa.feature.mfcc(x, sr=fs, n_fft=1024, n_mfcc=46, dct_type=1, hop_length=hop_length)
            if mfccs.shape[1] > 2:
                if mfccs.shape[1] % 2 == 0:
                    s = mfccs.shape[1] - 1
                else:
                    s = mfccs.shape[1]
                der = librosa.feature.delta(mfccs, width=s)
                der1 = librosa.feature.delta(mfccs, width=s, order=2)
                mel = librosa.feature.melspectrogram(x, sr=fs, n_fft=1024, n_mels=13, hop_length=hop_length)
            Xnew = np.hstack((mfccs.T / audio_segment.dBFS, der.T))
            Xnew = np.hstack((Xnew, der1.T / audio_segment.dBFS))
            Xnew = np.hstack((Xnew, mel.T / audio_segment.dBFS))
            arr = np.vstack((arr, Xnew))
            df = pd.DataFrame(arr)
            df = df.rename(columns=df.iloc[0])
            df = df.iloc[1:, :]
            liste_columns = df.columns
            for i in liste_columns:
                df[i] = df[i].astype(float)
            print(df)
            loaded_Xgboost = pickle.load(open('model_utilise/Dtree.pkl', 'rb'))
            resultat = loaded_Xgboost.predict(df)
            print(resultat)
            if resultat.mean() > 0.3:
                messages.add_message(request, SUCCESS, 'POSITIF',extra_tags='signup')
            else:
                messages.add_message(request, SUCCESS, 'NEGATIF',extra_tags='signup')

            return render(request, 'backoffice/predDtree.html')
    except:
        print("erreur")
        return render(request, 'backoffice/predDtree.html')




def SVMPred(request):
    try:
        if request.method == "GET":
            if request.GET.get('imported'):
                url = request.GET.get('imported')
            else:
                url = request.GET.get('import')
            x, fs = librosa.load(url)
            audio_segment = AudioSegment.from_file(url)
            feature_names = []
            feature_names += ["mfcc_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_secon_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            # feature_names.append('loudness')
            feature_names += ["mel_{0:d}".format(mfcc_i) for mfcc_i in range(1, 13 + 1)]
            arr = np.array(feature_names)
            hop_length = 1024
            mfccs = librosa.feature.mfcc(x, sr=fs, n_fft=1024, n_mfcc=46, dct_type=1, hop_length=hop_length)
            if mfccs.shape[1] > 2:
                if mfccs.shape[1] % 2 == 0:
                    s = mfccs.shape[1] - 1
                else:
                    s = mfccs.shape[1]
                der = librosa.feature.delta(mfccs, width=s)
                der1 = librosa.feature.delta(mfccs, width=s, order=2)
                mel = librosa.feature.melspectrogram(x, sr=fs, n_fft=1024, n_mels=13, hop_length=hop_length)
            Xnew = np.hstack((mfccs.T / audio_segment.dBFS, der.T))
            Xnew = np.hstack((Xnew, der1.T / audio_segment.dBFS))
            Xnew = np.hstack((Xnew, mel.T / audio_segment.dBFS))
            arr = np.vstack((arr, Xnew))
            df = pd.DataFrame(arr)
            df = df.rename(columns=df.iloc[0])
            df = df.iloc[1:, :]
            liste_columns = df.columns
            for i in liste_columns:
                df[i] = df[i].astype(float)
            print(df)
            loaded_Xgboost = pickle.load(open('model_utilise/svm.pkl', 'rb'))
            resultat = loaded_Xgboost.predict(df)
            print(resultat)
            if resultat.mean() > 0.3:
                messages.add_message(request, SUCCESS, 'POSITIF')
            else:
                messages.add_message(request, SUCCESS, 'NEGATIF')

            return render(request, 'backoffice/predSVM.html')
    except:
        print("erreur")
        return render(request, 'backoffice/predSVM.html')



from tensorflow import keras

from keras.models import load_model


def CNNPred(request):
    try:
        if request.method == "GET":
            if request.GET.get('imported'):
                url = request.GET.get('imported')
            else:
                url = request.GET.get('import')
            x, fs = librosa.load(url)
            audio_segment = AudioSegment.from_file(url)
            feature_names = []
            feature_names += ["mfcc_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_secon_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            # feature_names.append('loudness')
            feature_names += ["mel_{0:d}".format(mfcc_i) for mfcc_i in range(1, 13 + 1)]
            arr = np.array(feature_names)
            hop_length = 1024
            mfccs = librosa.feature.mfcc(x, sr=fs, n_fft=1024, n_mfcc=46, dct_type=1, hop_length=hop_length)
            if mfccs.shape[1] > 2:
                if mfccs.shape[1] % 2 == 0:
                    s = mfccs.shape[1] - 1
                else:
                    s = mfccs.shape[1]
                der = librosa.feature.delta(mfccs, width=s)
                der1 = librosa.feature.delta(mfccs, width=s, order=2)
                mel = librosa.feature.melspectrogram(x, sr=fs, n_fft=1024, n_mels=13, hop_length=hop_length)
            Xnew = np.hstack((mfccs.T / audio_segment.dBFS, der.T / audio_segment.dBFS))
            Xnew = np.hstack((Xnew, der1.T / audio_segment.dBFS))
            Xnew = np.hstack((Xnew, mel.T / audio_segment.dBFS))
            arr = np.vstack((arr, Xnew))
            df = pd.DataFrame(arr)
            df = df.rename(columns=df.iloc[0])
            df = df.iloc[1:, :]
            liste_columns = df.columns
            for i in liste_columns:
                df[i] = df[i].astype(float)
            mfcc_tenseur1 = np.empty((1, 198, 151, 1), dtype=np.float64)
            mfcc_reshape = df.to_numpy()
            print(type(mfcc_reshape))
            print(mfcc_reshape.shape)
            x = mfcc_reshape[0, :]
            k = 0
            while k < 198:
                for i in range(df.shape[0]):
                    k = k + 1
                    if k < 198:
                        x = np.vstack((x, mfcc_reshape[i, :]))
            mfcc_reshape1 = x.reshape(198, 151, 1)
            print(x.shape)
            mfcc_tenseur1[0, :, :, :] = mfcc_reshape1
            model = load_model('model_utilise/cnn.h5')
            resultat = model.predict(mfcc_tenseur1)
            r = pd.DataFrame(resultat)
            print(r.iloc[0, 1])
            if r.iloc[0, 1] > 0.3:
                messages.add_message(request, SUCCESS, 'POSITIF')
            else:
                messages.add_message(request, SUCCESS, 'NEGATIF')

            return render(request, 'backoffice/predCNN.html')
    except:
        print("erreur")
        return render(request, 'backoffice/predCNN.html')



def Resnet50Pred(request):
    try:
        if request.method == "GET":
            if request.GET.get('imported'):
                url = request.GET.get('imported')
            else:
                url = request.GET.get('import')
            x, fs = librosa.load(url)
            audio_segment = AudioSegment.from_file(url)
            feature_names = []
            feature_names += ["mfcc_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_secon_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            # feature_names.append('loudness')
            feature_names += ["mel_{0:d}".format(mfcc_i) for mfcc_i in range(1, 13 + 1)]
            arr = np.array(feature_names)
            hop_length = 1024
            mfccs = librosa.feature.mfcc(x, sr=fs, n_fft=1024, n_mfcc=46, dct_type=1, hop_length=hop_length)
            if mfccs.shape[1] > 2:
                if mfccs.shape[1] % 2 == 0:
                    s = mfccs.shape[1] - 1
                else:
                    s = mfccs.shape[1]
                der = librosa.feature.delta(mfccs, width=s)
                der1 = librosa.feature.delta(mfccs, width=s, order=2)
                mel = librosa.feature.melspectrogram(x, sr=fs, n_fft=1024, n_mels=13, hop_length=hop_length)
            Xnew = np.hstack((mfccs.T / audio_segment.dBFS, der.T / audio_segment.dBFS))
            Xnew = np.hstack((Xnew, der1.T / audio_segment.dBFS))
            Xnew = np.hstack((Xnew, mel.T / audio_segment.dBFS))
            arr = np.vstack((arr, Xnew))
            df = pd.DataFrame(arr)
            df = df.rename(columns=df.iloc[0])
            df = df.iloc[1:, :]
            liste_columns = df.columns
            for i in liste_columns:
                df[i] = df[i].astype(float)
            mfcc_tenseur1 = np.empty((1, 198, 151, 3), dtype=np.float64)
            mfcc_reshape = df.to_numpy()
            print(type(mfcc_reshape))
            print(mfcc_reshape.shape)
            x = mfcc_reshape[0, :]
            k = 0
            while k < 198:
                for i in range(df.shape[0]):
                    k = k + 1
                    if k < 198:
                        x = np.vstack((x, mfcc_reshape[i, :]))
            mfcc_reshape1 = x.reshape(198, 151, 1)
            print(x.shape)
            rgb_batch = np.repeat(mfcc_reshape1, 3, -1)
            mfcc_tenseur1[0, :, :, :] = rgb_batch
            model = load_model('model_utilise/resnet50.h5')
            resultat = model.predict(mfcc_tenseur1)
            r = pd.DataFrame(resultat)
            print(r.iloc[0, 1])
            if r.iloc[0, 1] > 0.3:
                messages.add_message(request, SUCCESS, 'POSITIF')
            else:
                messages.add_message(request, SUCCESS, 'NEGATIF')

            return render(request, 'backoffice/predResnet50.html')
    except:
        print("erreur")
        return render(request, 'backoffice/predResnet50.html')



import librosa


def XgboostPred(request):
    try:
        if request.method == "GET":
            if request.GET.get('imported'):
                url = request.GET.get('imported')
            else:
                url = request.GET.get('import')
            x, fs = librosa.load(url)
            audio_segment = AudioSegment.from_file(url)
            feature_names = []
            feature_names += ["mfcc_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            feature_names += ["der_secon_{0:d}".format(mfcc_i) for mfcc_i in range(1, 46 + 1)]
            # feature_names.append('loudness')
            feature_names += ["mel_{0:d}".format(mfcc_i) for mfcc_i in range(1, 13 + 1)]
            arr = np.array(feature_names)
            hop_length = 1024
            mfccs = librosa.feature.mfcc(x, sr=fs, n_fft=1024, n_mfcc=46, dct_type=1, hop_length=hop_length)
            if mfccs.shape[1] > 2:
                if mfccs.shape[1] % 2 == 0:
                    s = mfccs.shape[1] - 1
                else:
                    s = mfccs.shape[1]
                der = librosa.feature.delta(mfccs, width=s)
                der1 = librosa.feature.delta(mfccs, width=s, order=2)
                mel = librosa.feature.melspectrogram(x, sr=fs, n_fft=1024, n_mels=13, hop_length=hop_length)
            Xnew = np.hstack((mfccs.T / audio_segment.dBFS, der.T))
            Xnew = np.hstack((Xnew, der1.T / audio_segment.dBFS))
            Xnew = np.hstack((Xnew, mel.T / audio_segment.dBFS))
            arr = np.vstack((arr, Xnew))
            df = pd.DataFrame(arr)
            df = df.rename(columns=df.iloc[0])
            df = df.iloc[1:, :]
            liste_columns = df.columns
            for i in liste_columns:
                df[i] = df[i].astype(float)
            print(df)
            loaded_Xgboost = pickle.load(open('model_utilise/xgb.pkl', 'rb'))
            # model_xgb_2 = xgb.Booster()
            # model_xgb_2.load_model("model.json")
            resultat = loaded_Xgboost.predict(df)
            print(resultat)
            if resultat.mean() > 0.3:
                messages.add_message(request, SUCCESS, 'POSITIF')
            else:
                messages.add_message(request, SUCCESS, 'NEGATIF')

            return render(request, 'backoffice/predXgboost.html')
    except:
        print("erreur")
        return render(request, 'backoffice/predXgboost.html')




####################################### LES PAGE DE PRED######################################################################

def predDtree(request):
    return render(request, 'backoffice/predDtree.html')


def accueil(request):
    return render(request, 'backoffice/accueil.html')

def models(request):
    sim = dataSimulee.objects.all()[:10]
    return render(request, 'backoffice/models.html', context={"sim": sim})

def lesMétriques(request):
    lesMet = metriques.objects.all()
    return render(request, 'backoffice/tableau.html', context={'lesMet': lesMet})
