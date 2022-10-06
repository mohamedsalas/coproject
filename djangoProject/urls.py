# import importation as importation
from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from application.views import simulation, accueil, \
    models, historiqueScenario, DecisionTree, SVM, CNN, Resnet50, Xgboost, XgboostPred, \
    DecisionTreePred, SVMPred, CNNPred, Resnet50Pred, predDtree, reTree, reboost, resvm, recnn, reresnet50, \
    lesMétriques, remplacerDTREE, remplacerresnet50, remplacercnn, remplacersvm, remplacerxgb, annulerDtree, annulercnn, \
    annulerresnet50, annulersvm, annulerxgb, start_stream, Dd

urlpatterns = [
                  ...
              ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns = [
                  path('', accueil, name='accueil'),
                  path('sim/', simulation, name='simulation'),
                  path('Xgboost', Xgboost, name='Xgboost'),
                  path('DecisionTree', DecisionTree, name='DecisionTree'),
                  path('SVM', SVM, name='SVM'),
                  path('CNN', CNN, name='CNN'),
                  path('Resnet50', Resnet50, name='Resnet50'),
                  path('XgboostPred', XgboostPred, name='XgboostPred'),
                  path('DecisionTreePred', DecisionTreePred, name='DecisionTreePred'),
                  path('SVMPred', SVMPred, name='SVMPred'),
                  path('CNNPred', CNNPred, name='CNNPred'),
                  path('predDtree', predDtree, name='predDtree'),
                  path('Resnet50Pred', Resnet50Pred, name='Resnet50Pred'),
                  path('admin/', admin.site.urls),
                  path('models/', models, name='models'),
                  path('reTree/', reTree, name='reTree'),
                  path('reboost/', reboost, name='reboost'),
                  path('resvm/', resvm, name='resvm'),
                  path('recnn/', recnn, name='recnn'),
                  path('reresnet50/', reresnet50, name='reresnet50'),
                  path('lesMétriques/', lesMétriques, name='lesMétriques'),
                  path('remplacerDTREE/', remplacerDTREE, name='remplacerDTREE'),
                  path('remplacerresnet50/', remplacerresnet50, name='remplacerresnet50'),
                  path('remplacercnn/', remplacercnn, name='remplacercnn'),
                  path('remplacersvm/', remplacersvm, name='remplacersvm'),
                  path('remplacerxgb/', remplacerxgb, name='remplacerxgb'),
                  path('annulerDtree/', annulerDtree, name='annulerDtree'),
                  path('annulersvm/', annulersvm, name='annulersvm'),
                  path('annulerxgb/', annulerxgb, name='annulerxgb'),
                  path('annulerresnet50/', annulerresnet50, name='annulerresnet50'),
                  path('annulercnn/', annulercnn, name='annulercnn'),
                  path("start_stream/", start_stream, name="start_stream"),
                  path("Resnet50Pred", Resnet50Pred, name="Resnet50Pred"),
                  path("annulerDtree/reTree/", reTree, name="reTree"),
                  path("remplacerDTREE/reTree/", reTree, name="reTree"),
                  path("annulerxgb/reboost/", reboost, name="reboost"),
                  path("remplacerxgb/reboost/", reboost, name="reboost"),
                  path("annulersvm/resvm/", resvm, name="resvm"),
                  path("remplacersvm/resvm/", resvm, name="resvm"),
                  path("annulercnn/recnn/", recnn, name="recnn"),
                  path("remplacercnn/recnn/", recnn, name="recnn"),
                  path("annulerresnet50/reresnet50/", reresnet50, name="reresnet50"),
                  path("remplacerresnet50/reresnet50/", reresnet50, name="reresnet50"),
              ] + static(settings.STATIC_URL, document_root=settings.MEDIA_ROOT)
