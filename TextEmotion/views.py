'''
Created on 06-Sep-2018

@author: Anu
'''

from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import permission_classes
from rest_framework import permissions
from .Emotion_Prediction import prediction

@permission_classes((permissions.AllowAny,))
class Emotion(viewsets.ViewSet):
    def create(self, request):
        question = request.data
        result = {}
        result['Emotion'] = prediction(question["text"])
        return Response(result)
