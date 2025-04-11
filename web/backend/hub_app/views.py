from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import ModelWeight, Dataset
from .serializers import ModelWeightSerializer, DatasetSerializer

class ModelWeightViewSet(viewsets.ModelViewSet):
    queryset = ModelWeight.objects.all()
    serializer_class = ModelWeightSerializer
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(uploader=self.request.user)

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(uploader=self.request.user)
