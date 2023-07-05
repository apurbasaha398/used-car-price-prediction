from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_preprocessing = DataPreprocessing()
        self.model_trainer = ModelTrainer()
        
    def initiate_training_pipeline(self):
        train_path, test_path = self.data_ingestion.initiate_data_injection()
        preprocessor_path = self.data_preprocessing.get_data_preprocessing_object()
        self.model_trainer.initiate_model_training(train_path, test_path, preprocessor_path)
        
if __name__ == '__main__':
    training_pipeline = TrainingPipeline()
    training_pipeline.initiate_training_pipeline()