import os
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

class Metrics():
    def __init__(self):
        self.metrics_data = []
        self.results_dir = 'results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def evaluate(self, X_test, y_test, fold_num):
        # Load trained model from models directory
        model = load_model('models/sentiment_analysis_rnn.h5')

        # Load tokenizer for inference
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        X_test = tokenizer.texts_to_sequences(X_test)
        X_test = pad_sequences(X_test, maxlen=100, padding='post')
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Loss: {loss}, Accuracy: {accuracy}')

        # Make predictions
        y_pred = (model.predict(X_test) > 0.5).astype('int32')
        metrics_report = classification_report(y_test, y_pred)
        self.store_metrics(fold_num, metrics_report, loss, accuracy)
    
    def store_metrics(self, fold_num, metrics_report, loss, accuracy):
        metrics = {'fold_num': fold_num, 'classification_report': metrics_report, 'loss': loss, 'accuracy': accuracy}
        self.metrics_data.append(metrics)
    
    def save_metrics(self):
        filepath = 'results/model_metrics.txt'
        with open(filepath, 'w') as f:
            f.write("Model Metrics\n")
            f.write("=============\n\n")
            f.write("Fold Number | Loss | Accuracy\n")
            f.write("------------|------|---------\n")
            for metrics in self.metrics_data:
                f.write(f"{metrics['fold_num']} | {metrics['loss']} | {metrics['accuracy']}\n")
                f.write(f"{metrics['classification_report']}\n\n")
    
        if self.metrics_data:
            avg_loss = sum([metrics['loss'] for metrics in self.metrics_data]) / len(self.metrics_data)
            avg_accuracy = sum([metrics['accuracy'] for metrics in self.metrics_data]) / len(self.metrics_data)
            print(f'Average Loss: {avg_loss}, Average Accuracy: {avg_accuracy}')

            with open(filepath, 'a') as f:
                f.write(f"Average Loss: {avg_loss}, Average Accuracy: {avg_accuracy}\n")
        else:
            print('No metrics to save.')
        
        print(f'Metrics saved successfully in {filepath}.')
