import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Class to load trained DistilBERT model and predict text labels
class ToxicTextModel:
    def __init__(self, model_path="models/saved_harassing_model"):
        """
        Load the trained DistilBERT model and tokenizer.
        """
        # Choose device: GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model from saved folder
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)  # hint: tokenizer → convert text to tokens
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)  # hint: load trained model
        self.model.to(self.device)  # move model to device
        self.model.eval()  # evaluation mode → no training

    # Function to predict labels for a list of texts
    def predict(self, texts):
        """
        Args:
            texts (list of str): Input texts to classify.
        Returns:
            list of str: 'fake' for harassing, 'real' for non-harassing.
        """
        # Tokenize texts and convert to tensors on device
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)  # hint: texts → token tensors

        # Predict without gradients
        with torch.no_grad():
            outputs = self.model(**encodings)  # forward pass
            preds = torch.argmax(outputs.logits, dim=-1)  # get predicted class index
        
        # Convert integers to labels: 0 → fake, 1 → real
        return ["fake" if p == 0 else "real" for p in preds]  # hint: map predicted indices to labels


# Example usage if file is run directly
if __name__ == "__main__":
    model = ToxicTextModel("models/saved_harassing_model")
    sample_texts = [
        "You're terrible!", 
        "Well done on your presentation", 
        "Stop annoying everyone"
    ]
    predictions = model.predict(sample_texts)
    for text, label in zip(sample_texts, predictions):
        print(f"Text: {text} -> Label: {label}")  # hint: display results
