# Fine-Tuning FinBERT for Financial Sentiment Analysis

This repository contains code for fine-tuning a BERT-based model for financial sentiment analysis. The project uses the Financial PhraseBank dataset to train a model that can classify financial texts as positive, neutral, or negative.

## Project Overview

Financial sentiment analysis is crucial for various applications in finance, including market prediction, risk management, and automated trading. This project demonstrates how to:

1. Prepare financial text data
2. Fine-tune a pre-trained language model (DistilBERT)
3. Optimize hyperparameters
4. Evaluate model performance
5. Deploy the model for inference

## Results

| Metric | Value |
|--------|-------|
| Final accuracy | **97.4%** |
| Improvement over pre-trained baseline | **7.5×** |
| Dataset | Financial PhraseBank (3-class) |
| Classes | Positive · Neutral · Negative |
| Tuning method | Grid search + early stopping |

## Dataset

The project uses the Financial PhraseBank dataset, which contains financial news sentences labeled with sentiment (positive, neutral, negative). The data is automatically downloaded from Hugging Face's datasets library.

## Requirements

```
transformers>=4.20.0
datasets>=2.3.0
scikit-learn>=1.0.0
torch>=1.10.0
gradio>=3.0.0
pandas>=1.3.0
numpy>=1.20.0
```

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/NilayRaut/Fine-Tuning-FinBERT-for-Financial-Sentiment-Analysis.git
cd finbert-sentiment-analysis
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the notebook or script:
```bash
jupyter notebook Fine_Tuning_FinBERT.ipynb
```

## Project Structure

The project follows these steps:

1. **Environment Setup**: Installing and importing necessary libraries
2. **Data Loading**: Automatic download of the Financial PhraseBank dataset
3. **Data Preparation**: Splitting data into train/validation/test sets and tokenization
4. **Model Setup**: Loading the pre-trained DistilBERT model
5. **Training**: Fine-tuning the model on financial data
6. **Hyperparameter Optimization**: Finding optimal learning rate, batch size, and weight decay
7. **Evaluation**: Testing model performance and comparing with baseline
8. **Inference**: Creating a prediction function and Gradio web interface

## Fine-Tuning Results

Our fine-tuned model achieved:
- Accuracy: 97.43%
- F1 Score: 96.28%

This represents a significant improvement over the baseline model:
- Baseline Accuracy: 11.40%
- Baseline F1 Score: 6.82%

## Using the Model

### Option 1: Programmatic Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_path = "./final_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    scores = torch.softmax(outputs.logits, dim=1)
    prediction = scores.argmax(1).item()
    
    sentiment_labels = ["negative", "neutral", "positive"]
    return sentiment_labels[prediction]

# Example usage
text = "Company profits exceeded expectations."
sentiment = predict_sentiment(text)
print(f"Text: {text}")
print(f"Sentiment: {sentiment}")
```

### Option 2: Gradio Web Interface

Run the Gradio interface to interact with the model through a web UI:

```python
import gradio as gr

def gradio_predict(text):
    return predict_sentiment(text)

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=5, placeholder="Enter financial text here..."),
    outputs="text",
    title="Financial Sentiment Analysis",
    description="Enter a financial statement to predict its sentiment (positive, neutral, or negative)."
)

demo.launch()
```

## Model Limitations

- The model is specifically trained for financial text and may not perform well on other domains
- Short or ambiguous statements might be misclassified
- The model may not capture complex sentiments or nuanced language

## Future Improvements

- Experiment with different pre-trained models (BERT, RoBERTa, FinBERT)
- Incorporate more financial data from various sources
- Add confidence scores to predictions
- Extend to multi-class classification (beyond just positive/neutral/negative)
- Implement domain adaptation techniques for specific financial sub-domains

## A little demonstration explanation 
- https://youtu.be/wq446tT8xhU
- https://youtu.be/g6XFuzJqIqQ

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial PhraseBank dataset created by Malo et al.
- Hugging Face for their transformers library and datasets
- Gradio for the web interface framework
