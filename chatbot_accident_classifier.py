

# **Reqd Functions**
"""

#NLP Basic Prep
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def nlp_text_prep(text):
    # Lowercase conversion
    text = text.lower()
    # Punctuation, Special Charcters removal (optional)
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Adjust for desired punctuation handling
    #Stopwords and numeric characters removal
    #stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if not word.isdigit()]
    return ' '.join(words)

#RoBERTa Tokenizer
from transformers import RobertaTokenizer
tokenizer_r = RobertaTokenizer.from_pretrained("roberta-base")
def roberta_text_prep(text):
  # Max length of 256 ensures a larger yet more standard acceptance of text input size
  tokens = tokenizer_r.encode_plus(text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length')
  input_ids = tokens['input_ids']
  attention_mask = tokens['attention_mask']
  return input_ids, attention_mask

def predict_accident_roberta(text):
  with torch.no_grad():
      text = nlp_text_prep(text)
      input_ids, attention_mask = roberta_text_prep(text)
      logits = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask])).logits
      predicted_label = torch.argmax(logits, dim=1).item()
  return predicted_label

"""#### **Loading Model**"""

'''
#Libraries required:
!pip install transformers
!pip install torch
'''

import torch
from transformers import RobertaForSequenceClassification


#Function to load both model state dictionary and config
def load_model(model_name, num_classes):
  model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
  model.load_state_dict(torch.load(f"/content/drive/MyDrive/Capstone/Data Files/state_dict.pt"))
  return model

# Load the saved model
loaded_model = load_model("roberta-base", 5)

"""### **Integrating with Chatbot**"""

import streamlit as st
from transformers import RobertaTokenizer

# Load the saved model and tokenizer (assuming they're saved in 'saved_model')
model = loaded_model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")  # Adjust path if saved

def main():
  st.title("Accident Severity Prediction")
  user_input = st.text_input("Enter Accident Description:")

  if st.button("Predict"):
    prediction = predict_accident_roberta(user_input)

    # Display results
    class_names = [1,2,3,4,5]  # Replace with your actual class names
    st.write("Predicted Class:", class_names[prediction])

if __name__ == "__main__":
  main()
