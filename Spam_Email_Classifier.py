import pickle
import string
import streamlit as st
import nltk
from nltk.corpus import stopwords
import time
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
tfidf=pickle.load(open("countvectorizer.pkl","rb"))
model=pickle.load(open("model.pkl","rb"))

st.title("EMAIL Spam Classifier")
st.image("C:\Program Files\PycharmProjects\Python_GUI_Projects\VRSN_CompanyBrandedEmail_BlogImage8_201712-670x446.png", width=100)
input= st.text_area("Enter the Email", height=200)
st.markdown(
    """
    <style>
    .spam {
        font-size:24px !important;
        color: #ff0000;
    }
    .not-spam {
        font-size:24px !important;
        color: #00ff00;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.spinner("Processing...")as spinner:
   time.sleep(2)
#input=st.text_input("Enter the Email")
if st.button("Predict"):
   #1.Preprocess
   ps=PorterStemmer()
   def transform_text(text):
      text=text.lower() #Lower Case
      text=nltk.word_tokenize(text) #Tokenization
      y=[] #Removing Special Characters
      for i in text:
         if i.isalnum():
            y.append(i)

      text=y[:]
      y.clear()
      for i in text: #Removing stop words and punctuation
          if i not in stopwords.words("english") and i not in string.punctuation:
             y.append(i)

      text=y[:]
      y.clear()
      for i in text:
         y.append(ps.stem(i)) #Stemming


      return " ".join(y) #Return in the form of String

   input_email=transform_text(input)
#2.Vectorize
   vector_input=tfidf.transform([input_email])
   vector_input_dense=vector_input.toarray() #tfdif.transform return a spasrse matrix by default
#3.Predict
   result=model.predict(vector_input_dense)

#4.Show
   if result==1:
      st.header("Spam")
   else:
    st.header("Not Spam")



