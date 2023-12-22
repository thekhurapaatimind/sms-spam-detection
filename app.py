import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('model/vectorizer.pkl','rb'))
model = pickle.load(open('model/model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown("<h1 style='text-align: center; color: red;'>Spam</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: green;'>Not Spam</h1>", unsafe_allow_html=True)

st.text("You can try with these messages: ")
# suggest 10 messages of 1-3 sentences both spam and not spam messages
st.write("1. Hi, hope you are doing well")
st.write("2. You have won a prize worth $1000. Please call on 1234567890")
st.write("3. Love you, see you soon")
st.write("4. England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try: WALES, SCOTLAND 4txt/Ì¼1.20 POBOXox36504W45WQ 16+")
st.write("5. As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune")
st.write("6. WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only")



