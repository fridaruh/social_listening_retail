import pandas as pd
import streamlit as st
import tweepy
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
import time


auth = tweepy.OAuthHandler(st.secrets.tw_credentials.consumer_key, st.secrets.tw_credentials.consumer_secret)
auth.set_access_token(st.secrets.tw_credentials.access_token, st.secrets.tw_credentials.access_token_secret)
api = tweepy.API(auth)

st.title("Social Listening")

word_search = st.text_input("Introduce una palabra a clave: ")

time.sleep(5)

twitter_users = []
tweet_time = []
tweet_string = []

for tweet in tweepy.Cursor(api.search, q=word_search, count= 1000).items(1000):
    if (not tweet.retweeted) and ('RT @' not in tweet.text):
        if tweet.lang == 'es':
            twitter_users.append(tweet.user.name)
            tweet_time.append(tweet.created_at)
            tweet_string.append(tweet.text)

df = pd.DataFrame({'name':twitter_users, 'time':tweet_time, 'tweet':tweet_string })

min_time = df['time'].min()
max_time = df['time'].max()

data = df['tweet'].to_list()

pattern = r'''(?x)                  # Flag para iniciar el modo verbose
              (?:[A-Z]\.)+          # Hace match con abreviaciones como U.S.A.
              | \w+(?:-\w+)*        # Hace match con palabras que pueden tener un guión interno
              | \$?\d+(?:\.\d+)?%?  # Hace match con dinero o porcentajes como $15.5 o 100%
              | \.\.\.              # Hace match con puntos suspensivos
              | [][.,;"'?():-_`]    # Hace match con signos de puntuación
'''

texto = []

for x in range(0, len(data)):
    token_1 = data[x].lower()
    token_2 = nltk.regexp_tokenize(token_1, pattern)
    texto.append(token_2)

flatten = [w for l in texto for w in l]

puntuacion = list(string.punctuation)

puntuacion.append('https')
puntuacion.append('co')
puntuacion.append('t')

stop_words_n = nltk.corpus.stopwords.words('spanish')

df_2 = [w for w in flatten if w not in stop_words_n]

df_3 = [w for w in df_2 if w not in puntuacion]

freq_words = nltk.FreqDist(df_3)

num_palabras = st.slider('¿Cuántas palabras quieres ver?',min_value=5,max_value=20,value=15)

fig4, ax4 = plt.subplots()
freq_words.most_common(num_palabras)
fw_data_tweets = pd.DataFrame(freq_words.items(), columns=['word', 'frequency']).reset_index().sort_values(by='frequency', ascending=False)
w_plot_tweets = fw_data_tweets.head(num_palabras)
ax4.bar(w_plot_tweets['word'],w_plot_tweets['frequency'])
plt.xticks(rotation=90)
plt.show()
st.pyplot(fig4)

fig2, ax = plt.subplots()
wordcloud_tweets = WordCloud(background_color='white', collocations=False, max_words=30).fit_words(freq_words)
plt.imshow(wordcloud_tweets, interpolation='bilinear')
plt.axis('off')
plt.show()
st.pyplot(fig2)

c_vec = CountVectorizer(ngram_range=(2,3))
ngrams =c_vec.fit_transform(df_3)

count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,3))

nmf = NMF(n_components=3)

pipe = make_pipeline(tfidf_vectorizer, nmf)

pipe.fit(df_3)

#Funcion que imprime los temas

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        return(message)


st.write(print_top_words(nmf, tfidf_vectorizer.get_feature_names(), n_top_words=3))


busqueda_inv = st.text_input("Introduce una palabra a buscar en los tweets: ")

def busqueda_inversa(palabra):
    df_busqueda = df[df.apply(lambda row: row.astype(str).str.contains(palabra).any(), axis=1)]
    for i in range(0, len(df_busqueda)):
        print(df_busqueda['tweet'].iloc[i])
        print("----")
        

result_busq = busqueda_inversa(busqueda_inv)

df_busqueda_2 = df[df.apply(lambda row: row.astype(str).str.contains(busqueda_inv).any(), axis=1)]['tweet'].to_list()

st.write(df_busqueda_2)
