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
import calendar


auth = tweepy.OAuthHandler(st.secrets.tw_credentials.consumer_key, st.secrets.tw_credentials.consumer_secret)
auth.set_access_token(st.secrets.tw_credentials.access_token, st.secrets.tw_credentials.access_token_secret)
api = tweepy.API(auth)

st.title("Social Listening")

st.write(""" 
Este tablero está diseñado para poder escuchar a los usuarios de Twitter,
entender las tendencias que están al rededor de la cuenta de @justo_mx y poder rescatar
la información más relevante de los usuarios.
""")

word_search = st.text_input("Introduce una palabra a clave*: ", value="@justo_mx")
st.write("""
**Por default la palabra clave es @justo_mx, pero se pueden trackear otras cuentas/términos.""")


twitter_users = []
tweet_time = []
tweet_string = []
twitter_followers = []

for tweet in tweepy.Cursor(api.search, q=word_search, count= 1000).items(1000):
    if (not tweet.retweeted) and ('RT @' not in tweet.text):
        if tweet.lang == 'es':
            twitter_users.append(tweet.user.name)
            tweet_time.append(tweet.created_at)
            tweet_string.append(tweet.text)
            twitter_followers.append(tweet.user.followers_count)

df = pd.DataFrame({'name':twitter_users, 'time':tweet_time, 'tweet':tweet_string })

min_month_time = df['time'].min().month
min_day_time = df['time'].min().day
max_month_time = df['time'].max().month
max_day_time = df['time'].max().day

min_date = str(min_day_time)+"-"+str(calendar.month_abbr[min_month_time])
max_date = str(max_day_time)+"-" +str(calendar.month_abbr[max_month_time])

st.write("")
st.write("## Palabras más frecuentes")
st.write("")

col1, col2, col3 = st.columns([3, 1, 1])
col1.write("Palabras más frecuentes con el término: " + word_search)
col2.metric("Fecha inicio", value=min_date)
col3.metric("Fecha fin", value=max_date)

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

omitir_palabras = word_search

df_3 = [w for w in df_3 if w not in omitir_palabras]

freq_words = nltk.FreqDist(df_3)

num_palabras = st.slider('¿Cuántas palabras quieres ver?',min_value=5,max_value=20,value=15)

##### Función plotear etiquetas ######

def add_value_labels(ax, spacing=2):
        """Add labels to the end of each bar in a bar chart.

        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """

        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.0f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.


fig4, ax4 = plt.subplots()
freq_words.most_common(num_palabras)
fw_data_tweets = pd.DataFrame(freq_words.items(), columns=['word', 'frequency']).reset_index().sort_values(by='frequency', ascending=False)
w_plot_tweets = fw_data_tweets.head(num_palabras)
ax4.bar(w_plot_tweets['word'],w_plot_tweets['frequency'])
plt.xticks(rotation=90)
add_value_labels(ax4)
plt.show()
st.pyplot(fig4)

st.write("## Nube de palabras")

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


def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += ", ".join([feature_names[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]])
            return(message)

st.write(print_top_words(nmf, tfidf_vectorizer.get_feature_names(), n_top_words=3))

st.write("## Búsqueda inversa")

busqueda_inv = st.text_input("Introduce una palabra a buscar en los tweets: ")

df_busqueda_2 = df[df.apply(lambda row: row.astype(str).str.contains(busqueda_inv).any(), axis=1)]['tweet'].to_list()

st.write(df_busqueda_2)
