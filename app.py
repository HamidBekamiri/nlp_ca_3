from numpy.core.arrayprint import _make_options_dict
import streamlit as st
import pickle
import pandas as pd
import scipy.spatial
import numpy as np
import os, json
import glob
import re
import torch
import pandas as pd
# from sentence_transformers import SentenceTransformer, util
# from tokenizers import Tokenizer
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
import pandas as pd
import torch
import random
import itertools
import pickle
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
# %matplotlib inline

start = time.time()

st.set_page_config(layout="wide", page_title="Semantic Search for defining Indicators!", page_icon="🐞")

st.header("🐞 3. Keyword/Concept/Term analysis!")
st.subheader('AI-Growth-Lab AAU')

@st.cache
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv().encode('utf-8')

def get_top_n_similar_patents_df(new_claim, claim_embeddings):
    search_hits_list = []
    search_hits = util.semantic_search(new_claim, claim_embeddings, 10000, 5000000, 20)
    # save similar patents info
    top_doc_id = []
    top_similarity_scores = []
    for item in range(len(search_hits[0])):
        top_doc_id.append(search_hits[0][item]['corpus_id'])
        top_similarity_scores.append(search_hits[0][1]['score'])
        
    top_n_similar_patents_df = pd.DataFrame({
        'top_doc_id': top_doc_id,
        'cosine_similarity': top_similarity_scores
    })
    return top_n_similar_patents_df

class Interview_report:
    def __init__ (self, df):
        self.organization = df['Organization']
        self.expert = df['Expert_id']
        self.answer = df['Answer']
        
    def value_count(self):
        organization_count = self.organization.value_counts()
        expert_count = self.expert.value_counts()
        df_records = len(df)
        return df_records, organization_count, expert_count
    
    def clean(self, txt):
        txt = txt.str.replace("()", "")
        txt = txt.str.replace('(<a).*(>).*()', '')
        txt = txt.str.replace('(&amp)', '')
        txt = txt.str.replace('(&gt)', '')
        txt = txt.str.replace('(&lt)', '')
        txt = txt.str.replace('(\xa0)', ' ')  
        return txt 
    
    def preprocessing(self):
        answer_clean = self.clean(self.answer)
        # Converting to lower case
        answer_clean = answer_clean.apply(lambda x: " ".join(x.lower() for x in x.split()))
        # Removing the Punctuation
        answer_clean = answer_clean.str.replace('[^\w\s]', '')
        # Removing Stopwords
        import nltk
        from nltk.corpus import stopwords
        stop = stopwords.words('english')
        answer_clean = answer_clean.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        answer_clean.head()
        # Remove the Rare Words
        freq = pd.Series(' '.join(answer_clean).split()).value_counts()
        less_freq = list(freq[freq ==1].index)
        answer_clean = answer_clean.apply(lambda x: " ".join(x for x in x.split() if x not in less_freq))
        # Spelling Correction
        from textblob import TextBlob, Word, Blobber
        answer_clean.apply(lambda x: str(TextBlob(x).correct()))
        answer_clean.head()
        import nltk
        nltk.download('wordnet')
        answer_clean = answer_clean.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        interview_len = answer_clean.astype(str).apply(len)
        word_count = answer_clean.apply(lambda x: len(str(x).split()))
        polarity = answer_clean.map(lambda text: TextBlob(text).sentiment.polarity)
        df_answer_clean = answer_clean.to_frame('answer_clean')
        df_interview_len = interview_len.to_frame('interview_len')
        df_word_count = word_count.to_frame('word_count')
        df_polarity = polarity.to_frame('polarity')
        df_all = pd.concat([self.organization, self.expert, df_answer_clean, df_interview_len, df_word_count, df_polarity], join='outer', axis=1,)
        return df_all, freq, less_freq
 
    def get_top_n_words(self, corpus, n=None):
        from sklearn.feature_extraction.text import CountVectorizer
        vec=CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        common_words = words_freq[:n]
        df1 = pd.DataFrame(common_words, columns = ['Answer_clean_1', 'Count'])
        df1.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Top Words", ylabel = "Count", title = "Bar Chart of Top Words Frequency")
        # Frequency Charts
        return words_freq[:n]

    def get_top_n_bigram(self, corpus, n=None):
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(ngram_range=(2,2)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        common_words2 = words_freq[:n]
        df2 = pd.DataFrame(common_words2, columns=['Answer_clean_1', "Count"])
        df2.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Bigram Words", ylabel = "Count", title = "Bar chart of Bigrams Frequency")
        return words_freq[:n]

    def get_top_n_trigram(self, corpus, n=None):
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        common_words3 = words_freq[:n]
        df3 = pd.DataFrame(common_words3, columns = ['Answer_clean_1' , 'Count'])
        df3.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Trigrams Words", ylabel = "Count", title = "Bar chart of Trigrams Frequency")
        return words_freq[:n]

    def visulization_report(self):
        df_all, freq, less_freq = Interview_report(df).preprocessing()
        df_all[["interview_len", "word_count", "polarity"]].hist(bins=20, figsize=(15, 10))
        # Polarity vs expert
        from matplotlib import pyplot as plt
        import seaborn as sns
        plt.figure(figsize = (10, 8))
        sns.set_style('whitegrid')
        sns.set(font_scale = 1.5)
        sns.boxplot(x = 'Expert_id', y = 'polarity', data = df_all)
        plt.xlabel("Expert_id")
        plt.ylabel("Polatiry")
        plt.title("Expert_id vs Polarity")
        # save the figure
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_1.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure(figsize = (10, 8))
        sns.set_style('whitegrid')
        sns.set(font_scale = 1.5)
        sns.boxplot(x = 'Organization', y = 'polarity', data = df_all)
        plt.xlabel("Organization")
        plt.ylabel("Polatiry")
        plt.title("Organization vs Polarity")
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_2.png', dpi=300, bbox_inches='tight')
        plt.show()
        mean_pol = df_all.groupby('Expert_id')['polarity'].agg([np.mean])
        mean_pol.columns = ['mean_polarity']
        fig, ax = plt.subplots(figsize=(11, 6))
        plt.bar(mean_pol.index, mean_pol.mean_polarity, width=0.3)
        #plt.gca().set_xticklabels(mean_pol.index, fontdict={'size': 14})
        mean_pol = df_all.groupby('Expert_id')['polarity'].agg([np.mean])
        mean_pol.columns = ['mean_polarity']
        fig, ax = plt.subplots(figsize=(11, 6))
        for i in ax.patches:
            ax.text(i.get_x(), i.get_height()+0.01, str("{:.2f}".format(i.get_height())))
        plt.title("Polarity of Expert_id", fontsize=22)
        plt.ylabel("Polarity", fontsize=16)
        plt.xlabel("Expert_id", fontsize=16)
        plt.ylim(0, 0.35)
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_3.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure(figsize=(11, 6))
        sns.countplot(x='Expert_id', data=df_all)
        plt.xlabel("Expert_id")
        plt.title("Number of questions data of each Expert_id")
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_4.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure(figsize=(11, 6))
        sns.countplot(x='Organization', data=df_all)
        plt.xlabel("Organization")
        plt.title("Number of questions data of each Expert_id")
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_5.png', dpi=300, bbox_inches='tight')
        plt.show()
        # Length of the interview vs the Rating
        plt.figure(figsize=(11, 6))
        sns.pointplot(x = "Expert_id", y = "interview_len", data = df_all)
        plt.xlabel("Expert_id")
        plt.ylabel("Expert_id Length")
        plt.title("Product Expert_id vs Interview Length")
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_6.png', dpi=300, bbox_inches='tight')
        plt.show()
        # Length of the Interview vs the Expert_id
        plt.figure(figsize=(11, 6))
        sns.pointplot(x = "Organization", y = "interview_len", data = df_all)
        plt.xlabel("Organization")
        plt.ylabel("Organization Length")
        plt.title("Product Organization vs Interview Length")
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_7.png', dpi=300, bbox_inches='tight')
        plt.show()
        # Top 5 products based on the Polarity
        product_pol = df_all.groupby('Expert_id')['polarity'].agg([np.mean])
        product_pol.columns = ['polarity']
        product_pol = product_pol.sort_values('polarity', ascending=False)
        product_pol = product_pol.head()
        product_pol
        # WordCloud
        # conda install -c conda-forge wordcloud
        text = " ".join(interview for interview in df_all.answer_clean)
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        stopwords = set(STOPWORDS)
        stopwords = stopwords.union(["ha", "thi", "now", "onli", "im", "becaus", "wa", "will", "even", "go", "realli", "didnt", "abl"])
        wordcl = WordCloud(stopwords = stopwords, background_color='white', max_font_size = 50, max_words = 5000).generate(text)
        plt.figure(figsize=(14, 12))
        plt.imshow(wordcl, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_8.png', dpi=300, bbox_inches='tight')
        plt.show()
        # Frequency Charts
        from sklearn.feature_extraction.text import CountVectorizer
        common_words = self.get_top_n_words(df_all['answer_clean'], 20)
        df1 = pd.DataFrame(common_words, columns = ['Answer_clean_1', 'Count'])
        df1.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Top Words", ylabel = "Count", title = "Bar Chart of Top Words Frequency")
        df1.head()
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_9.png', dpi=300, bbox_inches='tight')
        # Frequency Charts
        common_words2 = self.get_top_n_bigram(df_all['answer_clean'], 30)
        df2 = pd.DataFrame(common_words2, columns=['Answer_clean_1', "Count"])
        df2.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Bigram Words", ylabel = "Count", title = "Bar chart of Bigrams Frequency")
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_10.png', dpi=300, bbox_inches='tight')
        # Frequency Charts
        common_words3 = self.get_top_n_trigram(df_all['answer_clean'], 30)
        df3 = pd.DataFrame(common_words3, columns = ['Answer_clean_1' , 'Count'])
        df3.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Trigrams Words", ylabel = "Count", title = "Bar chart of Trigrams Frequency")
        plt.savefig('/home/ubuntu/deeppatentsimilarity/python_screen/NLP_CA/images/plot_11.png', dpi=300, bbox_inches='tight')
        # Part-of -Speech Tagging
        from textblob import TextBlob, Word, Blobber
        blob = TextBlob(str(df_all['answer_clean']))
        pos_df = pd.DataFrame(blob.tags, columns = ['word', 'pos'])
        pos_df = pos_df.pos.value_counts()[:30]
        pos_df.plot(kind='bar', xlabel = "Part Of Speech", ylabel = "Frequency", title = "Bar Chart of the Frequency of the Parts of Speech", figsize=(10, 6))
        from sklearn.feature_extraction.text import CountVectorizer
        vec=CountVectorizer().fit(df.Answer)
        bag_of_words = vec.transform(df.Answer)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        common_words = words_freq[:10]
        df1 = pd.DataFrame(common_words, columns = ['Answer_clean_1', 'Count'])
        df1.groupby('Answer_clean_1').sum()['Count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel = "Top Words", ylabel = "Count", title = "Bar Chart of Top Words Frequency")
        # Frequency Charts


@st.cache(allow_output_mutation=True)
def df_clean (df_inter):
    df_all, freq, less_freq = Interview_report(df_inter).preprocessing()
    return df_all, freq, less_freq

st.write("A Sample of the CSV")
df_sample = pd.read_csv('df_nlp_ca.csv')
st.dataframe(df_sample.head(2))

uploaded_file_inter = st.file_uploader("Upload CSV file of Interviewees!")
if uploaded_file_inter is not None:    
    #read csv
    df_inter = pd.read_csv(uploaded_file_inter)
    df_inter.head()
    # st.select_slider('Select Type of Analysis', ['Word count', 'Bigram and Trigram'])
    
    cols = st.columns((4))
    df_all, freq, less_freq = df_clean(df_inter)
    ana_type = cols[0].selectbox('Select Type of Analysis', ['Word count', 'Bigram and Trigram'])

    if ana_type == 'Word count':
        Level_analysis = cols[1].selectbox(
        "level of the anlysis:", ["Overall", "Organization", "Interviewees"], index=0
        )
        if Level_analysis == "Overall":
            df_all_overall = df_all[['Organization', 'interview_len', 'word_count', 'polarity']]
            st.table(df_all_overall.head(3))

        if Level_analysis == "Organization":
            org_lits = df_inter.Organization.unique().tolist()
            org_select = cols[2].selectbox("Organization:", org_lits, index=1)
            st.write(org_select)
            df_all['answer_clean_org'] = df_all.groupby(['Organization'])['answer_clean'].transform(lambda x : ' '.join(x))
            df_all['organ_word_count'] = df_all.groupby(['Organization'])['word_count'].transform(lambda x: sum(x))
            df_all_org = df_all[df_all['Organization'] == 'org_select']
            st.table(df_all_org.head(3))

        if Level_analysis == "Interviewees":
            interviewees_lits = df_inter.Expert_id.unique().tolist()
            inter_select = cols[2].selectbox("Organization:", interviewees_lits, index=1)
            st.write(inter_select)
            df_all['answer_clean_interviewees'] = df_all.groupby(['Expert_id'])['answer_clean'].transform(lambda x : ' '.join(x))
            df_all['interviewees_word_count'] = df_all.groupby(['Expert_id'])['word_count'].transform(lambda x: sum(x))
            df_all_org = df_all[df_all['Expert_id'] == inter_select]
            st.table(df_all_org.head(3))
    if ana_type == 'Bigram and Trigram':
        Level_analysis = cols[1].selectbox(
        "level of the anlysis:", ["Overall", "Organization", "Interviewees"], index=0
        )
        if Level_analysis == "Overall":
            df_all_overall = df_all[['Organization', 'interview_len', 'word_count', 'polarity']]
            st.table(df_all_overall.head(3))

        if Level_analysis == "Organization":
            org_lits = df_inter.Organization.unique().tolist()
            org_select_gram = cols[2].selectbox("Organization:", org_lits, index=1)
            df_all['answer_clean_org'] = df_all.groupby(['Organization'])['answer_clean'].transform(lambda x : ' '.join(x))
            df_all_org_gram = df_all[df_all['Organization'] == org_select_gram]
            df_all_gram_organ = df_all[['Organization', 'interview_len', 'word_count', 'polarity']]

            get_top_n = cols[3].selectbox("Get Top N:", ['Words', 'Bigram', 'Trigram'], index=0) 
            if get_top_n == 'Words':
                get_top_n_words = Interview_report(df_inter).get_top_n_words(df_all_org_gram.answer_clean_org)
                dic_word = get_top_n_words[:10]
                st.table(pd.DataFrame(dic_word))
            if get_top_n == 'Bigram':
                get_top_n_bigram = Interview_report(df_inter).get_top_n_bigram(df_all.answer_clean_org)
                dic_bigram = get_top_n_bigram[:10]
                st.table(pd.DataFrame(dic_bigram))

            if get_top_n == 'Trigram':
                get_top_n_trigram = Interview_report(df_inter).get_top_n_trigram(df_all.answer_clean_org)
                dic_trigram = get_top_n_trigram[:10]
                st.table(pd.DataFrame(dic_trigram))

        if Level_analysis == "Interviewees":
            inter_lits = df_inter.Expert_id.unique().tolist()
            inter_select_gram = cols[2].selectbox("Expert_id:", inter_lits, index=1)
            df_all['answer_clean_inter'] = df_all.groupby(['Expert_id'])['answer_clean'].transform(lambda x : ' '.join(x))
            df_all_org_gram = df_all[df_all['Expert_id'] == inter_select_gram]
            df_all_gram_organ = df_all[['Organization', 'interview_len', 'word_count', 'polarity']]
            
            get_top_n = cols[3].selectbox("Get Top N:", ['Words', 'Bigram', 'Trigram'], index=0) 
            if get_top_n == 'Words':
                get_top_n_words = Interview_report(df_inter).get_top_n_words(df_all_org_gram.answer_clean_inter)
                dic_word = get_top_n_words[:10]
                st.table(pd.DataFrame(dic_word))

            if get_top_n == 'Bigram':
                get_top_n_bigram = Interview_report(df_inter).get_top_n_bigram(df_all.answer_clean_inter)
                dic_bigram = get_top_n_bigram[:10]
                st.table(pd.DataFrame(dic_bigram))

            if get_top_n == 'Trigram':
                get_top_n_trigram = Interview_report(df_inter).get_top_n_trigram(df_all.answer_clean_inter)
                dic_trigram = get_top_n_trigram[:10]
                st.table(pd.DataFrame(dic_trigram))


    if st.button("Submit"):
        st.write("This is a list of chemicals satisfying your criteria: ")
        st.table(satisfying_list)
        if st.button("Recommendation Products"):
            # recommendation_item = model.get_similar_items(items=[430018], k=10)
            # dfS = pd.DataFrame(recommendation_item)
            # dfS = dfS.replace({"similar":chemdict})
            # dfS = dfS.replace({"productId":chemdict})
            st.write("This is a list of recommendation products: ")
            st.table(dfS)





# st.markdown('<h1 style="background-color: gainsboro; padding-left: 10px; padding-bottom: 20px;">Indicator Search Engine</h1>', unsafe_allow_html=True)
# df_example = st.text_input('', help='Enter the search string and hit Enter/Return')

# # uploaded_file_example = st.file_uploader("Upload CSV file of Indicator example!")
# # if uploaded_file_example is not None:    
# #     #read csv
# #     df_example = pd.read_csv(uploaded_file_example)
# #     df_example.head()

# if st.button("Search"):

#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     sentences_example = df_example
#     sentences_docs = df_docs.paragraph_text.to_list()

#     embeddings_example = model.encode(sentences_example)
#     embeddings_docs = model.encode(sentences_docs)

#     df_results = get_top_n_similar_patents_df(embeddings_example, embeddings_docs)
#     st.table(df_results)
#     # st.download_button("Press to Download", df_results,"file.csv","text/csv",key='download-csv')
#     st.download_button(label='📥 Download Current Result', data=convert_df_to_csv(df_results), file_name= 'df_results.csv')
    
#     #Store sentences & embeddings on disc
#     with open('embeddings_docs.pkl', "wb") as fOut:
#         pickle.dump({'sentences': sentences_docs, 'embeddings': embeddings_docs}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#     #Store sentences & embeddings on disc
#     with open('embeddings_example.pkl', "wb") as fOut:
#         pickle.dump({'sentences': sentences_example, 'embeddings': embeddings_example}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#     #Load sentences & embeddings from disc
#     with open('embeddings_example.pkl', "rb") as fIn:
#         stored_data = pickle.load(fIn)
#         stored_sentences = stored_data['sentences']
#         stored_embeddings = stored_data['embeddings']
