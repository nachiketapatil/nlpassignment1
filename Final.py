import streamlit as st
import nltk
import pandas as pd
import numpy as np
from collections import defaultdict
from time import time
from random import shuffle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix

nltk.download('brown')
nltk.download('universal_tagset')

dataset = list(nltk.corpus.brown.tagged_sents(tagset = "universal"))
for sent in dataset:
    temp = ("^", "START")
    sent.insert(0, temp)

tags_temp = set()
tokens_temp = set()
for sent in dataset:
    for ele in sent:
        tokens_temp.add(ele[0])
        tags_temp.add(ele[1])
tags = list(tags_temp)
tokens = list(tokens_temp)
n_tags = len(tags)
n_tokens = len(tokens)

def training(dataset):
    transition_freq = defaultdict(int)
    emission_freq = defaultdict(int)
    tag_freq = defaultdict(int)

    for sent in dataset:
        prev_tag = "START"
        for tup in sent:
            transition_freq[(prev_tag, tup[1])] += 1
            emission_freq[(tup[1], tup[0])] += 1
            tag_freq[tup[1]] += 1
            prev_tag = tup[1]
    
    a = 1e-8
    transition_mat  = [[0 for _ in range(n_tags)] for _ in range(n_tags)]
    emission_mat = [[0 for _ in range(n_tokens)] for _ in range(n_tags)]
    
    for i in range(n_tags):
        for j in range(n_tags):
            transition_mat[i][j] = (transition_freq[(tags[i], tags[j])] + a) / (tag_freq[tags[i]] + a*n_tags)
        for j in range(n_tokens):
            emission_mat[i][j] = (emission_freq[(tags[i], tokens[j])] + a) / (tag_freq[tags[i]] + a*n_tags)

    return transition_mat, emission_mat

def preprocess_sent(sent):
    words = sent.split()
    exception_punc = ['\'', '-']
    proc_words = []
    for w in words:
        if (w == " "):
            continue
        else:
            w = w.lower()
            s = 0
            n = len(w)
            for i in range(n):
                if (w[i].isalnum()):
                    continue
                elif (w[i] not in exception_punc):
                    proc_words.append(w[s:i])
                    proc_words.append(w[i])
                    s = i+1
            if (s < n):
                proc_words.append(w[s:])
    if (proc_words[-1].isalnum()):
        proc_words.append(".")
    proc_words[0] = proc_words[0].capitalize()
    for ele in proc_words:
        if (ele == ''):
            proc_words.remove(ele)
    return (proc_words)

def Viterbi_decoder(t_mat, e_mat, sent):
    em_min = np.array(e_mat).min()
    words = preprocess_sent(sent)
    print(words)
    n = len(words)
    prob_mat = [[0 for _ in range(n_tags)] for _ in range(n)]
    tag_mat = [[0 for _ in range(n_tags)] for _ in range(n)]

    ind = tags.index("START")
    try:
        w_ind = tokens.index(words[0])
    except ValueError:
        w_ind = -1
    for i in range(n_tags):
        if (w_ind == -1):
            prob_mat[0][i] = t_mat[ind][i] * em_min
        else:
            prob_mat[0][i] = t_mat[ind][i] * e_mat[i][tokens.index(words[0])]

    for s in range(1, n):
        prev_token = words[s-1]
        curr_token = words[s]
        
        try:
            w_curr_ind = tokens.index(curr_token)
        except ValueError:
            w_curr_ind = -1

        for t in range(n_tags):
            p = 0
            temp_tag = 0
            for u in range(n_tags):
                temp_p = prob_mat[s-1][u] * t_mat[u][t]
                if temp_p > p:
                    p = temp_p
                    temp_tag = u
            if (w_curr_ind == -1):
                prob_mat[s][t] = p * em_min
            else:
                prob_mat[s][t] = p * e_mat[t][tokens.index(curr_token)]
            tag_mat[s][t] = temp_tag

    pred = []
    # Get the last tag 
    pmax = 0
    last_tag = 0
    for i in range(n_tags):
        if prob_mat[-1][i] > pmax:
            pmax = prob_mat[-1][i]
            last_tag = i
    pred.append(tags[last_tag])

    t = last_tag
    for i in range(n-1, 0, -1):
        t = int(tag_mat[i][t])
        pred.append(tags[t])

    pred.reverse()
    return (pred, words)
   # tup = (pred2, words)
    #return tup

tm, em = training(dataset)

# for i in range(len(tm)):
#     for j in range(len(tm)):
#         if (tm[i][j] == 0):
#             print("%d\t%d"%(i,j))
# for i in range(len(em)):
#     for j in range(len(em[0])):
#         if(em[i][j] == 0):
#             print("%d\t%d"%(i,j))

# sent  = input("Enter the sentence to be checked : \n")
# start = time()
# output = Viterbi_decoder(tm, em, sent)
# end = time()
# dict = {'Words' : output[0], 'Predicted Tags' : output[1]}
# df = pd.DataFrame(dict)
# df

# Function to convert DataFrame to HTML with styling
def df_to_html(df):
    return df.style.to_html(classes='styled-table')

# Title of the app
st.title("POS Tagger Using Hidden Markov Model")

# Subheader to explain the functionality
st.subheader("Enter a sentence below to get its Part-of-Speech tags")

# Text input for user to enter a sentence
sent = st.text_input("Sentence")

if sent:
    # Button to trigger the POS tagging process
    if st.button("Get POS Tags"):
        # Measure the time taken for the Viterbi decoder to process the input
        start = time()
        output = Viterbi_decoder(tm, em, sent)
        end = time()

        # Create a DataFrame to display the results
        data = {
            'Words': output[1],
            'Predicted Tags': output[0]
        }
        df = pd.DataFrame(data)

        # Convert DataFrame to HTML
        html_table = df_to_html(df)
        
        # Inject HTML into Streamlit
        st.markdown(html_table, unsafe_allow_html=True)

        # Show the time taken for processing
        st.caption(f"Processing Time: {end - start:.4f} seconds")

# Add some additional info or notes if necessary
st.markdown("""
### About the assignment
In the English language, one of the most important feature for understanding the context and nuances of the sentences is the presence and identification of "Parts of Speech". For any NLP task, it is crucial for a machine to understand and identify the correct Parts of speech for the words that occur in a sentence. This assignment is focused on solving this problem.

To perform POS tagging, there are a multitude of models that can either be rule-based, use stochastic models/processes or make use of neural methods. In this assignment, we shall be dealing with the second type which makes use of the probability and stochastic models to get the correct POS tags. We make use of a Hidden Markov Model to parse through the sentences and get the path probabilities. Then, we proceed to perform Viterbi decoding that reduces the complexity from exponential to linear and get the path of maximum probability.

Presented here is a small demo where the user can input any sentence of their choice and the model shall return the POS tags. The accuracy, precision, recall and other metrics for the model have been elaborated in the presentation.
""")

# Adding custom CSS for table styling (you can add this in the HTML string)
st.markdown("""
<style>
.styled-table {
    width: 100%;
    border-collapse: collapse;
}
.styled-table th, .styled-table td {
    border: 1px solid #ddd;
    padding: 8px;
}
.styled-table tr:nth-child(even) {
    background-color: #f2f2f2;
}
.styled-table tr:hover {
    background-color: #ddd;
}
.styled-table th {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)
