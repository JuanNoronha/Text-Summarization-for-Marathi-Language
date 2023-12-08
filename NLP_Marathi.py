#Importing all the necessary libraries and framework

import streamlit as st
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


nltk.download('punkt')


#Defining marathi stopwords

# Marathi stopwords
marathi_stopwords = set([
    "आहे", "या", "आणि", "व", "नाही", "आहेत", "यानी", "हे", "तर",
    "ते", "असे", "होते", "केली", "हा", "ही", "पण", "करणयात", "काही",
    "केले", "एक", "केला", "अशी", "मात्र", "त्यानी", "सुरू", "करून",
    "होती", "असून", "आले", "त्यामुळे", "झाली", "होता", "दोन", "झाले",
    "मुबी", "होत", "त्या", "आता", "असा", "याच्या", "त्याच्या", "ता",
    "आली", "की", "पम", "तो", "झाला", "त्री", "तरी", "म्हणून", "त्याना",
    "अनेक", "काम", "माहिती", "हजार", "सागितले", "दिली", "आला", "आज",
    "ती", "तसेच", "एका", "याची", "येथील", "सर्व", "न", "डॉ", "तीन",
    "येथे", "पाटील", "असलयाचे", "त्याची", "काय", "आपल्या", "म्हणजे",
    "याना", "म्हणाले", "त्याचा", "असलेल्या", "मी", "गेल्या", "याचा",
    "येत", "म", "लाख", "कमी", "जात", "टा", "होणार", "किवा", "का",
    "अधिक", "घेऊन", "परयतन", "कोटी", "झालेल्या", "निर्ण्य", "येणार",
    "व्यकत"
])




#Data pre-processing and Text extraction
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in marathi_stopwords]
    return tokens

#Function for using cosine similarity in Text 
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

#Text summarization using extraction method
def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    sentence_tokens = [preprocess_text(sentence) for sentence in sentences]
    sentence_vectors = {sentence: np.zeros(len(marathi_stopwords)) for sentence in sentences}
    for i, sentence in enumerate(sentences):
        tokens = preprocess_text(sentence)
        for token in tokens:
            if token in marathi_stopwords:
                sentence_vectors[sentence][list(marathi_stopwords).index(token)] = 1

    summary = []
    for i in range(len(sentences)):
        sim_scores = [cosine_similarity(sentence_vectors[sentences[i]],  sentence_vectors[sentences[j]]) for j in range(len(sentences))]
        top_indices = np.argsort(sim_scores)[-num_sentences:]
        summary.extend([sentences[idx] for idx in top_indices])

    return ' '.join(list(set(summary)))





