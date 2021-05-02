import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def lambda_nltk_news(df: pd.DataFrame, col: str) -> list:

    df[col] = [i[2:-2].split('","') for i in df[col]]

    df.loc[:, col] = df[col].fillna('None')
    stop_words = stopwords.words('english')
    special_char = re.compile(r'[\W]')

    comment_words = ''
    all_text = []
    for index, news_list in enumerate(df[col]):
        cleaned_text_list = []
        for news in news_list:
            word_tokens = word_tokenize(news)
            no_stops = [i for i in word_tokens if i.lower() not in stop_words]
            no_special_char = [special_char.sub('',i) for i in no_stops if special_char.sub('', i) != '']
            cleaned_text = " ".join(i.lower() for i in no_special_char)
            comment_words += " ".join(i.lower() for i in no_special_char)+" "
            cleaned_text_list.append(cleaned_text)
        all_text.append(' '.join(i for i in cleaned_text_list))

    return all_text
