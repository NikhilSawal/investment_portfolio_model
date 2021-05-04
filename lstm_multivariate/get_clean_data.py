import re
import typing
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

def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    object_columns = list(data.select_dtypes(include=['object']).columns)
    float_columns = list(data.select_dtypes(include=['float64']).columns)

    object_indexes = [data.columns.get_loc(i) for i in object_columns]
    float_indexes = [data.columns.get_loc(i) for i in float_columns]

    data.iloc[:,float_indexes] = data.iloc[:,float_indexes].fillna(0.0)
    data.iloc[:,object_indexes] = data.iloc[:,object_indexes].fillna('{None}')

    # Perform feature engineering
    data.loc[:,'top_3_news'] = lambda_nltk_news(data, 'top_3_news')
    data.loc[:,'news_source'] = lambda_nltk_news(data, 'news_source')

    data = data[['date', 'delta_price', 'delta_price_perc', 'top_3_news',
                 'news_source', 'snp_500', 'snp_500_delta', 'snp_500_delta_perc',
                 'dow_30', 'dow_30_delta', 'dow_30_delta_perc', 'nasdaq',
                 'nasdaq_delta', 'nasdaq_delta_perc', 'price']]

    return data
