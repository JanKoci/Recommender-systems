import numpy as np
import math
import datetime
import random


def add_date_value(docvecs, metadata, dataframe, new_min=None, new_max=None):
    all_dates = [metadata[dataframe.id_2_url[i]]['date'][0].split(sep='T')[0] 
                                        for i in range(0, docvecs.shape[0])]
    dates_only = [datetime.datetime.strptime(date_str, '%Y-%m-%d') 
                    for date_str in all_dates if date_str]
    min_date = min(dates_only)
    max_date = max(dates_only)
    dates = [datetime.datetime.strptime(date_str, '%Y-%m-%d') 
                            if date_str else '' for date_str in all_dates]
    
    date_values = []
    for date in dates:
        if not date:
            value = random.randint(0, (max_date - min_date).days)
        else:
            value = (date - min_date).days
        date_values.append(value)
    
    min_value = min(date_values)
    max_value = max(date_values)
    
    if new_min is not None:
        date_values = [(((value - min_value) * (new_max - new_min)) / 
                    (max_value - min_value)) + new_min for value in date_values]

    date_values = np.array(date_values)
    date_values = date_values.reshape((date_values.shape[0], 1))
    return np.concatenate((docvecs, date_values), axis=1)
        
    


def get_confidence(df, uid, url, alpha=12, epsilon=1.39e-04):
    preference = df.loc[(df.uid == uid) & (df.page_url == url)].time.sum()
    confidence = 1 + (alpha * math.log(1 + (preference/epsilon)))
    return confidence
    

def train_test_df(df, min_interactions=6, state=1):
    df_temp = df_min_interactions(df, min_interactions)
    users_unique_urls = df_temp.groupby('uid')['page_url'].unique()
    test_indexes = list()
    for uid, urls in users_unique_urls.items():
        half_index = len(urls) // 2
        if state == 0:
            test_urls = urls[:half_index]
        else:
            test_urls = urls[half_index:]
        test_indexes.extend(df_temp.loc[(df_temp.uid == uid) &
                                        (df_temp.page_url.isin(test_urls))].index)

    return df.loc[~df.index.isin(test_indexes)], df_temp.loc[test_indexes]


def make_item_mappings(df):
    unique_urls = list(df.page_url.unique())
    url_2_id = {}
    id_2_url = {}
    for i, url in enumerate(unique_urls):
        url_2_id[url] = i
        id_2_url[i] = url

    return (url_2_id, id_2_url)


def make_user_mappings(df):
    unique_users = list(df.uid.unique())
    uid_2_id = {}
    id_2_uid = {}
    for i, uid in enumerate(unique_users):
        uid_2_id[uid] = i
        id_2_uid[i] = uid

    return (uid_2_id, id_2_uid)



def make_labels_for_visualisation(df, url_2_id, uid_2_id, filename='user_labels.tsv'):
    users_unique_urls = df.groupby('uid')['page_url'].unique()
    users_unique_url_ids = users_unique_urls.apply(lambda urls: [url_2_id[url] for url in urls])
    users_unique_url_ids = users_unique_url_ids.sort_values()
    userId_pageId_list = list(zip(users_unique_url_ids.index, users_unique_url_ids.values))

    userId_pageId_list = [(uid_2_id[tup[0]], list(tup[1])) for tup in userId_pageId_list]

    userId_pageId_list = sorted(userId_pageId_list, key=lambda tup: tup[0])
    return userId_pageId_list
    with open(filename, 'w') as file:
        file.write("Id\tPages\n")
        for tup in userId_pageId_list:
            file.write(str(tup[0]))
            file.write('\t')
            file.write(str(tup[1]))
            file.write('\n')


# SLOOOOOOOOW :)
def df_min_interactions(df, min_interactions):
    """ take only users that interacted with at least min_interactions
    different pages
    """
    users_unique_urls = df.groupby('uid')['page_url'].unique()
    users_url_count = users_unique_urls.apply(lambda urls: len(urls))
    df = df.loc[df.uid.isin(users_url_count[users_url_count >= min_interactions].index)]

    return df
