import pandas as pd
import json
import random
import helpers
import numpy as np
import torch


raw_data_path = '../data/raw'
processed_data_path = '../data/processed'


def load_df_pickle(filename):
    return pd.read_pickle(filename)


def load_metadata_json(filename):
    with open(filename, 'r') as f:
        metadata = json.load(f)
    return metadata


def make_df_metadata(metadata):
    temp = {}
    for url, data in metadata.items():
        if 'url' not in temp:
            temp['url'] = []
        temp['url'].append(url)
        for (key, values) in data.items():
            if key not in temp:
                temp[key] = []
            temp[key].append(values)
    return pd.DataFrame(temp)
    

def make_url_tags(df_metadata):
    temp = {}
    temp['url'] = []
    temp['tag'] = []
    for _, row in df_metadata.iterrows():
        for tag in row.tags:
            temp['url'].append(row.url)
            temp['tag'].append(str.lower(tag))
    return pd.DataFrame(temp)


def get_item_tags(url_tags, url_2_id=None, num_tags=20, padding=True):
    my_url_tags = url_tags.copy()
    tag_counts = my_url_tags.loc[my_url_tags.tag != ''].groupby('tag')['url'].count()
    tag_counts = tag_counts.sort_values(ascending=False)
    tag_counts_list = list(tag_counts.index)[2 : num_tags+2]
        

    my_url_tags.loc[:, 'tag'] = my_url_tags.tag.apply(lambda tag: num_tags 
                                             if tag not in tag_counts_list
                                             else tag_counts_list.index(tag))
        
    if url_2_id:
        my_url_tags.loc[:, 'url'] = my_url_tags.url.apply(
                                    lambda url: -1 if url not in url_2_id
                                                    else url_2_id[url])
        my_url_tags = my_url_tags.loc[my_url_tags.url != -1]    
        
    item_tags = my_url_tags.groupby('url')['tag'].unique()
        
    if padding:
        longest = item_tags.str.len().max()
        item_tags = item_tags.apply(lambda tags: np.append(tags, 
                        [num_tags for _ in range(0, longest-len(tags))]))
    return item_tags  


def get_top_tags(url_tags, n=20):
    tag_counts = url_tags.loc[url_tags.tag != ''].groupby('tag')['url'].count()
    tag_counts = tag_counts.sort_values(ascending=False)
    return list(tag_counts.index)[2 : n+2]   



class RecommenderDataFrame(object):
    def __init__(self, df, uid_2_id=None, url_2_id=None):
        self.df = df
        if url_2_id:
            self.url_2_id = url_2_id
            self.id_2_url = {id:url for url,id in url_2_id.items()}
            self.df = self.df.loc[self.df.page_url.isin(url_2_id)]
        else:
            self.url_2_id, self.id_2_url = helpers.make_item_mappings(self.df)
            
        if uid_2_id:
            self.uid_2_id = uid_2_id
            self.id_2_uid = {id:uid for uid,id in uid_2_id.items()}
            self.df = self.df.loc[self.df.uid.isin(uid_2_id)]
        else:
            self.uid_2_id, self.id_2_uid = helpers.make_user_mappings(self.df)


def make_user_items(dataframe, padding_index, device):
    temp = dataframe.df.loc[:, ('uid', 'page_url')]
    temp.loc[:, 'uid'] = temp.uid.apply(lambda uid: dataframe.uid_2_id[uid])
    user_items = temp.groupby('uid')['page_url'].unique()
    # transform urls to ids
    user_items = user_items.apply(lambda urls: 
                                [dataframe.url_2_id[url] for url in urls 
                                if url in dataframe.url_2_id])
    longest = user_items.str.len().max()
    # add padding
    user_items = user_items.apply(lambda urls: np.append(urls, 
                    [padding_index for _ in range(0, longest-len(urls))]))
    return user_items


class DataPipeline(object):
    def __init__(self, dataframe, batch_size, num_negatives, padding_index):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.stack = []
        self.padding_index = padding_index
        self.page_unique_views = None
        self.user_items = None
        self.unique_uids = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_pipeline()
        
    
    def init_pipeline(self):
        print("  -- creating pipeline")
        self.unique_uids = self.dataframe.df.uid.unique()
        # create user_items
        self.user_items = make_user_items(self.dataframe, self.padding_index, self.device)
    
        # initialize neg samples
        self.page_unique_views = self.dataframe.df.groupby('page_url')['uid'].unique()
        self.neg_item_ids = self.page_unique_views.apply(lambda pos_users: 
            [self.dataframe.uid_2_id[neg_user] for neg_user in self.unique_uids if neg_user not in pos_users])
        # initialize the stack    
        for url, uids in self.page_unique_views.items():
            for uid in uids:
                self.stack.append((self.dataframe.url_2_id[url], 
                                   self.dataframe.uid_2_id[uid]))
        
        

                
    def generate_batch(self):
        for idx in range(0, len(self.stack), self.batch_size):
            stack_batch = self.stack[idx : idx+self.batch_size]
            item_batch = [tuple[0] for tuple in stack_batch]
            
            user_batch = [tuple[1] for tuple in stack_batch]
            context_batch = self.user_items[user_batch]
            neg_uids = []
            
            for item in item_batch:
                neg_samples = random.sample(self.neg_item_ids[self.dataframe.id_2_url[item]], 
                                            self.num_negatives)
                neg_uids.append(neg_samples)
            neg_uids = torch.LongTensor(neg_uids).to(self.device)
            neg_batch = self.user_items[neg_uids]
            
            item_batch = torch.LongTensor(item_batch).to(self.device)
            
            yield (item_batch, context_batch, neg_batch)
           
            
            
    def prepare_neg_samples(self):
        self.neg_stack = []
        for (item, user) in self.stack:
            neg_ids = random.sample(self.neg_item_ids[self.dataframe.id_2_url[item]],
                                        self.num_negatives)
            self.neg_stack.append(neg_ids)
        
            
#    PREFERED
    def generate_batch_fast(self):
        self.prepare_neg_samples()
        random.shuffle(self.stack)
        stack_size = random.randint(0, len(self.stack))
        for idx in range(0, stack_size, self.batch_size):
            stack_batch = self.stack[idx : idx+self.batch_size]
            item_batch = [tuple[0] for tuple in stack_batch]
            item_batch = torch.LongTensor(item_batch).to(self.device)
            
            context_batch = [tuple[1] for tuple in stack_batch]
            context_batch = self.user_items[context_batch]
            
            neg_ids = self.neg_stack[idx : idx+self.batch_size]
            neg_ids = torch.LongTensor(neg_ids).to(self.device)
            neg_batch = self.user_items[neg_ids]
            
            yield (item_batch, context_batch, neg_batch)


           
            
            
