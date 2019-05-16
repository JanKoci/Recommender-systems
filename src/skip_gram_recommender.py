"""
Author: Jan Koci

Implementation of SkipGramModel and SkipGramRecommender classes.

"""
import torch
import torch.nn as nn
from data_utils import DataPipeline, get_item_tags, make_user_items, get_top_tags
import torch.nn.functional as F
import torch.optim as optimizer
from abstract.recommender import RecommenderAbstract


class SkipGramModel(nn.Module):
    """SkipGramModel class implements a Skip-gram based neural network model
    that creates vector representations of items using their document vectors
    and tags. It uses the negative sampling method to learn the weights of its
    hidden layer and also the tag embeddings.
    
    Attributes:
        docvecs   : matrix of document vectors of the items
        item_tags : matrix defining the tags of each item, the row index 
                    identifies the item and the values in it represent the tags,
                    note that this matrix has to have a padding, it is adviced 
                    to create it using the get_item_tags function
        embeddings_shape : shape of the tag embeddings that will be created, 
                           tuple with (num_of_tags, tag_dimensionality)
        hidden_dimension : dimensionality of the hidden layer
        out_dimension  : dimensionality of the output layer
        
    """
    def __init__(self,
                 docvecs,
                 item_tags,
                 embeddings_shape=(20, 20), 
                 hidden_dimension=20, 
                 out_dimension=10):
        
        super(SkipGramModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.docvecs = torch.Tensor(docvecs).to(self.device)
        self.item_padding_index = docvecs.shape[0]
        padding_row = torch.Tensor([0] * docvecs.shape[1]).to(self.device)
        padding_row = padding_row.reshape(1, docvecs.shape[1]).to(self.device)
        self.docvecs = torch.cat((self.docvecs, padding_row), 0)
        
        self.tag_embeddings = nn.Embedding(embeddings_shape[0]+1, 
                                           embeddings_shape[1], 
                                           padding_idx=embeddings_shape[0]).to(self.device)
        
        self.item_tags = torch.LongTensor(item_tags).to(self.device)
        padding_row = torch.LongTensor([self.tag_embeddings.padding_idx] * 
                                        self.item_tags.shape[1]).to(self.device)
        padding_row = padding_row.reshape(1, self.item_tags.shape[1])
        self.item_tags = torch.cat((self.item_tags, padding_row), 0)
        
        
        self.linear1 = nn.Linear(embeddings_shape[1]+docvecs.shape[1], hidden_dimension).to(self.device)
        self.linear2 = nn.Linear(hidden_dimension, out_dimension).to(self.device)
        self.logsigmoid = nn.LogSigmoid().to(self.device)
                
    
    def forward(self, item_batch, context_batch, neg_batch):
        """
        Forward function performing negative sampling with user-item interactions.
        Attributes: 
            item_batch : [batch_size] batch_size indexes of target items
            
            context_batch : [batch_size, num_items] batch_size of lists with 
                            item indexes, every list represents a user that 
                            interacted with corresponding target item from item_batch
                            the lists may contain a padding index and are 
                            always of size num_items.
                            
            neg_batch : [batch_size, num_samples, num_items] contains a list 
                        of num_samples users that did not interacted with the 
                        corresponding item in item_batch, these users are the 
                        negative samples, again users are lists of seen items
       
        """
        # 1. create item_vectors
        tag_sums = torch.sum(self.tag_embeddings(self.item_tags[item_batch]), 1)
        item_vectors = torch.cat((tag_sums, self.docvecs[item_batch]), 1)
        # feed them to NN
        h_relu = F.relu(self.linear1(item_vectors))
        item_out = self.linear2(h_relu)
        
        
        # 2. create context user vectors
        context_item_tag_embeddings = self.tag_embeddings(self.item_tags[context_batch])
        item_tag_sums = torch.sum(context_item_tag_embeddings, 2)
        context_item_vectors = torch.cat((item_tag_sums, 
                                          self.docvecs[context_batch]), 2)
        context_user_vectors = torch.sum(context_item_vectors, 1)
        # feed them to NN
        h_relu = F.relu(self.linear1(context_user_vectors))
        context_out = self.linear2(h_relu)

        
        # 3. create neg user vectors
        neg_item_tag_embeddings = self.tag_embeddings(self.item_tags[neg_batch])
        neg_item_tag_sums = torch.sum(neg_item_tag_embeddings, 3)
        neg_item_vectors = torch.cat((neg_item_tag_sums, 
                                      self.docvecs[neg_batch]), 3)
        neg_user_vectors = torch.sum(neg_item_vectors, 2)
        # 4. reshape neg vectors before feeding it to NN
        old_dim1, old_dim2, old_dim3 = neg_user_vectors.size()
        neg_user_vectors = neg_user_vectors.reshape(old_dim1*old_dim2, old_dim3)
        # feed them to NN
        h_relu = F.relu(self.linear1(neg_user_vectors))
        neg_out = self.linear2(h_relu)
        neg_out = neg_out.reshape(old_dim1, old_dim2, self.linear2.out_features)
        
        # compute the loss !!!
        # 1. compute positive value
        positive_dot = torch.bmm(context_out.view(len(item_batch), 1, 
                                                  self.linear2.out_features),
                                 item_out.view(len(item_batch), 
                                               self.linear2.out_features, 1)).squeeze()
        positive_value = self.logsigmoid(positive_dot)
        
        # 2. compute negative value
        negative_dot = torch.bmm(neg_out, item_out.unsqueeze(2)).squeeze()
        negative_value = self.logsigmoid(-1 * negative_dot)
        negative_value = torch.sum(negative_value, 1)
        
        # 3. return mean of the computed loss
        loss = positive_value + negative_value
        return -loss.mean()
    
    
    def get_vector_embedding(self, vector):
        h_relu = F.relu(self.linear1(vector))
        embedding = self.linear2(h_relu)
        return embedding
    
    
    
    
    
class SkipGramRecommender(RecommenderAbstract):
    """This class creates a recommender from the SkipGramModel
    
    Attributes:
        dataframe : object of RecommenderDataFrame class containing the dataset
        docvecs   : document vectors of items
        url_tag   : pandas dataframe object containing tags for each item
        
    """
    def __init__(self, dataframe, docvecs, url_tags):
        self.dataframe = dataframe
        self.docvecs = docvecs
        self.item_tags = None
        self.url_tags = url_tags
        self.model = None
        self.data_pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   

    @property
    def url_2_id(self):
        return self.dataframe.url_2_id
    
    @property
    def id_2_url(self):
        return self.dataframe.id_2_url
    
    @property
    def uid_2_id(self):
        return self.dataframe.uid_2_id
    
    @property
    def id_2_uid(self):
        return self.dataframe.id_2_uid

    
    def train(self, train_df=None, epochs=20,
                    num_tags=30,
                    tag_dimension=20,
                    hidden_dimension=20, 
                    out_dimension=10,
                    learning_rate=1.0,
                    num_negatives=20):
        """
        Trains the SkipGramModel to create item and user vectors 
        
        Attributes:
            num_tags : number of used tags
            tag_dimension : dimensionality of tag embeddings
            hidden_dimension : dimensionality of the hidden layer
            out_dimension  : dimensionality of the output layer
            num_negatives  : number of negative samples per target item
        
        """       
        embeddings_shape = (int(num_tags), int(tag_dimension))
        hidden_dimension = int(hidden_dimension)
        out_dimension = int(out_dimension)
        num_negatives = int(num_negatives)
        num_tags = int(num_tags)
        epochs = int(epochs)
        self.item_tags = get_item_tags(self.url_tags, self.dataframe.url_2_id, num_tags)
        self.item_tags = torch.LongTensor(self.item_tags)

        batch_size = 32
        
        if not self.data_pipeline:
            self.data_pipeline = DataPipeline(self.dataframe, batch_size, 
                        num_negatives, padding_index=len(self.dataframe.url_2_id))
        else:
            self.data_pipeline.num_negatives = num_negatives
            
        self.model = SkipGramModel(self.docvecs, self.item_tags, 
                                   embeddings_shape, hidden_dimension, 
                                   out_dimension).to(self.device)
        
        self._optimizer = optimizer.Adam(self.model.parameters(), 
                                               lr=learning_rate)
        step = 0
        avg_loss = 0
        
        for epoch in range(0, epochs):
            batch_generator = self.data_pipeline.generate_batch_fast()
            for batch in batch_generator:
                item_batch = batch[0]
                context_batch = batch[1]
                neg_batch = batch[2]
                
                loss = self.model(item_batch, context_batch, neg_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                step += 1
                avg_loss += loss
            
            print("[Epoch {0} of {1}] ... avg_loss = {2}".format(epoch, epochs, avg_loss/step))
            step = 0
            avg_loss = 0
        
        self.user_items = self.data_pipeline.user_items
        self.precompute_item_embeddings()
        
    
    def create_user_vector(self, uid):
        """
        Creates vector for user as a sum of his items.
        
        Attributes:
            uid : identifier of the user
            
        Returns:
            the created vector of user uid
            
        """
        user_items = self.user_items[self.dataframe.uid_2_id[uid]]
        item_tags = self.model.item_tags[user_items]
        tag_embeddings = self.model.tag_embeddings(item_tags)
        tag_sums = torch.sum(tag_embeddings, 1)
        item_vectors = torch.cat((tag_sums, self.model.docvecs[user_items]), 1)
        user_vector = torch.sum(item_vectors, 0)
        return user_vector
    
        
        
    def recommend(self, uid, n=None, exclude_seen_pages=False):
        """
        Creates list of recommendations for user uid in format: [(score, url),]
        containing the computed score and url of the recommended item.
        
        Attributes:
            uid : identifier of the user
            n   : number of tuples (recommendations) that will be returned
            
        Returns:
            the created list of recommendations
        """
        if uid not in self.dataframe.uid_2_id:
            return
        user_vector = self.create_user_vector(uid)
        user_embedding = self.model.get_vector_embedding(user_vector)
        scores = torch.mv(self.item_embeddings, user_embedding)
        scores = scores.sort(descending=True)
        if self.device == torch.device('cuda'):
            score_values = scores[0].cpu()
            score_indices = scores[1].cpu()
        else:
            score_values = scores[0]
            score_indices = scores[1]
        scores = list(zip(score_indices.numpy(), score_values.data.numpy()))
        
        if exclude_seen_pages:
            seen_pages = self.user_items[self.dataframe.uid_2_id[uid]].unique()
            scores = [(self.dataframe.id_2_url[tup[0]], tup[1]) for tup in scores
                                              if tup[0] not in seen_pages]
        else:
            scores = [(self.dataframe.id_2_url[tup[0]], tup[1]) for tup in scores]
        return scores[:n]
        
    
    def predict(self, uid, url):
        """
        Predict the score between a user and an item
        
        Attributes:
            uid  : the user identifier
            url  : url of the item
            
        Returns:
            the predicted score
            
        """
        user_vector = self.create_user_vector(uid)
        user_embedding = self.model.get_vector_embedding(user_vector)
        score = user_embedding.dot(self.item_embeddings[self.url_2_id[url]])
        return score
    
    
    def nearest_tags(self, tag_id, n=None):
        """
        Find nearest tags for tag specified by its id
        
        Attributes:
            tag_id  : position of the tag in tag_embeddings
            n       : number of nearest tags that will be returned
        
        Returns:
            list of n tuples (tag, score)
            
        """
        tag_vec = self.model.tag_embeddings.weight.data[tag_id]
        scores = torch.mv(self.model.tag_embeddings.weight.data[:-1], tag_vec)
        scores = scores.sort(descending=True)
        if self.device == torch.device('cuda'):
            score_values = scores[0].cpu()
            score_indices = scores[1].cpu()
        else:
            score_values = scores[0]
            score_indices = scores[1]
        scores = list(zip(score_indices.numpy(), score_values.data.numpy()))
        scores = [(self.tags[tup[0]], tup[1]) for tup in scores]
        return scores[:n]
    
    
    def nearest_items(self, url, n=None):
        """
        Find nearest items for item specified by its url
        
        Attributes:
            url  : item
            n    : number of nearest items that will be returned
        
        Returns:
            list of n tuples (url, score)
            
        """
        item = self.item_embeddings[self.url_2_id[url]]
        scores = torch.mv(self.item_embeddings, item)
        scores = scores.sort(descending=True)
        if self.device == torch.device('cuda'):
            score_values = scores[0].cpu()
            score_indices = scores[1].cpu()
        else:
            score_values = scores[0]
            score_indices = scores[1]
        scores = list(zip(score_indices.numpy(), score_values.data.numpy()))
        scores = [(self.id_2_url[tup[0]], tup[1]) for tup in scores]
        return scores[:n]    
        
    
    
    def precompute_item_embeddings(self):
        """
        Precomputes item vectors and stores them in self.item_embeddings
        """
        self.item_embeddings = torch.Tensor().to(self.device)
        batch_size = 16
        item_count = self.model.item_tags.shape[0]-1
        for i in range(0, item_count, batch_size):
            item_ids = list(range(i, i+batch_size if i+batch_size <= item_count else item_count))
            tag_sums = torch.sum(self.model.tag_embeddings(
                                        self.model.item_tags[item_ids]), 1)
            item_vectors = torch.cat((tag_sums, self.model.docvecs[item_ids]), 1)
            embeddings = self.model.get_vector_embedding(item_vectors)
            self.item_embeddings = torch.cat((self.item_embeddings, 
                                              embeddings), 0)
        
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
        
    def load_model(self, path, num_tags, tag_dimension, hidden_dimension, 
                   out_dimension, learning_rate=None, num_negatives=None, epochs=None):
        
        item_tags = get_item_tags(self.url_tags, self.dataframe.url_2_id, num_tags)
        self.tags = get_top_tags(self.url_tags, num_tags)
        
        self.model = SkipGramModel(self.docvecs, item_tags, (num_tags, tag_dimension), 
                                   hidden_dimension, out_dimension)
        self.model.load_state_dict(torch.load(path, map_location=self.device.type))
        self.model.eval()
        self.user_items = make_user_items(self.dataframe, len(self.dataframe.url_2_id), self.device)
        self.precompute_item_embeddings()
        
        
        
        
        
        
        
        
        

    
