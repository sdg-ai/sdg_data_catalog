#import sys
#sys.path.append("..")

import torch
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_transformation import Preprocessor
from evaluation import EvaluationIndex
from active_learning import *
from model import *
from metrics import SampleMetrics

import torch
import json
import ast

import spacy
nlp = spacy.load("en_core_web_sm")

from transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=18)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 200)
    
    def forward(self, ids, mask, labels):
        output_1= self.l1(ids, mask, labels = labels)
        output_2 = self.l2(output_1[0])
        output = self.l3(output_2)
        return output

def vec_to_tags(tags, vecs, max_seq_len=256):
    """
    change vector to tags
    """
    idx_to_tag = {key: idx for key, idx in enumerate(tags)}
    print(idx_to_tag)
    tags = []

    for vec in vecs:
        tag = [idx_to_tag.get(idx) for idx in vec[:max_seq_len]]
        tags.append(tag)

    return tags 

def sentences_to_vec(sentence, d_word_id, max_len, max_seq_len=256):
    sent = nlp(sentence)
    vec = [d_word_id.get(str(word), max_len) for word in sent[:max_seq_len]]
    return vec + [0] * (max_seq_len - len(vec))

def tags_to_vec(tags, d_tags_id, max_seq_len=256):
    vec = [d_tags_id.get(tag) for tag in tags[:max_seq_len]]
    return vec + [0] * (max_seq_len - len(vec))

def sentences_to_vec_nopad(sentence, d_word_id, max_len, max_seq_len=256):
    sent = nlp(sentence)
    vec = [d_word_id.get(str(word), max_len) for word in sent[:max_seq_len]]
    return vec

def tags_to_vec_nopad(tags, d_tags_id, max_seq_len=256):
    vec = [d_tags_id.get(tag) for tag in tags[:max_seq_len]]
    return vec

class DataPool(object):
    """
    DataPool is used to store all datas, including annotated and unannotated data.
    DataPool also provide a unified data acquisition interface.
    """

    def __init__(self, annotated_texts: list, annotated_labels: list,
                 unannotated_texts: list, unannotated_labels: list) -> None:
        """
        initialize DataPool object
        :param annotated_texts: annotated samples' text, must equal to annotated_labels
        :param annotated_labels: annotated samples' label, must equal to annotated_texts
        :param unannotated_texts: unannotated samples' text, must equal to unannotated_labels if unannotated labels is not None
        :param unannotated_labels: unannotated samples' label
        """

        self.annotated_texts = np.array(annotated_texts)
        self.annotated_labels = np.array(annotated_labels)
        self.unannotated_texts = np.array(unannotated_texts)
        self.unannotated_labels = np.array(unannotated_labels)
        self.new_annotated_texts = np.array([])
        self.new_annotated_labels = np.array([])

    def update(self, mode: str, annotated_texts=None, annotated_labels=None,
               unannotated_texts=None, unannotated_labels=None, selected_idx=None):
            '''
            This function update the values of annotated_texts and unannotated_texts
            selected_idx: the ids that we just annotated
            '''
            if selected_idx is None:
                if len(unannotated_labels) != len(selected_idx):
                    raise ValueError(f"In {mode} mode, if unannotated_labels is not None, they must"
                                     f" have the same length as the selected_idx")
                #updating the annotated labels
                if self.annotated_labels.shape[-1] != 0:
                    self.annotated_labels = np.concatenate((self.annotated_labels, np.array(unannotated_labels)))
                else:
                    self.annotated_labels = np.array(unannotated_labels)
                #updating the new annotated labels
                if self.annotated_labels.shape[-1] != 0:
                    self.new_annotated_labels = np.concatenate((self.new_annotated_labels, np.array(unannotated_labels)))
                else:
                    self.new_annotated_labels = np.array(unannotated_labels)
            else:
                if self.annotated_labels.shape[-1] != 0:
                    self.annotated_labels = np.concatenate(
                        (self.annotated_labels, self.unannotated_labels[selected_idx]))
                else:
                    self.annotated_labels = self.unannotated_labels[selected_idx]
            if self.annotated_labels.shape[-1] != 0:
                self.annotated_texts = np.concatenate((self.annotated_texts, self.unannotated_texts[selected_idx]))
            else:
                self.annotated_texts = self.unannotated_texts[selected_idx]
            # delete samples form unannotated database
            self.unannotated_texts = np.delete(self.unannotated_texts, selected_idx, axis=0)
            self.unannotated_labels = np.delete(self.unannotated_labels, selected_idx, axis=0)

    def get_total_number(self) -> int:
        """
        Get total number of samples.
        """
        return len(self.annotated_texts) + len(self.unannotated_texts)

class NER_AND_AL_Pipeline():
    """
    Word2vec(pre-trained) + BiLSTM + CRF + AL
    ===============================
    """

    def __init__(self, training_data_path='db/annotations/combined_annotation_launchlab_ai4g.csv', config_file='config.json', model_to_train='lstm'):
        self.preprocessor = Preprocessor(vocab=[], tags=[])
        self.model_to_train = model_to_train
        self.training_data_path = training_data_path
        self.training_dtf = pd.read_csv(self.training_data_path)
        #self.training_dtf = self.training_dtf[self.training_dtf['are_same_lenght']==True]
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.training_dtf['text'], self.training_dtf['biluo'], random_state=42, stratify=self.training_dtf['has_entity'], test_size=.15)
        print('Training set size: ', len(self.x_train), ', Testing set size: ', len(self.x_val))
        with open(config_file) as f:
            self.config = json.load(f)
        #super(NER_AND_AL_Pipeline, self).__init__()
        #if model_to_train=='bert':
        #    from transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel

    def word_embedding(self):
        """
        Embedding with word2vec.
        """
        logging.info("Step1 Begin: word embedding.\n")
        word_embedding_path = self.config['WORD2VEC']['word_embedding_path']
        embedding_dim = self.config["WORD2VEC"]["embedding_dim"]
        max_seq_len = self.config["WORD2VEC"]["max_seq_len"]
        #load file with the list of tags (change it with Marie's code)
        self.tags = self.config["WORD2VEC"]["tags"].split(' ')
        #labels are tags that are not null of pad
        self.labels = [label for label in self.tags if label not in ['O', '[PAD]', '[CLS]', '[SEP]', 'X']]
        all_words_embeds = pickle.load(open(word_embedding_path, 'rb'))
        awe = dict()
        for key, value in all_words_embeds[0].items():
            awe[key] = all_words_embeds[2][value]

        self.vocab = list(awe.keys())
        self.vocab.insert(0, "[PAD]")
        self.d_word_id = {key: idx for idx, key in enumerate(self.vocab)}
        self.d_tags_id = {key: idx for idx, key in enumerate(self.tags)}
        self.max_len = len(self.vocab) - 1
        

        self.training_text_vec = list(self.x_train.apply(lambda x: sentences_to_vec(x, d_word_id=self.d_word_id, max_len=self.max_len)).values)
        self.training_tags_vec = list(self.y_train.apply(lambda x: tags_to_vec(ast.literal_eval(x), d_tags_id=self.d_tags_id)).values)

        self.training_text_vec_nopad = list(self.x_train.apply(lambda x: sentences_to_vec_nopad(x, d_word_id=self.d_word_id, max_len=self.max_len)).values)
        self.training_tags_vec_nopad = list(self.y_train.apply(lambda x: tags_to_vec_nopad(ast.literal_eval(x), d_tags_id=self.d_tags_id)).values)
        assert len(self.training_tags_vec_nopad)==len(self.training_text_vec)
        assert len(self.training_tags_vec)==len(self.training_tags_vec_nopad)
        non_aligned = [i for i in range(len(self.training_text_vec_nopad)) if len(self.training_text_vec_nopad[i])!=len(self.training_tags_vec_nopad[i])]
        ind_keep = [el for el in range(1133) if el not in non_aligned]
        self.training_text_vec = list(pd.Series(self.training_text_vec)[ind_keep])
        self.training_tags_vec = list(pd.Series(self.training_tags_vec)[ind_keep])

        print('# non aligned ', len(non_aligned))
        
        self.preprocessor = Preprocessor(vocab=self.vocab, tags=self.tags)

        self.word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06),
                                             (len(self.preprocessor.vocab), embedding_dim))

        for word in awe.keys():
            self.word_embeds[self.preprocessor.word_to_idx[word]] = awe[word]                                                                                             

        #self.eval_ys = vec_to_tags(self.tags, self.eval_ys, max_seq_len)
        logging.info("Step1 Finish: word embedding.\n")
        return

    def build_bilstm_crf(self):
        """
        Build BiLSTM+CRF Model.
        """
        logging.info("Step2 Begin: build bilstm crf.\n")

        batch_size = self.config["BiLSTMCRF"]["batch_size"]
        device = self.config["BiLSTMCRF"]["device"]
        embedding_dim = self.config["BiLSTMCRF"]["embedding_dim"]
        hidden_dim = self.config["BiLSTMCRF"]["hidden_dim"]
        learning_rate = self.config["BiLSTMCRF"]["learning_rate"]
        model_path_prefix = self.config["BiLSTMCRF"]["model_path_prefix"]
        num_rnn_layers = self.config["BiLSTMCRF"]["num_rnn_layers"]
        num_epoch = self.config["BiLSTMCRF"]["num_epoch"]

        train_xs, train_ys = self.training_text_vec, self.training_tags_vec
        train_xs = torch.Tensor(train_xs)
        train_ys = torch.Tensor(train_ys)
        #DataLoader is used here to batch the data, shuffle the data and load the data in parallel using multiprocessing workers
        train_dl = DataLoader(TensorDataset(train_xs, train_ys), batch_size, shuffle=True)

        #if self.model_to_train == 'lstm':
        self.model = BiLSTMCRF(vocab_size=len(self.preprocessor.vocab),
                        tag_to_ix=self.preprocessor.tag_to_idx,
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        #adding our word2vec embedding
                        pre_word_embed=self.word_embeds,
                        num_rnn_layers=num_rnn_layers,
                        )
        #elif self.model_to_train == 'bert':
        #    self.model = BERTClass()
            #model.to(dev)

        #self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0)
        info = ""
        #if self.model_to_train == 'lstm':
        for epoch in range(num_epoch):
                self.model.train()
                bar = tqdm(train_dl)
                for bi, (xb, yb) in enumerate(bar):
                    self.model.zero_grad()
                    loss = self.model.loss(xb, yb)
                    loss.backward()
                    optimizer.step()
                    bar.set_description(f"{epoch + 1:2d}/{num_epoch} loss: {loss:5.2f}")
                info += f"{epoch + 1:2d}/{num_epoch} loss: {loss:5.2f}\n"
        logging.info(f"{info}")
        torch.save(self.model.state_dict(), model_path_prefix + ".pth")
        logging.info("Step2 Finish: bilstm crf.\n")

        return

    def predict_eval(self):
        """
        This function is using the model to make predictions on the training and test set 
        and report evaluation metrics calculated on the test set.
        Predict and evaluate:
        entity-level-F1
        sentence-level-accuracy
        """
        ###TODO add recall and precision
        logging.info("Step3 Begin: Predicting and evaluation.\n")
        #device = self.config.param("BiLSTMCRF", "device", type="string")
        max_seq_len = self.config["WORD2VEC"]["max_seq_len"]
        entity_digits = self.config["EvalF1"]["digits"]
        entity_return_report = self.config["EvalF1"]["return_report"]
        entity_average = self.config["EvalF1"]["average"]

        self.model.eval()
        self.y_val_restrict = [el[:256] for el in self.y_val]
        self.val_text_vec = list(self.x_val.apply(lambda x: sentences_to_vec(x, d_word_id=self.d_word_id, max_len=self.max_len)).values)
        self.val_tags_vec = list(self.y_val.apply(lambda x: tags_to_vec(ast.literal_eval(x), d_tags_id=self.d_tags_id)).values)

        self.val_text_vec_nopad = list(self.x_val.apply(lambda x: sentences_to_vec_nopad(x, d_word_id=self.d_word_id, max_len=self.max_len)).values)
        self.val_tags_vec_nopad = list(self.y_val.apply(lambda x: tags_to_vec_nopad(ast.literal_eval(x), d_tags_id=self.d_tags_id)).values)
        non_aligned = [i for i in range(len(self.val_text_vec_nopad)) if len(self.val_text_vec_nopad[i])!=len(self.val_tags_vec_nopad[i])]
        ind_keep = [el for el in range(len(self.val_text_vec)) if el not in non_aligned]
        self.val_text_vec = list(pd.Series(self.val_text_vec)[ind_keep])
        self.val_tags_vec = list(pd.Series(self.val_tags_vec)[ind_keep])
        self.y_val_ = list(pd.Series(self.y_val.values)[ind_keep])
        self.y_val_ = [ast.literal_eval(el) for el in self.y_val_]

        self.eval_dl = torch.from_numpy(np.array(self.val_text_vec)).int()
        scores, tag_seq, probs = None, None, None
        with torch.no_grad():

            scores, self.tag_seq, probs = self.model(self.eval_dl)
        self.tag_seq_ = vec_to_tags(self.tags, self.tag_seq, max_seq_len)
        self.val_tags = vec_to_tags(self.tags, self.val_tags_vec, max_seq_len)

        eval = EvaluationIndex()

        non_aligned = [i for i in range(len(self.tag_seq_)) if len(self.tag_seq_[i])!=len(self.y_val_[i])]
        ind_keep = [el for el in range(len(self.val_text_vec)) if el not in non_aligned]
        self.tag_seq_ = list(pd.Series(self.tag_seq_)[ind_keep])
        self.y_val_ = list(pd.Series(self.y_val_)[ind_keep])

        if entity_return_report:
            entity_f1_score, entity_return_report = eval.entity_level_f1(self.y_val_, self.tag_seq_,
                                                                         entity_digits, entity_return_report,
                                                                         entity_average)
            print(f"Classification report(Entity level):\n{entity_return_report}")
        else:
            entity_f1_score = eval.entity_level_f1(self.y_val_, self.tag_seq_, entity_digits,
                                                   entity_return_report, entity_average)

        logging.info(f"Entity-level F1: {entity_f1_score}")

        sentence_ac_score = eval.sentence_level_accuracy(self.y_val_, self.tag_seq_)
        print(f"Sentence-level Accuracy: {sentence_ac_score}")

        logging.info("Step03 Finish: Predicting and evaluation.\n")

        return entity_f1_score,sentence_ac_score


    def eval(self, unannotated_texts, unannotated_labels):
        """
        This function is using the model to make predictions on unnanotated data, for
        which we don't have labels already. These unnanotated data can be fed to the 
        active learning part of the pipeline further down in order to add more annotations
        and improve the model performances
        """
        logging.info("Step03 Begin: Predicting and evaluation.\n")
        #device = self.config.param("BiLSTMCRF", "device", type="string")
        max_seq_len = self.config["WORD2VEC"]["max_seq_len"]

        self.unannotated_texts_vec = list(unannotated_texts.apply(lambda x: sentences_to_vec(x, d_word_id=self.d_word_id, max_len=self.max_len)).values)

        unannotated_texts = torch.from_numpy(np.array(self.unannotated_texts_vec)).int()
        unannotated_labels = torch.from_numpy(np.array(unannotated_labels)).int()
        eval_dl = DataLoader(TensorDataset(unannotated_texts, unannotated_labels), 256, shuffle=False)
        self.model.eval()
        scores, tag_seq_l, probs, tag_seq_str = [], [], [], []
        with torch.no_grad():
            bar = tqdm(eval_dl)
            for bi,(xs,ys) in enumerate(bar):
                score, tag_seq, prob = self.model(xs)
                score, prob = score.cpu().detach().numpy(), prob.cpu().detach().numpy()
                tag_seq_l.extend(tag_seq)
                scores.extend(score.tolist())
                probs.extend(prob.tolist())
                tag_seq_str.extend(vec_to_tags(self.tags, tag_seq, max_seq_len))
        scores = np.array(scores)
        probs = np.array(probs)
        return scores, tag_seq_l, probs, tag_seq_str

    def active_learning(self, unannotated_texts, unannotated_labels, 
                        strategy='LTP', query_batch_fraction=.05, ret_idx=False):
        '''
        Pool-based active learning framework (from: A New Active Learning Strategy for CRF-Based Named Entity Recognition, M. Liu & al.)
        Require: 
            Labeled data set L,
            unlabeled data pool U,
            selection strategy φ(·),
            query batch size B
        while not reach stop condition do
            // Train the model using labeled set L
            train(L);
                for b = 1 to B do
                //select the most informative instance
                x∗ = arg maxx∈U φ(x)
                L = L union < x∗, label(x∗) >
                U = U − x∗
                end for
            end while
        '''
        '''
        Args:
        - strategy: which function is used to estimate the next most informative data point to annotate?
        - query_batch_fraction: the proportion of mosst informative unnanotated data that are selected for next annotation
        '''
        #other strategies to consider: https://modal-python.readthedocs.io/en/latest/
        logging.info("Begin active_learning.")
        strategy_mapping = {
            "RANDOM": RandomStrategy,
            "LC": LeastConfidenceStrategy,
            "NLC": NormalizedLeastConfidenceStrategy,
            "LTP": LeastTokenProbabilityStrategy,
            "MTP": MinimumTokenProbabilityStrategy,
            "MTE": MaximumTokenEntropyStrategy,
            "LONG": LongStrategy,
            "TE": TokenEntropyStrategy,
        }
        strategy_name = strategy.lower()
        max_seq_len = self.config["WORD2VEC"]["max_seq_len"]
        choice_number = int(len(unannotated_texts) * query_batch_fraction)
        strategy = strategy_mapping[strategy]

        #start an active learning iteration
        #self.build_bilstm_crf()
        #entity_f1_score,sentence_ac_score = self.predict_eval() 
        scores, tag_seq, probs, tag_seq_str = self.eval(unannotated_texts=unannotated_texts, unannotated_labels=unannotated_labels)
        idx = strategy.select_idx(choices_number=choice_number, probs=probs, scores=scores, best_path=tag_seq)
        tag_seq_str = [tag_seq_str[id] for id in idx]
        del self.model
        torch.cuda.empty_cache()
        if ret_idx:
            return idx

    @property
    def tasks(self):
        return [
            self.word_embedding,
            #self.active_learning,
        ]


if __name__ == '__main__':
    NER_AND_AL_Pipeline()