from nltk.corpus import wordnet as wn
import networkx as nx
import random
from collections import defaultdict
from gensim.models.fasttext import load_facebook_model
import os
import logging
import numpy as np
from string import punctuation


# ----------------------------------------------------------------
# generate graph and graph successors
# ----------------------------------------------------------------


def get_graph_and_root(wordnet, pos, directed=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    roots = []
    for synset in wordnet.all_synsets(pos):
        for hypernym in synset.hypernyms():
            G.add_edge(hypernym.name(), synset.name())

        if len(synset.hypernyms()) == 0 and len(synset.hyponyms()) > 0:
            roots.append(synset)

    return G, roots


def get_successors(graph, node, node_successors):
    number = set(graph.successors(node))
    for hyponym in wn.synset(node).hyponyms():
        cur_number, node_successors = get_successors(graph, hyponym.name(), node_successors)
        number.update(cur_number)
    node_successors[node] = number
    return number, node_successors


# ----------------------------------------------------------------
# generate and save test parents
# ----------------------------------------------------------------

def select_candidates_test_parents(node_successors, graph):
    under_threat_of_deprivation_of_parental_rights = defaultdict(set)

    for synset in node_successors:
        synset = wn.synset(synset)

        cur_hypernyms = synset.hypernyms()
        cur_hyponyms = synset.hyponyms()

        if len(cur_hyponyms) == 0 and 1/nx.shortest_path_length(graph, 'entity.n.01', synset.name()) >= 0.2:
            for hypernym in cur_hypernyms:
                under_threat_of_deprivation_of_parental_rights[hypernym].add(synset.name())

    return under_threat_of_deprivation_of_parental_rights


def select_test_parents(under_threat_of_deprivation_of_parental_rights):
    deprived_of_parental_rights = list(under_threat_of_deprivation_of_parental_rights)
    random.shuffle(deprived_of_parental_rights)
    random.seed(42)
    deprived_of_parental_rights = deprived_of_parental_rights[:1000]
    test_synsets = set()

    for test_synset in deprived_of_parental_rights:
        test_synsets.update(under_threat_of_deprivation_of_parental_rights[test_synset])

    return test_synsets


def save_test_data(test_data, path_to_save):
    with open(path_to_save, 'w') as t:
        for item in test_data:
            t.write(item + "\n")


def load_test_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    return [t.strip() for t in test_data]


# ----------------------------------------------------------------
# generate and save test children
# ----------------------------------------------------------------

def compute_lemmas(lemmas, lemma2id, model, pos, directed=True):
    link_pairs_ids = []
    node_triplets_lemmas = []

    for lemma in lemmas:
        _id = lemma2id[lemma.lower()]
        for synset in wn.synsets(lemma, pos=pos):
            for hypernym in synset.hypernyms():
                for h_lemma in hypernym.lemmas():
                    link_pairs_ids.append((lemma2id[h_lemma.name().lower()], _id))
                    if not directed:
                        link_pairs_ids.append((_id, lemma2id[h_lemma.name().lower()]))

        vector = get_multiword_vectors(model, [lemma])[0]
        node_triplets_lemmas.append((_id, vector, lemma))

    return node_triplets_lemmas, link_pairs_ids


def compute_synsets(synsets, _lemma2id, model, _pos, directed=True):
    link_pairs_offsets = []
    node_triplets_synsets = []

    for synset in synsets:
        for hypernym in wn.synset(synset).hypernyms():
            link_pairs_offsets.append((hypernym.offset(), wn.synset(synset).offset()))
            if not directed:
                link_pairs_offsets.append((wn.synset(synset).offset(), hypernym.offset()))

        synset_lemmas = [lemma.name() for lemma in wn.synset(synset).lemmas()]
        vector = get_synset_vectors(model, synset_lemmas)
        node_triplets_synsets.append((wn.synset(synset).offset(), vector, synset))

    return node_triplets_synsets, link_pairs_offsets


# ----------------------------------------------------------------
# save datasets for Graph-BERT
# ----------------------------------------------------------------


def save_node(dir_path, triplets):
    with open(f"{dir_path}/node", 'w') as w:
        for (id_, vector, name_) in triplets:
            string_vector = "\t".join([str(i) for i in vector])
            w.write(f"{id_}\t{string_vector}\t{name_}\n")


def save_link(dir_path, pairs):
    with open(f"{dir_path}/link", 'w') as w:
        for (i, j) in pairs:
            w.write(f"{i}\t{j}\n")



def load_src_embeddings(path):
    idx_features_labels = np.genfromtxt(path, dtype=np.dtype(str))
    matrix = np.array(idx_features_labels[:, 1:], np.dtype(float))
    synsets_ordered = list(idx_features_labels[:, 0])
    return matrix, synsets_ordered


def load_tgt_embeddings(path):
    fasttext_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.split("\t")
            fasttext_dict[line_split[-1].strip()] = np.array([float(num) for num in line_split[1:-1]])
    return fasttext_dict


# ----------------------------------------------------------------
# generate node embeddings
# ----------------------------------------------------------------


def get_data_vectors(model, data):
    vectors = np.zeros((len(data), model.vector_size))
    for i, word in enumerate(data):
        vectors[i, :] = model.wv[word]
    return vectors


def get_multiword_vectors(model, data):
    vectors = np.zeros((len(data), model.vector_size))
    for i, multi_word in enumerate(data):
        words = multi_word.replace("_", " ").split()
        vectors[i, :] = np.sum(get_data_vectors(model, words), axis=0) / len(words)
    return vectors


def get_synset_vectors(model, texts):
    sum_vector = np.zeros(model.vector_size)
    for text in texts:
        words = [i.strip(punctuation) for i in text.replace("_", " ").split()]
        sum_vector += np.sum(get_data_vectors(model, words), axis=0) / len(words)
    return sum_vector / len(texts)


# ----------------------------------------------------------------
# main class
# ----------------------------------------------------------------

class DatasetCreator:
    def __init__(self, fasttext_path, dir_path):
        self.dir_path_synsets = dir_path + '_synsets/'
        self.dir_path_lemmas = dir_path + '_lemmas/'
        self.dir_path_all_synsets = dir_path + '_all_synsets/'
        self.dir_path_all_lemmas = dir_path + '_all_lemmas/'
        self.check_if_exists()

        logging.info("===== Loading facebook model =====")
        self.model = load_facebook_model(fasttext_path)

    def check_if_exists(self):
        for directory in [self.dir_path_synsets, self.dir_path_lemmas,
                          self.dir_path_all_synsets, self.dir_path_all_lemmas]:
            os.makedirs(directory, exist_ok=True)

    def select_and_save_test_data(self, node_successors, graph):

        candidates_test_parents = select_candidates_test_parents(node_successors, graph)
        test_parents = select_test_parents(candidates_test_parents)
        test_lemmas = set([j.name() for i in test_parents for j in wn.synset(i).lemmas()])
        self.save_test_data(test_parents, test_lemmas)
        return test_parents, test_lemmas

    def save_test_data(self, test_parents, test_lemmas):
        save_test_data(test_parents, f"{self.dir_path_synsets}/synsets_deprived_of_parental_rights.txt")
        save_test_data(test_parents, f"{self.dir_path_lemmas}/synsets_deprived_of_parental_rights.txt")
        save_test_data(test_lemmas, f"{self.dir_path_synsets}/test_lemmas.txt")
        save_test_data(test_lemmas, f"{self.dir_path_lemmas}/test_lemmas.txt")

    def compute_and_save_data(self, dir_path, fn, data, lemma2id, pos, directed):
        node_triplets_lemmas, link_pairs_ids = fn(data, lemma2id, self.model, pos, directed)
        save_node(dir_path, node_triplets_lemmas)
        save_link(dir_path, link_pairs_ids)

    def compute_and_save_train_lemma_data(self, test_lemmas, pos, directed):
        all_lemmas = set(wn.all_lemma_names(pos))
        id2lemma = {i: lemma for (i, lemma) in enumerate(all_lemmas)}
        lemma2id = dict(map(reversed, id2lemma.items()))

        train_lemmas = all_lemmas.difference(test_lemmas)

        self.compute_and_save_data(self.dir_path_lemmas, compute_lemmas, train_lemmas, lemma2id, pos, directed)
        self.compute_and_save_data(self.dir_path_all_lemmas, compute_lemmas, all_lemmas, lemma2id, pos, directed)

    def compute_and_save_train_synset_data(self, test_parents, pos, directed):
        all_synsets = set([i.name() for i in wn.all_synsets(pos)])
        train_synsets = all_synsets.difference(test_parents)

        self.compute_and_save_data(self.dir_path_synsets, compute_synsets, train_synsets, None, pos, directed)
        self.compute_and_save_data(self.dir_path_all_synsets, compute_synsets, all_synsets, None, pos, directed)

    def create_dataset(self, pos, directed):

        logging.info("===== Loading graph data =====")
        graph, roots = get_graph_and_root(wn, "n", True)
        _, node_successors = get_successors(graph, 'entity.n.01', {})

        logging.info("===== Selecting and saving test data =====")
        test_parents, test_lemmas = self.select_and_save_test_data(node_successors, graph)

        logging.info("===== Computing train lemma data ======")
        self.compute_and_save_train_lemma_data(test_lemmas, pos, directed)

        logging.info("===== Computing train synset data ======")
        self.compute_and_save_train_synset_data(test_parents, pos, directed)
