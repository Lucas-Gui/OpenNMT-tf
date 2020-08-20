import numpy as np

class AttnResult():
    """A class just to store the results of analyze_attn_dep"""

    def __init__(self, L, H, dependencies, dmax=0, BOS=False, cats=None, delta=10):
        D = len(dependencies)
        self.score_dep = np.zeros((L, H, len(dependencies), 2, 2))  # The  [L,H,D,2,2] array counting the score.
        # score[l,h,d,s,0] count all encountered dependencies,  [l,h,d,s,1] count captured dependencies
        # score[l,h,d,s,i] mark 'positive' direction dependencies (->), [l,h,d,1,i] mark inverse direction (<-)
        self.distr_dep = np.zeros(
            (D, 2 * delta + 1))  # evaluates frequency of dependencies in relative position -delta to delta (delta = 10)
        self.distr_attn = np.zeros((2 * delta + 1,))  # evaluates frequency of max attn in each place
        self.random_baseline = np.zeros((D,))
        self.distr_baseline = np.zeros((2 * delta + 1,)) # probability of choosing a token at a given relative position.
        self.rarity_baseline = 0  # probability of choosing the rarest token in a sentence
        self.score_pos = np.zeros((L, H, 2))  # counts captured tokens at position -1 and 1
        self.score_rare = np.zeros((L, H))  # counts captured tokens that are the rarest in sentence
        self.N_words = 0  # number of words (<s> or </s> included)
        self.N_sent = 0  # number of sentences
        # distribution of attention weight as a function of relevance (of rank in attention)
        # will not necessarily be updated
        self.distr = np.zeros((0, dmax))  # will be (L*H,dmax) at the end
        self.distr_sm = np.zeros((0, dmax))
        self.sup_token = "<s>" if BOS else None

        self.dependencies = list(dependencies.keys())  # the studied dependencies

        self.cats = cats or {}
        self.cats_scores = {k: {l: np.zeros((L, H, len(dependencies), 2, 2)) for l in v} for k, v in cats.items()}
        # L,H,D,2,2 : counts, for each label of each category, the number of captured dependecies where the tail or head has the label
        # (eg, for nb, sing, all occurences of sing subjects attending verbs and of sing subjects attended by their verbs)
        # l,h,dep,i,0 is for label at foot, l,h,dep,i,1 is for label at head
        self.cats_occurences = {k: {l: np.zeros((len(dependencies), 2)) for l in v} for k, v in cats.items()}
        # idem but counts all occurences

    def add_cat_score(self, score, l, h):
        for k, labels in self.cats.items():
            for lab in labels:
                self.cats_scores[k][lab][l, h, :, :, :] += score[k][lab]

    def add_cats_occ(self, count):
        for k, labels in self.cats.items():
            for lab in labels:
                self.cats_occurences[k][lab] += count[k][lab]