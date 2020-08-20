"""Script to analyze the capture of dependencies by attention
Note : all the function in this script work under the assumption that sentences begin with a <S> token,
which is added if the model is a Transformer"""

import numpy as np
import extract_model as ex
import argparse
import datetime
from pickle import dump
from opennmt import models
from copy import  deepcopy
from contextlib import ExitStack
from attn_result import AttnResult

dependencies = ["nsubj", "obj", "amod","advmod"]
D = len(dependencies)
_temp = {}
for _i, _dep in enumerate(dependencies):
    _temp[_dep]=_i
dependencies = _temp #dict str(dependency) -> int
delta = 10

categories = {
    "gen": ["Masc", "Fem"],
    "nb" : ["Sing", "Plur"],
    "tense" : ["Imp","Past", "Pres", "Fut"],
    "voice" : ["Act", "Pass"]
}

testT  = False
testT2 = False

def np_softmax(A):
    """softmax of a 1D array"""
    A = np.exp(A)
    return A/np.sum(A)

def get_freq(file):
    """returns a dict {token -> rank in rarity (i.e. number of occurences in the corpus)} as saved in file"""
    res = {}
    for line in file:
        t,i = line.strip('\n').split(" ")
        if t in res:
            raise ValueError(f"Token {t} is repeated in {file.name}")
        res[t] = int(i)
    return res


def get_rarest(freq, tokens):
    """freq : number of occurences"""
    l=[1] #0 is <S>
    f=freq[tokens[1]] if tokens[1] in freq else 0
    for i in range(2,len(tokens)):
        f2 = freq[tokens[i]] if tokens[i] in freq else 0
        if f2 < f:
            f = f2
            l=[i]
        elif f2 == f:
            l.append(i)
    return l

def switch(attn):
    """To be used if the model uses EOS rather than BOS
    Reshapes [L,H,S,S] so that last line and columns become first"""
    result = np.zeros(attn.shape)
    result[:,:,0,0] = attn[:,:,-1,-1]
    result[:,:,1:,0] = attn[:,:,:-1, -1]
    result[:,:,0,1:] = attn[:,:,-1, :-1]
    result[:,:,1:,1:] = attn[:,:,:-1,:-1]
    return result

def pad_attn(attn):
    """To be used if the model uses no special token.
    Adds a column and a line of -inf """
    l,h,s,_ = attn.shape
    s+=1
    result = np.zeros((l,h,s,s))
    result[:, :, :, 0] = np.ones((s)) * -np.inf
    result[:, :, 0, :] = np.ones((s)) * -np.inf
    result[:, :, 1:, 1:] = attn
    return result




def step(attention, tokens, deps, head_table, rarest, cats, nbest=1):
    """Args:
        attention : the attention matrix over the sentence.
        head_table : list containing at i the span (j,k) of the tokens consituting the head of token i
        rarest: list of indices of the rarest tokens. Usually, only one index
            (may be several if the rarest token is present multiple times, or if several tokens have same 'rarity')
        cats : the dict of list of labels for each category

    returns: score_dep :an array [D,2] containing for each dependency,for both directions,
                the number of captured occurences
            score_pos : an array [2] containing the number of captured tokens in -1 or +1
            score_rare : the number of captured tokens that are the rarest token
            random_score : the same that previous array but with a token chosen at random
            cats_score : a dict of dict of ndarrays, such that cats_score[category][label][d, i, j] counts the number
                of captured dependencies d with the label in the category, in the direction i. j indicates whether the
                label is on the tail or on the head of the dependency.

    note : captured = among the nbest most attended to.
     """
    score_dep = np.zeros((D,2))
    score_pos = np.zeros((2,))
    score_rare= 0
    attn_distr= np.zeros((delta*2+1,))
    cats_score= {k : {l : np.zeros((D, 2,2)) for l in v} for k,v in categories.items()}
    for i in range(1, len(tokens)):
        best = np.argpartition(attention[i,1:], -nbest)[-nbest:] +1 #A[i,j] = attention of i over j, ie w_i+=\sum_j(A_i,j * w_j), .we do not consider <s>
        for j in best:
            #reverse dependency j->i (i is head of j): j bears the dependency tag
            tag = deps[j].split(":")[0] #removes any subtags (e.g. nsubj:caus)
            if tag in dependencies:
                if head_table[j][0] <= i <= head_table[j][1]:
                    dep_i = dependencies[tag]
                    score_dep[dep_i,1]+=1
                    for k, tab in cats.items(): #labels for each category
                        label = tab[j]
                        res_tab = cats_score[k].get(label)
                        if res_tab is not None:
                            res_tab[dep_i,1,0]+=1 #label on tail
                        label = tab[i]
                        res_tab = cats_score[k].get(label)
                        if res_tab is not None:
                            res_tab[dep_i, 1, 1] += 1  # label on head
            # count direct dependencies i->j (j is head of i)
            tag = deps[i].split(":")[0]  # removes eventual subtags (e.g. nsubj:caus)
            if tag in dependencies:
                dep_i = dependencies[tag]
                if head_table[i][0] <= j <= head_table[i][1]:#count direct dependencies i->j (j is head of i)
                    score_dep[dep_i,0]+=1
                    for k, tab in cats.items(): #labels for each category
                        label = tab[i]
                        res_tab = cats_score[k].get(label)
                        if res_tab is not None:
                            res_tab[dep_i,0,0]+=1 #label on tail
                        label = tab[j]
                        res_tab = cats_score[k].get(label)
                        if res_tab is not None:
                            res_tab[dep_i, 0, 1] += 1 #label on head
            if j in rarest:
                score_rare+=1
            if j == i-1 and j!= 0: # do not count EOS/BOS as a +/-1
                score_pos[0]+=1
            elif j == i+1:
                score_pos[1]+=1
            if abs(j-i) <= delta:
                attn_distr[delta + j-i ]+=1

    return score_dep, score_pos, score_rare, attn_distr, cats_score

def step_distrib(attention : np.ndarray, d=10):
    """Step for the analyze of attention distribution on the d most attended tokens"""
    l,_ = attention.shape #sentence length
    result = np.zeros((l,d))
    result_softmax = np.zeros((l,d))
    for i,line in enumerate(attention): #iterates over whole lines to have a correct softmax
        line_sm = np_softmax(line)
        result[i,:] = np.flip(np.sort(np.partition(line[1:], -d)[-d:])) #takes d best, sorts them, then reverse them
        result_softmax[i,:] = np.flip(np.sort(np.partition(line_sm[1,:], -d)[-d:]))
    return result, result_softmax #shape : (S,d)

def step_text(deps, head_table,  cats_labels):
    """Computes scores that do not depend of attention : relative position frequency,
    number of occurences, frequency of heads for each dependency,
    random baselines for dep. detection and for positional focus """
    score_dep = np.zeros((D, 2))  # counts the number of dependencies encountered (with repetitions)
    score_pos = np.zeros((D, 2*delta+1))
    count_baseline=np.zeros((D,)) # counts the number of heads for each dependency
    cats_occ = {k:{l: np.zeros((D,2)) for l in v} for k,v in categories.items()} #counts, for each label of each cat,
        # the number of dependencies where the tail/head has the label
    n = len(head_table) -1
    pos_baseline = np.array([max(0, n-abs(d_i)) for d_i in np.arange(-delta, delta+1)])
        # just counts the number of tokens at each relative pos of each other
    for i in range(1, len(head_table)):
        a,b = head_table[i]
        tag = deps[i].split(":")[0]  # removes any subtag (e.g. in nsubj:caus)
        if tag in dependencies:
            dep_i = dependencies[tag]
            score_dep[dep_i, :] += np.array([1, 1])  # count all actual dependencies
            for k,line in cats_labels.items(): #cats count for label on tail
                label = line[i]
                A = cats_occ[k].get(label)
                if A is not None:
                    A[dep_i,0]+=1
            count_baseline[dep_i] += b-a+1
            for l in range (a, b+1):  # iterates over subword tokens composing head of i
                if abs(l - i) <= delta:
                    score_pos[dep_i, delta + i-l] += 1
                for k, line in cats_labels.items():  # cats count for label on head
                    label = line[l]
                    A = cats_occ[k].get(label)
                    if A is not None:
                        A[dep_i,1] += 1
    return score_dep, score_pos, count_baseline/len(head_table), pos_baseline/n, cats_occ

def get_heads(heads, t=None, get_words=False):
    """Returns a list l such that, for i the index of a token, l[i] is (j,k) where j is the index of the first token of i.head and k is the last
    If get_words, also returns the lists words -> (start token, end token) and token -> word
    t is used for debbuging, it is just printed in case of an error"""
    scan = [(0,0)] #scan is the word reconstitution (word -> (start token, end token) )
    #the <s> will be added twice and removed once
    cur_l = [0]
    try :
        for i, h in enumerate(heads):
            if "&" in h:# marks places where a word lacks BEFORE the token
                scan.append(scan[-1])
            if ";" in h: #new word
                cur_l.append(i)
                scan.append(tuple(cur_l))
                cur_l=[i+1]
            if h[0:2] == "-1":
                scan.pop() #was actually a non word token
            if "~" in h: # marks places where a word lacks AFTER the token
                scan.append(scan[-1])
        cur_l.append(len(heads)-1)
        scan.append(tuple(cur_l))
        if h[0:2] == "-1":
            scan.pop()  # was actually a non word token
        scan.append((None,None)) #for -1 pointing tokens
        l = []
        for h in heads:
            i = int(h.strip("~;&"))
            l.append(scan [i])
            assert scan[i] != (None, None) or i==-1
        if testT:
            raise Exception("Test")

    except Exception as e:
        # debugging info
        print("Error in heads : ")
        print(e)
        print(t)
        print(len(heads), heads)
        print(len(scan),scan)
        for i, (a, b) in enumerate (zip(l, t)):
            print(f"{i}{a}:{b}", end=" ")
        print("\n")
        for i, (a, b) in enumerate (zip(heads, t)):
            print(f"{i}:{a}{b}", end=" ")
        print("\n")
        for i, a in enumerate (scan):
            print(f"{i}:{a}", end=" ")
        print("\n")
        raise e
    if get_words:
        scan_inverse = []  # token -> word . Used if get_words
        for i, (a,b) in enumerate(scan):
            for _ in range(a,b+1):
                scan_inverse.append(i)
        return l, scan, scan_inverse
    return l


def main(modelpath, cfg,tokensf,headsf, depf, frequencies, freq_save, dest_save, cats=None, cats_file = None):
    model = ex.get_model(modelpath, cfg, model_type=models.TransformerBase)
    ds = ex.get_dataset(model, tokensf,mode='inference' )
    L,H = model.num_layers, model.num_heads
    if isinstance(model, models.LanguageModel ):
        BOS = True
    elif isinstance(model, models.SequenceToSequence):
        BOS = False
    else:
        raise ValueError(f"Model type {type(model)} is not recognized")
    result = AttnResult(L,H, dependencies, dmax=dmax, BOS = BOS, cats=cats)
    count_skipped = 0
    i = 0
    err=0
    with ExitStack() as exit_stack:
        headsr = exit_stack.enter_context(open(headsf))
        depr = exit_stack.enter_context(open(depf))
        cats_readers = {k : exit_stack.enter_context(open(f)) for k,f in cats_file.items()}
        for data, deps, heads in zip(ds,depr, headsr):
            if len(heads)>3000:
                count_skipped+=1
                continue #avoid OOM on a solo sentence by skipping overlong sentences
            cats_lines = {k: ["-1"] + r.readline().strip("\n").split(" ") for k,r in cats_readers.items()}
            if not test_text:
                tokens, attn = model(data, return_attn = True, training = False)
                tokens = list(tokens["tokens"].numpy()[0])
                tokens = ["<s>"] + [i.decode() for i in tokens]
                if not BOS:
                    attn = pad_attn(attn)
            else :
                tokens = ["None" for _ in deps]
            heads = ["-1;"] + heads.strip("\n").split(" ")
            try :
                head_table = get_heads(heads, t=tokens)
            except Exception as e:
                err += 1
                print(i)
                if testT:
                    raise e
                continue
            deps = ["<s>"] + deps.strip("\n").split(" ")

            if test_distr and len(heads) < dmax +1:
                count_skipped+=1
                continue
            assert len(tokens) == len(deps) == len(heads) #sanity check
            result.N_words += len(tokens) #counting tokens
            try :
                all_dep, temp2, temp_bsl, temp_pos_bsl, cats_occ = step_text(deps, head_table, cats_lines)
                result.distr_dep += temp2 # evaluates frequency of dependencies in relative position -delta to delta (delta = 10)
                result.random_baseline +=temp_bsl
                result.distr_baseline +=temp_pos_bsl
                result.add_cats_occ(cats_occ)
                if test_text: #sped-up version without attention
                    result.score_dep[0, 0, :, :, 0] += all_dep  # all occurences
                else:
                    rarest = get_rarest(frequencies, tokens)
                    result.rarity_baseline+=len(rarest)
                    for l in range(L):
                        for h in range(H):
                            if test_distr:
                                seq_distr, seq_distr_sm = step_distrib(attn[l,h,:,:], d=dmax )
                                result.distr = np.concatenate((result.distr, seq_distr ))
                                result.distr_sm = np.concatenate((result.distr_sm, seq_distr_sm))
                            temp_dep, temp_pos, temp_rare, temp_attn_distr, cat_score = step(attn[l, h, :, :],
                                                                                                 tokens, deps,
                                                                                                 head_table, rarest, cats_lines)
                            result.score_dep[l,h,:,:,1] += temp_dep  #captured occurences for dep
                            result.score_dep[l,h,:,:,0] += all_dep #all occurences
                            result.score_rare[l,h] += temp_rare #counts captured tokens
                                # that are the rarest in sentence [L,H]
                            result.score_pos[l,h,:] += temp_pos #counts captured tokens at position -1 and 1 [L,H,2]
                            result.distr_attn += temp_attn_distr #counts relative distrib of max attn. [2delta+1]
                            result.add_cat_score(cat_score, l,h)

            except Exception as e:
                print(i, datetime.datetime.now())
                print(tokens)
                print(deps)
                print(head_table)
                print(heads)
                print(attn.shape)
                raise e

            if i % 5000 == 0:
                print_log(i, tokens, deps, head_table, count_skipped, err)
            i += 1
            if not i % freq_save :
                save_result(result, dest_save, result.score_dep,i, count_skipped)

    if test_distr:
        print(f"Skipped {count_skipped} lines shorter than {dmax} on a total of {i+count_skipped}.")
    result.N_sent = i - count_skipped - err
    # normalization of baselines
    result.random_baseline /= result.score_dep[0,0,:,0,0]
    result.distr_baseline  /= (result.N_words - result.N_sent)
    result.rarity_baseline /= (result.N_words - result.N_sent)

    print(f'{err} errors and {count_skipped} skipped lines on {i+count_skipped+err} lines')
    return result

def print_log(i, tokens, deps, head_table, skipped, err):
    print(datetime.datetime.now(), i, "skipped : ", skipped,", errors : ", err)
    print(tokens)
    print(deps)
    for a, (b, c) in enumerate(zip(tokens, head_table)):
        print(f"{b}:{c}{a}", end=" ")
    print("\n", flush=True)


def save_result(result : AttnResult, dest_save, score,n, count_skipped):
    result_c = deepcopy(result)
    result_c.N_sent = n - count_skipped
    result_c.random_baseline /= score[0, 0, :, 0, 0]
    result_c.distr_baseline /= (result_c.N_words - result_c.N_sent)
    result_c.rarity_baseline /= (result_c.N_words - result_c.N_sent)
    with open(dest_save, "wb") as writer:
        dump(result_c, writer)


def add_dependencies(dep : dict,l):
    """Adds supplementary dependencies from l to dep.
    Returns the total number of dependencies"""
    i_0 = len(dep.values())
    for i,dep_to_add in enumerate(l):
        dep[dep_to_add] = i+i_0
    return  len(dep.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tokens")
    parser.add_argument("heads")
    parser.add_argument("dep_file", help="dependencies")
    parser.add_argument("dest")
    parser.add_argument("--dep", help = "Dependencies to add to the list of studied dependencies", nargs="+")
    parser.add_argument("-T","--test", action="store_true")
    parser.add_argument("--text", action="store_true", help="Do not calculate the data on attention, only those depending on the text")
    parser.add_argument("--distr", help="Aslo returns the values of attention for the d-th more attended values for each word"
                                        "The point is to study the relative weight of the more important tokens",
                        type = int, default=None)
    parser.add_argument("--nmax", help="Take the n best instead of only the best. Beware : it's experimental",
                        default=1)
    parser.add_argument("-M", help="Model to use", required=True)
    parser.add_argument("-R", help="Rarity file", required=True)
    parser.add_argument("--cfg", help="Config file")
    parser.add_argument("-f", help="Dumps saves this often", default=5000, type = int)
    for k in categories.keys():
        parser.add_argument(f"--{k}", help=f"File for {k} category.")
    args = parser.parse_args()
    print(vars(args))
    testT = args.test
    categories = {k : v for k,v in categories.items() if vars(args)[k]}
    test_text = args.text
    test_distr = args.distr is not None
    if test_distr :
        dmax = args.distr
    else :
        dmax = 0
    if args.dep:
        D = add_dependencies(dependencies, args.dep)
    modelpath = args.M

    with open(args.R) as reader:
        frequencies = get_freq(reader)
    #main
    result = main(
        modelpath, args.cfg, headsf=args.heads, depf=args.dep_file,tokensf=args.tokens,
        frequencies=frequencies, freq_save=args.f, dest_save= args.dest+".save",
        cats = categories,cats_file = {k: vars(args)[k] for k in categories.keys()}
    )
    with open(args.dest, "wb") as writer :
        dump(result, writer)









