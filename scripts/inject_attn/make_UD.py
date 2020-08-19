"""Script to produce UD_like documents from any french text using spacy
Spacy doesn't use UD tag scheme for german or english..."""

import spacy
import argparse
from contextlib import ExitStack
from datetime import datetime


spacy.prefer_gpu()

spacy_models = { "fr" :"fr_core_news_md" } #"de":"de_core_news_md",

cor_fr = {} #what to replace to go from spacy to UD. Useless for french, might be useful for other languages.
# But it might also be more complicated than that.

corresp_list = {"fr":cor_fr}


def dep_info(sent, corresp):
    """Returns dependencies and heads, as they would have been extracted from the UD format"""
    dep = []
    heads = []
    orth = []
    for word in sent:
        ud = corresp.get(word.dep_)
        if ud is None:
            ud = word.dep_
        dep.append(ud)
        heads.append(word.head.i+1 if word.head.i != word.i else 0)
        orth.append(word.orth_)
    return dep, heads, orth

def clean_reader(reader):
    """Iterator to give to Spacy's pipeline.
    Removes spaces that spacy sometimes regards as words."""
    for line in reader:
        yield line.strip(" \n")

def split_alpha(word: str,head, sup: list):
    """Split word in contiguous sequences of alphabetic or non-alphabetic characters"""
    A = [a.isalpha() for a in word]
    temp = [word[0]]
    result = []
    heads = []
    sups = [[] for s in sup]
    for i in range(len(word)-1):
        if A[i] != A[i+1]:
            append(result, temp,heads, head, sups,sup)
            temp = []
        temp.append(word[i+1])
    append(result, temp, heads, head, sups, sup)
    return  result, heads, sups

def cased(word: str,head, sup : list):
    """Returns the word, with pseudowords associated that correspond to tokens that would appear.
    sup is the list of supplementary categories (a str each) to align  (only dep a priori)"""
    A = [a.isalpha() for a in word]
    if not any(A):
        return  [word]
    assert all(A)
    heads = []
    sups = [[] for s in sup]
    token_sup = list("<upper>")
    caps = [(not a.islower()) for a in word ]
    if caps[0] and not any (caps[1:]):
        return ["<maj>", word]
    result = []
    temp = [word[0]]
    if caps[0]:
        append(result, token_sup, heads, -1, sups, "###")
    for i in range(len(caps)-1):
        if caps[i] != caps[i+1]:
            append(result, temp, heads, head, sups, sup)
            append(result, token_sup, heads, -1, sups, "###")
            temp = []
        temp.append(word[i + 1])
    append(result, temp, heads, head, sups, sup)
    if caps[-1] :
        append(result, token_sup, heads, -1, sups, "###")
    return result, heads, sups

def append(result, chars, heads, head, sups, sup):
    result.append("".join(chars))
    heads.append(head)
    for s, s0 in zip(sups, sup):
        s.append(s0)

def split_and_prealign(words, heads, sup):
    """Splits the words in alphabetic/non-alphabetic sequences, add case markup pseudo words,
    while keeping heads and sup aligned"""
    temp_words, temp_heads = [], []
    temp_sup = [[] for _ in sup]
    for w, h, s in zip(words, heads, sup):
        w,h,s = split_alpha(w,h,s)
        for w2, h2, s2 in zip(w,h,s):
            w2,h2,s2=cased(w2, h2, s2)
            temp_words.extend(w2)
            temp_heads.extend(h2)
            for sup_i, s2_i in zip(sup, s2):
                sup_i.extend(s2_i)
    return temp_words, temp_heads, temp_sup

def sent_it(pipe):
    """Takes a pipe (iterator over docs) and returns an iterator over sentences of said docs."""
    for doc in pipe:
        if doc.ort_.count('-') < 20 and doc.orth_.count("=") < 20:  # roughly, to remove tables
            for sent in doc.sents:
                yield sent.as_doc()

def main(reader, w_h, w_d, w_t, lang, w_cats, cats, w_s = None):
    """w_s : writer for sentences"""
    assert len(w_cats) == len(cats)
    print("Loading model...", end=" ", flush=True)
    nlp = spacy.load(spacy_models[lang], disable=["ner", "textcat", "entity-ruler"])
    corresp = corresp_list[lang]
    print("Done")
    pipe = nlp.pipe(clean_reader(reader))
    if args.S :
        pipe = sent_it(pipe)

    for i,sent in enumerate(pipe) :
        dep,heads,orth = dep_info(sent, corresp)
        # orth = remove_spaces(orth) #space entities will be safely ignored by align pos
        w_h.write(" ".join(str(h) for h in heads) + "\n")
        w_d.write(" ".join(dep) + "\n")
        w_t.write("_".join(orth)+"\n")
        for cat, writer in zip(cats, w_cats):
            tags = get_tag(sent, r_tags[cat])
            writer.write(" ".join(tags)+"\n")
        if w_s is not None:
            w_s.write(sent.text+"\n")
        if i%50_000 == 0 or args.V:
            print_log(i,dep, heads, sent)
        if args.N is not None and i==args.N -1 :
            break
    return i

def remove_spaces(sent):
    l=[]
    for w in sent:
        w= w.strip(" ")
        if w:
            l.append(w)
    return l

def print_log(i,dep,heads, sent):
    print(datetime.now(), i)
    print("\t".join(t.orth_ for t in sent))
    print("\t".join(dep))
    print("\t".join(str(t) for t in heads))
    print(flush=True)

def get_tag(sent, r_tag):
    """Returns the line of tags for the category r_tag"""
    s = ["."]*len(sent)
    for i,t in enumerate(sent):
        if t.tag_ in ["_SP","X"]:
            continue
        tags = t.tag_.split("|") #eg ADJ__Gender=Fem|Number=Sing|NumType=Ord ; SYM
        if len(tags) == 1:
            continue
        POS, tags[0] = tags[0].split ("__") #eg : VERB__Mood=Cnd
        for tag in tags:
            k, v = tag.split("=")
            if k == r_tag:
                s[i] = v
                break
            if r_tag == "Voice" and POS == "VERB": #if we're searching for verb voice and it isn't passive, then it's active
                s[i] = "Act"
    return s


cats = ["gen", "nb", "tense", "voice"]
r_tags = {"gen" : "Gender", "nb":"Number", "tense":"Tense", "voice":"Voice"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser("A script to get syntactic info from spacy on unannotated text. "
                                     "French only for now\n"
                                     "Will write files for: heads of words (tgt.heads), dependencies of words (tgt.dep)"
                                     ", words separated by _ (entities.tgt)(useful, because some 'words' may "
                                     "contain spaces), labels of words for each given category (tgt.cat), "
                                     "and Spacy-separated sentences if -S if given (tgt.sentences)")
    parser.add_argument('src', help="source")
    parser.add_argument('tgt', help="Destination prefix")
    parser.add_argument('-V', help="Outputs all logs", action="store_true")
    parser.add_argument('-S', help="Separate sentences", action="store_true")
    parser.add_argument('-N', help = "Number of sentences to process", type=int, default=None)
    parser.add_argument("--cats",help=f"Categories to get labels of. Allowed : {cats}", nargs="+")

    args = parser.parse_args()
    with ExitStack() as exit_stack:
        reader = exit_stack.enter_context(open(args.src))
        writer_h = exit_stack.enter_context(open(args.tgt+".heads", "w"))
        writer_d = exit_stack.enter_context(open(args.tgt+".dep",   "w"))
        writer_t = exit_stack.enter_context(open(args.tgt+".entities","w"))
        writer_cats = []
        for suf in args.cats:
            assert suf in cats
            w = exit_stack.enter_context(open(args.tgt + "." + suf + ".cat", "w"))
            writer_cats.append(w)
        writer_s = None
        if args.S:
            writer_s = exit_stack.enter_context(open(args.tgt+".sentences", "w"))
        i = main(reader, writer_h, writer_d, writer_t, "fr", writer_cats, args.cats, writer_s)
        print(f"Done. Written {i} lines")
