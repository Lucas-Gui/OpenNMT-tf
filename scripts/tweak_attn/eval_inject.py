"""Script to evaluates the changes brought by attention injection"""

from argparse import ArgumentParser
from contextlib import ExitStack
import spacy
from pprint import pprint
import os

gen_test = 0


# Matching of german Spacy labels with UD : functions that take a spacy.Token object and returns a boolean indicating
# whether the token is the tail of the corresponding UD dependency
dep_de_UD = {
    "nsubj" : lambda tok : tok.dep_ in ["sb","sbp","sp"],
    "amod" : lambda tok : tok.pos_ == "ADJ" and tok.dep_ in ["adc", "nk"],
    "obj" : lambda tok : tok.dep_ in ["oa","oa2"],
    "det" : lambda tok : tok.pos_ == "DET" and tok.dep_ == "nk",
}

def get_pairs(doc,dep):
    """Returns the set of pairs (tail, head) of the given dep"""
    r = set()
    for t in doc:
        if dep_de_UD[dep](t):
            r.add((t.orth_, t.head.orth_))
    return r

def remove_nb_placeholders(sent):
    """Replace all numeral placeholders by numbers, to help spacy's work"""
    if isinstance(sent, str):
        sent = sent.split(" ")
    for i, w in enumerate(sent):
        if "\uff5f" in w:
            sent[i] = str(1980 + i%25)
    return " ".join(sent)

# eval functions :
# should take two Doc objects as input
# and return two booleans : one for the test itself, the other to check whether the test applies

def identity_eval(orig, modif):
    return orig.text.strip(" ") != modif.text.strip(" "), True


def make_change_dep_eval(dep,):
    """Produce an evaluation function that returns True if any change has happened on any occurence of the given dep"""
    def change_dep_eval(orig,modif):
        s1 = get_pairs(orig, dep)
        s2 = get_pairs(modif, dep)
        return s1 != s2, bool(s1)
    return change_dep_eval

def main(file1, file2, tests, writers, source=None):
    print(f"Reading from {file1.name} and {file2.name}")
    scores = {k:0 for k in tests.keys()}
    counts = {k: 0 for k in tests.keys()}
    pipe1 = nlp.pipe(clean_iter(file1))
    pipe2 = nlp.pipe(clean_iter(file2))
    i = i_tot = 0
    for (line1, line2) in (zip(pipe1, pipe2)):
        i_tot+=1
        if not line2.text :
            if gen_test:
                print(i_tot)
                print(line1)
                print(line2)
            continue
        i+=1
        id_t, _ =  identity_eval(line1,line2) #could be more efficient but ok
        for k,test_f in tests.items():
            t, is_applicable = test_f(line1, line2)
            if not is_applicable:
                continue
            counts[k] += 1
            if not (t and id_t):
                continue
            scores[k]+=1
            if source is not None:
                writers[k].write( source.readline() )
            writers[k].write("\n".join([line1.text, line2.text])+"\n\n")
        if not i%5000 or gen_test:
            print_log(i_tot,line1,line2, scores, counts)
            if gen_test and i_tot > gen_test:
                break
    print(f"Done with {i} valid sentences on a total of {i_tot}. Scores:")
    pprint(scores)
    counts["total"]=i_tot
    return scores, counts

def print_log(i, line1, line2, scores, counts):
    print(i)
    print(line1)
    print(line2)
    pprint(scores)
    pprint(counts)
    print(flush=True)


def clean_iter(file):
    """Iterates over lines, removing EOS and replacing numeral placeholders"""
    for line in file:
        line = line.strip("\n ").split(" ")
        while "</s>" in line:
            line.remove("</s>")
        line = remove_nb_placeholders(line)
        yield line


if __name__ == '__main__':
    with ExitStack() as exit_stack:
        openr = lambda x: exit_stack.enter_context(open(x))
        openw = lambda x: exit_stack.enter_context(open(x, "w"))
        parser = ArgumentParser()
        parser.add_argument("orig", help="Original translation file", type=openr)
        parser.add_argument("comp", help="Modified translation file", type=openr)
        parser.add_argument("dest", help="Folder to puts results to")
        parser.add_argument("--source", help="Source text", type=openr, default=None)
        for k in dep_de_UD.keys():
            parser.add_argument(f"--{k}", help=f"Adds test on {k} ", action="store_true")
        parser.add_argument("--test",type = int, default = 0)
        args = parser.parse_args()
        nlp = spacy.load("de_core_news_md")
        gen_test = args.test
        try:
            os.mkdir(args.dest)
        except OSError:
            pass

        tests = {"identity" : identity_eval}
        writers={"identity" : openw(os.path.join(args.dest,"id"))}
        for k in dep_de_UD.keys():
            if vars(args)[k] :
                tests[f"test_change_{k}"] = make_change_dep_eval(k)
                writers[f"test_change_{k}"] = openw(os.path.join(args.dest,f"test_change_{k}"))
        scores, counts = main(args.orig, args.comp, tests, writers, args.source)
        with open(os.path.join(args.dest, "info.txt"), "w") as info:
            info.write(f"Tests on files {args.orig.name} and {args.comp.name}\n")
            for k in tests.keys():
                scores[k+" (frac)"]= scores[k]/ counts[k]
            info.write("Scores : \n")
            pprint(scores, info)
            info.write("Counts : \n")
            pprint(counts, info)


