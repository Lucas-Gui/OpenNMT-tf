"""Script to test the injection of info into the model, for different tasks."""

import analyze_attn_dep as aad #for get_heads, categories
from argparse import ArgumentParser
from contextlib import ExitStack
import numpy as np
import extract_model as ex
from opennmt import models
from random import gauss
from datetime import datetime
import os
from pyonmttok import Tokenizer
from pprint import pprint

rng= np.random.default_rng()

gen_test= False




def get_stats(folder):
    """Loads statistics as saved by measure_attn.py
    Returns a dictionary, with arrays of shape(L,H) as items. """
    r = {}
    for name in ("mean","std", "quantiles.list"):
        r[name] = np.load(folder+"/"+name+".npy")
    quantiles = np.load(folder+"/quantiles.npy")
    for d, A in zip(r["quantiles.list"], quantiles):
        r[d] = A
    return r

def get_main(heads ):
    """Returns the indices of the tokens of the head of the sentence, or None if the sentence has 0 or several heads
    Not in use for now, but could be useful to design tasks"""
    r = []
    s1 = False
    for i,h in enumerate(heads.split(" ")):
        l = int(h.strip("&;~"))
        if l == 0 :
            if s1 : # checks for tokens with head 0 after the end of first word
                return None
            s1 = s1 or ";" in h #";" marks the end of a word
            r.append(i) #account for multi-tokens words
    return r or None # returns None if r is empty

def switch_tup(tup1, tup2):
    """takes (a,b) , (c,d), returns (a,d), (c,b)"""
    return (tup1[0], tup2[1]) , (tup2[0], tup1[1])

def get_dep(dep, dep_line, heads, head_table,n=1, head_i = None):
    """Returns a list of tuples of (tail tokens, head tokens) for the given dep
    Returns only the n first encountered.
    If the sentence has no such dep, or is invalid, returns None
    If head_i is given, only returns tokens whose head is head_i """
    r = []
    tail = []
    head = []
    for i,(d,h ) in enumerate(zip(dep_line, head_table)):
        if d == dep and (head_i is None or h[0]<=head_i<=h[1]):
            h = tuple([u for u in range(h[0], h[1]+1)]) #(i_min, imax) -> (i_min, ..., i_max)
            tail.append(i)
            if not head :
                head = h
            elif head != h:
                if gen_test:
                    print(head, h)
                    raise ValueError("Different heads for tokens of a same word")
                else:
                    return None
            if ";" in heads[i]:
                r.append((tail, head))
                if len(r) == n:
                    if gen_test:
                        print("get_dep:")
                        print(dep, dep_line)
                        print(r)
                    return r
                tail, head = [],[]
    return None #has not met enough deps.

def main(tasks ):
    """Parcourt les fichiers d'entrée, et exécute les différentes taches sur chaque ligne
    Signature of a task : (model, data, deps, heads, head_table, {cats: cat_line}, stats) -> output of model"""
    model = ex.get_model(args.M, args.C, model_type=models.TransformerBase)
    print(f"Reading {args.corp}. Writing to {args.dest}. Printing logs every {delta_t}.")
    print(f'Amplification factor is {ampli}')
    if gen_test :
        print("Testing on.")
    print()
    pprint(model.params)
    print()
    ds = ex.get_dataset(model, args.corp, mode='inference')
    #tokenizer
    tokenizer = Tokenizer(
        "aggressive",
        case_markup = True,
        joiner_annotate = True,
    )
    n_err = 0
    with ExitStack() as exit_stack:
        #readers and writers
        control_w = exit_stack.enter_context(open(args.dest+"/vanilla", "w"))
        source_w = exit_stack.enter_context(open(args.dest+"/source", "w"))
        task_writers = {k : exit_stack.enter_context(open(args.dest+"/"+k, "w")) for k in tasks.keys()}
        cat_readers = {k : exit_stack.enter_context(open(vars(args)[k])) for k in aad.categories.keys() if vars(args)[k] is not None }
        dep_r = exit_stack.enter_context(open(args.dep))
        heads_r = exit_stack.enter_context(open(args.head))
        #body : loop over data
        for i,(data, dep_line, heads_line )in enumerate(zip(ds, dep_r, heads_r)):
            #lines. Adjust everything as if BOS was there, for heads_table
            cat_lines = {k : ["-1"]+r.readline().strip("\n").split(" ") for k,r in cat_readers.items()}
            dep_line = ["<S>"] + dep_line.strip("\n").split(" ")
            dep_line  = [dep.split(":")[0] for dep in dep_line]
            heads_line = ["-1;"] + heads_line.strip("\n").split(" ")
            try :
                heads_table = aad.get_heads(heads_line, list(data["tokens"]))
            except (AssertionError, IndexError) as e:
                n_err+=1
                print(f"Error in get_heads on line {i}\n{decode_tokens(data, tokenizer)}\n")
                continue
            #doing the tasks
            if gen_test:
                print("data", " ".join([t.decode() for t in data["tokens"].numpy().flat]))
            for k,task_f in tasks.items():
                if gen_test:
                    print(f"\nTask {k}")
                output = task_f(model, data, dep_line, heads_line, heads_table, cat_lines )
                if output is not None:
                    output = decode_tokens(output, tokenizer)
                    task_writers[k].write(output+"\n")
                else:
                    task_writers[k].write("\n") #writes empty lines where task doesn't apply
                if gen_test :
                    print(output)
            #control : translation with no task
            _,output = model(data, training=False)
            output = decode_tokens(output, tokenizer)
            control_w.write(output +"\n")
            if gen_test:
                print("\n",output)
            source_w.write (decode_tokens(data, tokenizer) + "\n")
            if not i %delta_t or (gen_test):
                print_log(i, data, tokenizer, n_err)
    print(f"Finished with {n_err} errors.")


def print_log(i, data, tokenizer, n_err):
    print(datetime.now(), i, flush=True)
    print(f"{n_err} errors so far.")
    print(decode_tokens(data, tokenizer))
    print(flush=True)

def decode_tokens(model_output, tokenizer):
    return tokenizer.detokenize([t.decode() for t in model_output["tokens"].numpy().flat])


def make_inject_target(L,H, S, dep_tup, direction, heads, value_generator, inject=None):
    """Return a inject tuple with values generated by value_generator for the (tail and/or head) of the given pairs
    in dep_tup in the given heads
    dep_tup : list of tuples ([indices of tail], [indices of heads])
    Direction : 2-list in {0,1}
    value_generator : (l,h) -> float
    If inject is given, overwrites it where modification are made"""
    if inject is None :
        inject = (np.zeros((L, H, S, S)), np.zeros((L, H, S, S), dtype=bool))
    for tails_i, heads_i in dep_tup:
        for t_i in tails_i:
            t_i-=1 #t_i and h_i are compute with a virtual <s>
            for h_i in heads_i:
                h_i -=1
                for l, h in heads:
                    if direction[0]:  # tail attends head:
                        inject[0][l, h, t_i, h_i] = value_generator(l, h)
                        inject[1][l, h, t_i, h_i] = True
                    if direction[1]:  # head attends tail :
                        inject[0][l, h, h_i, t_i] = value_generator(l, h)
                        inject[1][l, h, h_i, t_i] = True
    return inject

def make_task(inject_function):
    """arg: inject_function :
            (L, H, S, deps_line, heads_line, head_table, cats_lines) -> a inject tuple or None
        returns : the corresponding task"""
    def task(model, data, deps_line, heads_line, head_table, cats_lines):
        L, H = model.num_layers, model.num_heads
        _, S = data["tokens"].shape
        inject = inject_function(L,H,S,deps_line, heads_line, head_table, cats_lines)
        if inject is None:
            return None
        if gen_test:
            where = np.argwhere(inject[1])
            print("Where : ", list(where) if len(list(where))<15 else "Too long")
            print("Values :", [inject[0][tuple(d)] for d in where] if len(list(where))<15 else "Too long")
        _, output = model(data, training=False, inject=inject)
        return output
    return task

# initialization so everything can be used as a module
stats = None

ampli = 1 #amplification factor, control the amplitude of the injected attention

def normal_generator(l,h):
    return gauss(stats["mean"][l,h], ampli* stats["std"][l,h])

def low_generator(l,h):
    return stats[0.5][l,h]+ ampli * (stats[0.1][l,h] - stats[0.5][l,h])

def high_generator(l,h):
    return stats[0.5][l,h]+ ampli * (stats[0.9][l,h] - stats[0.5][l,h])


def make_cross_task(dep, direction, heads, heads_reverse = None):
    """For sentences that have two of the given dep, tries to invert them.
    Direction, an array of two booleans, indicate whether the direct and/or reverse dep should be affected
    If only heads is given, it will make all changes on heads.
    If heads_reverse is given, heads is regarded as the heads attending to the direct dependency and heads_reverse
        to the reverse dependecy.
    e.g. : on "The blue sheep likes the green dog", make_cross_task(amod, [False, True], [(0,0)]) will suppress
        attention of sheep on blue and dog on green and activate it for sheep on green and dog on blue. """
    def cross_inj(L, H, S, deps_line, heads_line, head_table, cats_lines):
        dep_tup = get_dep(dep, deps_line, heads_line, head_table, n=2)
        if dep_tup is None:
            return None
        if heads_reverse is None:
            # inhibiting attention on original pairs
            inject = make_inject_target(L, H, S, dep_tup, direction, heads, low_generator)
            # switching
            pair1, pair2 = dep_tup
            dep_tup = [(pair1[0], pair2[1]), (pair2[0], pair1[1])]
            # stimulating switched pairs
            inject = make_inject_target(L, H, S, dep_tup, direction, heads, high_generator, inject=inject)
        else:
            assert all(direction)
            inject = make_inject_target(L, H, S, dep_tup, [True, False], heads, low_generator)
            inject = make_inject_target(L, H, S, dep_tup, [False, True], heads_reverse, low_generator, inject)
            pair1, pair2 = dep_tup
            dep_tup = [(pair1[0], pair2[1]), (pair2[0], pair1[1])]
            inject = make_inject_target(L, H, S, dep_tup, [True, False], heads, high_generator, inject)
            inject = make_inject_target(L, H, S, dep_tup, [False, True], heads_reverse, high_generator, inject)

        return inject
    return make_task(cross_inj)

def make_target_task(dep, direction, heads, value_generator):
    """Args:
       dep : a dependency
       direction : a tuple of two bools. Indicates whether tail and/or head attention over the other should be changed
       heads : the tuples (l,h) of the heads to target
       value_generator : a function (l,h) -> the value to inject.
       Returns a task that inject a value generated by value_generator for the first (tail and/or head) of the given dep
        in the given heads"""
    def target_inj(L, H, S, deps_line, heads_line, head_table, cats_lines):
        dep_tup = get_dep(dep, deps_line, heads_line, head_table, n=1)
        if dep_tup is None:
            return None
        inject = make_inject_target(L, H, S, dep_tup, direction, heads, value_generator)
        return inject
    return make_task(target_inj)

def make_switch_task(dep1, heads1, dep2, heads2, direction):
    """Makes a task that tries to switche dependencies dep1 and dep2, which share a head.
        arg : direction : 2-list in {0,1} 
        e.g. : for "The quick dog eats a bone", dep1 = nsubj, dep2 = obj, direction = [0,1], 
            the task will lower the attention of "eats" on "dog" for heads1 and on "bone" for heads2,
            and increase the attention of "eats" on "dog" for heads2 and on "bone" for heads1.
    """
    def switch_inj(L, H, S, deps_line, heads_line, head_table, cats_lines):
        dep_tup1 = get_dep(dep1,deps_line, heads_line, head_table, n=1 )
        dep_tup2 = get_dep(dep2,deps_line, heads_line, head_table, n=1 )
        if dep_tup1 is None or dep_tup2 is None:
            return None
        if dep_tup1[0][1] != dep_tup2[0][1]:
            return None
        # turns off the existing deps
        inject = make_inject_target(L,H,S, dep_tup1, direction, heads1, low_generator )
        inject = make_inject_target(L,H,S, dep_tup2, direction, heads2, low_generator, inject=inject )
        # turns on the new deps
        inject = make_inject_target(L, H, S, dep_tup1, direction, heads2, high_generator, inject=inject)
        inject = make_inject_target(L, H, S, dep_tup2, direction, heads1, high_generator, inject=inject)
        return inject
    task = make_task(switch_inj)
    return task

def make_label_reverse_task(dep, direction,heads,cat,direction_labels, labels_on=None, labels_off=None):
    """Makes a task that switches on/off the given heads on the given labels for the given deps"""
    labels_on = labels_on or []
    labels_off = labels_off or []

    def reverse_inj(L, H, S, deps_line, heads_line, head_table, cats_lines):
        dep_tup = get_dep(dep, deps_line, heads_line,head_table,n=1 )
        if dep_tup is None:
            return None
        label = cats_lines[cat][dep_tup[0][direction_labels][0]] #dep_tup := [((t_0,...,t_i),(h_0,...,h_j))]
        if gen_test:
            print(label)
        if label in labels_on:
            inj = make_inject_target(L,H,S,dep_tup, direction, heads, high_generator)
        elif label in labels_off:
            inj = make_inject_target(L,H,S,dep_tup, direction, heads, low_generator)
        else:
            inj = None
        return inj
    return make_task(reverse_inj)


def make_random_noise(head_list = None):
    """Task just injecting random noise on all tokens pairs of selected heads.
    If none, all heads are selected"""
    def random_noise_inj(L,H,S,  deps_line, heads_line, head_table, cats_lines):
        inject_0 = np.zeros((L,H,S,S))
        inject_1 = np.zeros((L,H,S,S), dtype = bool)
        if head_list is None or head_list=='all':
            heads = [(i,j) for i in range(L) for j in range(H)]
        else:
            heads = head_list
        for l,h in heads:
            # faster to create a random array with numpy than to iterate over all values with make_injct_target
            inject_0[l,h] = rng.normal(stats["mean"][l,h], stats["std"][l,h]  * ampli, (S,S))
            inject_1[l,h] = np.ones((S,S),dtype=bool)
        return (inject_0, inject_1)
    return make_task(random_noise_inj)


task_dict = {
    "sto_nsubj_vb" : make_target_task("nsubj",[False, True],[(3,0),(2,1)], normal_generator),
    "sto_nsubj_subj" : make_target_task("nsubj", [True, False],[(3,3)], normal_generator),
    "low_nsubj_vb" : make_target_task("nsubj",[False, True],[(3,0),(2,1)], low_generator),
    "low_nsubj_subj" : make_target_task("nsubj", [True, False], [(3,3)], low_generator),
    "cross_nsubj_vb" : make_cross_task("nsubj", [False, True], [(3,0),(2,1)]),
    "switch_subj_obj" : make_switch_task("nsubj",[(3,0),(2,1)], "obj", [(3,7),(4,4)],[0,1] ),
    "reverse_passive_nsubj" : make_label_reverse_task("nsubj", [False,True], [(4,4)], "voice", 1, labels_on=["Act"]
                                                      ,labels_off=["Pass"] ),
    "low_nsubj_both" :  make_target_task("nsubj",[True, True],[(3,0),(2,1),(3,3)], low_generator),
    "cross_amod_adj"  : make_cross_task("amod", [True, True], [(4,4),], heads_reverse=[(3,4),(4,6)]),
    "random_noise" : make_random_noise('all'),
    "sto_det_both" : make_target_task("det", [True,True],[(0,3),(1,3),(3,4),(4,4),(2,4)], normal_generator),

}

delta_t= 1_000
if __name__ == '__main__':
    parser = ArgumentParser("Inject attention into the model, for different tasks\n"
                            "Works only on fr->de for now.\n"
                            f"Available tasks : {list(task_dict.keys())}\n"
                            f"Creates a file per given task")
    parser.add_argument("corp", help="Tokens")
    parser.add_argument("dest", help="Folder to save results to")
    parser.add_argument("--dep", help="Aligned dependencies file", required=True)
    parser.add_argument("--head", help="Aligned head file", required=True)
    parser.add_argument("--stats", help="Folder containing the attention stats on the model", required=True)
    parser.add_argument("-M", help="Model", required=True)
    parser.add_argument("-C", help="Model configuration", required=True)
    parser.add_argument("--test", help="Testing", action="store_true")
    parser.add_argument("-T", help=f'Tasks among {list(task_dict.keys())}', nargs='+', default=[])
    parser.add_argument("--ampli", help = "Amplification factor controlling the amplitude of injected attention",
                        type = int, default = 1)
    for k in aad.categories.keys():
        parser.add_argument(f"--{k}", help=f"Labels file for {k} category.")
    args = parser.parse_args()
    pprint(vars(args))
    ampli = args.ampli
    # getting stats
    stats = get_stats(args.stats)
    # setting testing
    gen_test = args.test
    tasks = {k : task_dict[k] for k in args.T}
    try :
        os.mkdir(args.dest)
    except OSError:
        pass

    main(tasks)

