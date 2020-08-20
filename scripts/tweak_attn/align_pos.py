"""'Align' a tokenized corpus with the corresponding POS,
by repeating the POS as many times as there are subparts of a word"""

import argparse
from datetime import datetime
from contextlib import ExitStack
from unicodedata import category


punc     = list("""()"{}/=+*%:;.,?!&[]#°""") #no apostrophe or minus
assert ("'" not in punc)
adpdet = ['au', 'aux', 'du', "des", "desdites", "desdits", "dudit"]
adppron = ["auquel", "duquel", "desquels", "desquelles", "auxquels", "auxquelles"]

def is_numeral(word):
    word = word.strip(spacer)
    for l in word:
        if not category(l) == 'Nd':
            return False
    return True

def replace_dashes(line : str):
    """It seems that pyonnmttok replaces any series of ----- by a — during tokenization"""
    l = []
    switch = False
    for c in line:
        if c != "-":
            if switch :
                l.append("—")
                switch=False
            l.append(c)
        elif not switch:
            switch=True
    if switch:
        l.append("—")
    return "".join(l)

class UnknownException(ValueError):
    """Exception used to control what errors are acceptable in align_ref"""
    pass

def contains_placeholder(tokens):
    """check for placeholders tokens"""
    ph = ["⦅ph_ent_uri#1⦆", "⦅ph_unk#1⦆"]
    return  any([p in tokens for p in ph])

def main(args):
    i = 0
    sup_readers = []
    sup_writers = []
    empty_lines=0
    errors=0
    skipped=0
    with ExitStack() as exit:
        #readers and writers
        tokens_r = exit.enter_context(open(args.tokens))
        pos_r = exit.enter_context(open(args.dep))
        writer = exit.enter_context(open(args.dep+args.suffix, "w"))
        text_r = exit.enter_context(open(args.text))
        tokens_writer = exit.enter_context(open(args.tokens+args.suffix, "w") )
        for filename in args.sup:
            sup_readers.append(exit.enter_context(open(filename)))
            sup_writers.append(exit.enter_context(open(filename + args.suffix, "w")))
        err_log = exit.enter_context(open("log.align_pos.err", "w"))
        print("Writing to :", writer.name, " ", [w.name for w in sup_writers], tokens_writer.name)
        #main loop
        for line_tok, line_pos, line_text in zip(tokens_r, pos_r, text_r):
            i+=1
            log = Log(i)
            log.write(line_text)
            sup_lines = [(r.readline()) for r in sup_readers]
            if not line_tok.strip("\n"): #if line is empty for any reason, pass
                empty_lines+=1
                log.print(err_log)
                continue

            if line_text.count('-')>30 or contains_placeholder(line_tok.split(" ")): #skip tables
                # and lines containing placeholders
                skipped+=1
                log.write(skipped)
                log.print(err_log)
                continue

            line_text = replace_dashes(line_text)
            sup_lines = [l.strip("\n").split(" ") for l in sup_lines]
            try : #try aligning
                pos, others = align_ref(line_tok.strip("\n").split(" "), line_pos.strip("\n").split(" "),
                                        line_text.strip("\n").split("_"), sup = sup_lines, log =log)
            except UnknownException as e:
                errors+=1
                if args.T2:
                    print("line ",i)
                    raise e
                else:
                    log.print(err_log)
            else: #if everything went well, writes the result
                writer.write(pos + "\n")
                tokens_writer.write(line_tok)
                for w, line in zip(sup_writers, others):
                    w.write(line + "\n")

            if not i % args.f :
                print(f'{datetime.now()} : line {i}, {errors} errors')
                if args.L:
                    log.print()
    print(f"Found {empty_lines} empty lines and {errors} errors, skipped {skipped} lines, on {i} lines")

def is_nonword_token(token):
    """Checks whether a token is a token created by the tokenizer (i.e. a case delimiter token) (or is empty)"""
    A =  "\uff5f" in token and not "\uff03" in token #"⦅"marks special tokens, but # marks numbers
    return A or not token

def is_numeral_token(token):
    return  "\uff5f" in token and  "\uff03" in token

def strip_num(word : str):
    """Returns the word, without numerical chars at the beginning"""
    i =0
    while i<len(word) and word[i].isnumeric():
        i+=1
    return word[i:]

def align_ref(tokens, pos, text : list, sup = None, log = None):
    """Align tokens and pos by matching tokens with text, then text with pos.
    Also align any line in sup"""
    if sup is None:
        sup = []
    matching = [-1 for _ in range(len(tokens))] #store at index i the index of the word matched with token i
    j = -1
    text_copy = [t for t in text]
    word = ''
    if sup:
        sup[0] = sup[0]+["-1"]
        sup[1:] =  [s+["<maj>"] for s in sup[1:]]
    pos.append("<maj>") #sup. token
    tildes = [] #marks places where to add & : mark a missing word BEFORE token (not after like ~)
    if log is None:
        log = Log()
    try :
        for i in range(len(tokens)):
            token = tokens[i].strip(spacer)
            if is_nonword_token(token):
                matching[i] = len(text_copy)
                if args.T2 or args.L:
                    log.write(token, i, [word], len(text_copy), )
                continue
            if not word : #end of word, next word.
                while not word: #ignores empty next words (that have no token)
                    word = text.pop(0)
                    word = word.strip(" ")
                    if not word:
                        tildes.append(i)
                    j+=1
                # These remain for compatibility with French UD corpus, which count words such as "du", "desquelles" as
                # two words ("de le", "de lesquelles")
                if word.lower() in adpdet and pos[j] == "ADP" and j+1<len(pos) and pos[j+1] == "DET":
                    pos [j] = "ADPDET"
                    pos.pop(j+1)
                    for l in sup:
                        l.pop(j) #will count the ADPDETs as their DET. Not perfect but ok
                    if sup:     # to mark places where a word lacks
                        sup[0][j]+="~"
                if word.lower() in adppron and pos[j] == "ADP" and j+1<len(pos) and pos[j+1] == "PRON":
                    pos[j] = "ADPPRON"
                    pos.pop(j + 1)
                    for l in sup:
                        l.pop(j)
                    if sup:
                        sup[0][j]+="~"
            if args.T2 or args.L:
                log.write(token, i, [word], j)
            if is_numeral_token(token): #if theres a numeral placeholder, it's much longer than the number
                word=strip_num(word)
            # elif all([not c.isalnum() for c in word]) and all([not c.isalnum() for c in token]): #both token and word are all symbols : next
            #     word = ""
            else:
                word = word[len(token) : ].strip(" ") #cut to token length, and removes in-word spaces (e.g. "12 000")
            matching[i] = j
        #sanity checks
        assert not any (t.strip(" ") for t in text) #it's okay if spaces remain
        assert not word
        assert all([i > -1 for i in matching])
        assert len(pos) == len(text_copy)+1
        assert all([len(l)== len(text_copy)+1 for l in sup])
        result = [pos[i] for i in matching]
        k = matching[0]
        result_sup = [ [l[i] for i in matching] for l in sup]
        if result_sup:
            for i,j in enumerate(matching): #writes word separator
                if j != k or j == len(text_copy): #tokens are from different word or nonword tokens
                    result_sup[0][i-1]+=";"
                k = j
        for i in tildes:
            result_sup[0][i]+="&"
    except (IndexError, AssertionError) as e:
        # Errors still happen on special cases (e.g. when spacy has separated a "---" in individual dashes).
        if not args.T:
            raise UnknownException()
        log.print()
        print(e)
        print(spacer)
        print(tokens, len(tokens))
        print(text_copy, len(text_copy))
        print(" ".join(pos), len(pos))
        for l in sup:
            print(" ".join(l), "len :",len(l))
        for match, token in zip(matching, tokens) :
            if match <len(pos):
                print(f"{token}:{pos[match]}", end=" ")
        print(matching, f"max : {max(matching)}")
        print("\n")
        raise UnknownException()

    return " ".join(result), [" ".join(l) for l in result_sup]

class Log():
    """To keep track and print what happened during main"""
    def __init__(self, i=None):
        self.text=[str(i)+"\n"] if i is not None else []
        self.i = i

    def write(self, *args):
        self.text.append(" ".join(str(a) for a in args)+"\n")

    def print(self, file=None):
        print("".join(self.text), file=file)

    def clear(self):
        self.text=[]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Align the words properties with a tokenized text. "
                                     "There should be one property by text part "
                                     "(word, number, punctuation, supbart of a contraction)")
    parser.add_argument("tokens", help="Token file")
    parser.add_argument('text', help="Em-dash-separated speech components ")
    parser.add_argument("dep", help="Dependencies or part-of-speech file")
    parser.add_argument('--heads', help='Heads file')
    parser.add_argument('--sup', help = "Other space_separated properties files to align", nargs="+")
    parser.add_argument("--suffix", help="suffix to append at the name of the written files.", default=".aligned")
    parser.add_argument("--spacer", help="Is square or underscore used as a spacer character for tokens ?",
                        required=True)
    parser.add_argument("-T", action="store_true", help="Testing")
    parser.add_argument("-T2", action="store_true", help="Stronger testing")
    parser.add_argument('-L', help='More complete logs', action="store_true")
    parser.add_argument('-f', help='Dumps logs this often', default=10000, type=int)
    args = parser.parse_args()
    if args.heads:
        args.sup = [args.heads]+args.sup

    args.T = args.T or args.T2

    if args.spacer.lower() in ["__","_","u","underscore"]:
        spacer = "_"
    elif args.spacer.lower() in ["￭", "s", "sq", "square", "\uffed"]:
        spacer = "\uffed"
    else :
        raise ValueError("Invalid spacer")

    main(args)
