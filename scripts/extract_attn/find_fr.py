"""Removes non-french text in a corpus (such as AFP)"""
import fasttext
import argparse
from tqdm import tqdm


langs = {"de": "__label__de", "fr":"__label__fr", "en":"__label__en"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script to remove text from languges other than a chosen language in a text")
    parser.add_argument("input", help="Text to process")
    parser.add_argument("--ft", help="Path to the fasttext language identification model", required=True)
    parser.add_argument('-t', help="File to write extracted text", default=None)
    parser.add_argument('-l', help=f"ISO code of language to keep (fr, de, en...)", default="fr")
    args = parser.parse_args()
    model = fasttext.load_model(args.ft)
    print(f"Extracting {args.l} sentences from "+args.input)
    others = {}
    i=0
    r=0
    target = "__label__"+args.l
    with open(args.input) as reader:
        with open(args.t, 'w') as writer:
            for line in tqdm(reader):
                i+=1
                ling = model.predict(line.strip("\n"), k=1)[0][0]
                if ling == target:
                    writer.write(line)
                else:
                    print(ling, line, end="")
                    r+=1
                    if ling in others:
                        others[ling]+=1
                    else :
                        others[ling] =1
    print(f"Done. Found : ")
    for k,v in others.items() :
        print(k,":",v)
    print(f"Removed {r} on {i} lines. ")
