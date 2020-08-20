"""A script to rank tokens in a corpus by rarity"""

import argparse
from contextlib import ExitStack

if __name__ == '__main__':
    parser = argparse.ArgumentParser("A script to rank tokens in a corpus by rarity")
    parser.add_argument("-f", help="Files to search", nargs="+", required = True)
    parser.add_argument("-t", help="File to write to", required=True)

    args = parser.parse_args()

    with ExitStack() as exit_stack:
        readers = [exit_stack.enter_context(open(file)) for file in args.f]
        writer = exit_stack.enter_context(open(args.t, "w"))

        freq = {}
        count_empty_tokens = 0
        for reader in readers :
            print(f"Reading frequencies in {reader.name}")
            for i,line in enumerate(reader):
                words = line.strip("\n").split(" ")

                for w in words:
                    if not w:
                        count_empty_tokens+=1
                    if w in freq:
                        freq[w] +=1
                    else:
                        freq[w] = 1
        freq_list =  sorted(freq.items(), key = lambda x : x[1])

        for t, j in freq_list:
            writer.write(t+" "+str(j)+"\n")
    print(f"Found {count_empty_tokens} empty tokens on {i+1} lines")
