"""Remove unicode unusual spaces"""

import argparse
from unicodedata import category
from contextlib import ExitStack

if __name__ == '__main__':
	parser = argparse.ArgumentParser("Removes unusual spaces from file")
	parser.add_argument("src", nargs="+")

	args = parser.parse_args()
	with ExitStack() as exit_stack:
		readers = [exit_stack.enter_context(open(f)) for f in args.src]
		writers = [exit_stack.enter_context(open(f+".spaces", "w")) for f in args.src]
		for reader, writer in zip(readers, writers):
			for line in reader:
				s = [c if category(c)!= 'Zs' else ' ' for c in line]
				writer.write(''.join(s))
