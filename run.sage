import numpy as np
from sage.misc.prandom import randrange
from pfunc import _pfunc, _check


# Pick a length for the sequence.
length = randrange(1, 30)
zeros = np.zeros(length, dtype='int')

# Randomly generate a sequence of integers with a given length.
initial_integers = np.array([randrange(length + 1) for i in range(length)])

# Sequence copy
orig_seq = np.array(initial_integers.tolist())

print("Length {} sequence \n".format(length))
print("\nGo from\n {} \nto\n {}".format(zeros, initial_integers))
func = _pfunc(zeros, 0, initial_integers, length)
print("\np = {}".format(func))

print("\nChecking...\n")
equivalence = _check(func, orig_seq, length)

print("Solution is {}".format(equivalence))
