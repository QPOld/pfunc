r"""
This creates a function that evaluates to a sequence over the integer domain
from zero to the length of the sequence.

Every sequence can be written as sums of binary sequences. If there was a way
to represent binary sequences of some length then it would be possible to
represent any sequence of some length by summing together each binary
sequence needed to create the sequence. This program will deduce all required
binary sequences for a given input sequence then uses each binary sequence to
create a function to represent each of those sequences. Each sequence is then
summed them together for the final function. When the function is evaluated
from zero to the length of the sequence it will produce the wanted
sequence. The sequence will repeat if you keep iterating the function but only
after 2^(2^(ceil(logb(logb(length, 2) + 1, 2) + 1) - 1) - 1) - 1 - l zeros. The
algorithm scales like Log(l) * l^2, where l is the length of the sequence.

EXAMPLES::
    // sage --preparse pfunc.sage
    // mv pfunc.sage.py pfunc.py

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



AUTHORS:

- Michael Quinn Parkinson (2019-01-01): initial version

"""


# ****************************************************************************
#       Copyright (C) 2019 Michael Quinn Parkinson <mqparkinson@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************


from sage.functions.log import logb
from sage.all import var, prod, cos, pi, ceil, n
import sage.all
import numpy as np


###############################################################################


var('n')


###############################################################################


def p(a, n):
    r"""
    When the p function is evaluated over the period it produce a single 1
    followed by 2^(2^(a - 1) - 1) - 1 zeros then repeats.

    INPUT:

    - ``a`` -- integer (default: `0`); The scale factor of the p function.

    - ``n`` -- variable (default: `n`); Sage variable for iterating over
    the function.

    OUTPUT: A sage.expression of a product of cosines squared.

    TESTS::
        >>> p(1, 0)
        1

    .. SEEALSO::

        :func:`_calculate`
    """

    # Return sage product of cosines squared
    return prod(
        pow(cos(n*pi / pow(2, q)), 2)
        for q in range(1, pow(2, a - 1))
    )


###############################################################################


def _deduce(start, end, length):
    r"""
    Deduce all required p functions by +/- one from each element of the
    original sequence if nonzero then storing either +/- 1 if nonzero or 0 if
    zero into a new sequence. This process is repeated until the orginal
    sequence is the new sequence. At this point all required p functions are
    deduced.

    INPUT:

    - ``start`` -- np.ndarray (default: `[]`); An array of the initial starting
    sequence.

    - ``end`` -- np.ndarray (default: `[]`); An array of the final ending
    sequence.

    - ``length`` -- integer (default:`0`); The length of original_sequence.

    OUTPUT: An array of arrays of every required step.

    TESTS::

        >>> ops = _deduce([0,0,0,0], [4,2,0,4], 4)
        >>> print(ops)
        [[1 1 0 1]
         [1 1 0 1]
         [1 0 0 1]
         [1 0 0 1]]


    .. SEEALSO::

        :func:`pfunc`
    """

    # Initial operation matrix
    operations = []

    # While start and end are not equal.
    while not np.array_equal(start, end):

        # Initial step array
        step = []

        # Adjust lengths of list to match by inserting zeros.
        start_len = len(start)
        end_len = len(end)
        while(start_len != length):
            start = np.insert(start, [0], [0])
            start_len = len(start)
        while(end_len != length):
            end = np.insert(end, [0], [0])
            end_len = len(end)

        # Loop length of sequence
        for i in range(length):

            # If not equal deduce operation
            if start[i] != end[i]:

                if start[i] > end[i]:
                    start[i] = start[i] - 1
                    step.append(-1)
                elif start[i] < end[i]:
                    start[i] = start[i] + 1
                    step.append(1)
                else:
                    step.append(0)
            else:
                step.append(0)

        # Append operation step to operation matrix
        operations.append(step)

    # Return operation matrix as numpy array.
    return np.asarray(operations)


###############################################################################


def _calculate(ops, length):
    r"""
    Calculte the sum of p functions needed to generate each operation step.
    Each sequence that represents each step is calculated with p functions
    then summed together to create the final sequence.

    INPUT:

    - ``ops`` -- np.ndarray (default: `[]`); An array of arrays that contains
    every required step to get from one sequence to another.

    - ``length`` -- integer (default:`0`); The length of original_sequence.

    OUTPUT: The sum of every required p function to represent the sequence.

    TESTS::

        >>> l = 4
        >>> ops = _deduce([0,0,0,0], [4,2,0,4], l)
        >>> func = _calculate(ops, l)
        >>> print(type(func))
        <type 'sage.symbolic.expression.Expression'>


    .. SEEALSO::

        :func:`pfunc`
    """

    total = 0  # Function defaults to zero
    steps = len(ops)
    alpha = ceil(logb(logb(length, 2) + 1, 2) + 1)

    # Loop all steps in operations matrix
    for i in range(steps):

        # Start with 1
        _row = p(alpha, n)

        # Loop all elements per operation
        for j in range(length):

            # Place a 1 in the jth position
            if ops[i][j] == 1 and j != 0:
                _row = _row + p(alpha, n - j)

            # Place a -1 in the jth position
            if ops[i][j] == -1 and j != 0:
                _row = _row - p(alpha, n - j)

            # Start with -1
            if ops[i][j] == -1 and j == 0:
                _row = -p(alpha, n)

            # Start with 0
            if ops[i][j] == 0 and j == 0:
                _row = p(alpha + 1, n + 1)

        # Sum it all together
        total = total + _row

    # Return sum of all functions.
    return total


###############################################################################


def _evaluate(sum_of_funcs, length):
    r"""
    Return an array of the value of the total sum of p functions when
    evaluated with integers between 0 and the length of the original sequence.

    INPUT:

    - ``sum_of_funcs`` -- sage.expression (default: `0`); The sum of all the
    functions that potentially represents the original sequence.

    - ``length`` -- integer (default:`0`); The length of original_sequence.

    OUTPUT: A numpy array of values when sum_of_funcs is evaluated.

    TESTS::

        >>> l = 4
        >>> ops = _deduce([0,0,0,0], [4,2,0,4], l)
        >>> func = _calculate(ops, l)
        >>> comp_seq = _evaluate(func, l)
        >>> print(comp_seq)
        [4 2 0 4]

    .. SEEALSO::

        :func:`check`
    """

    # Temp array to create numpy array
    _arr = []

    # When only a single digit is allowed the sum_of_funcs is type integer.
    if isinstance(sum_of_funcs, sage.rings.integer.Integer):
        _arr.append(sum_of_funcs)
    else:

        # Evaluate function when the iterator equals i from 0 to the length.
        for i in range(length):
            _arr.append(int(sum_of_funcs(n=i)))

    # Returns numpy array
    return np.asarray(_arr)


###############################################################################


def _check(function, original_sequence, length):
    r"""
    This evaluates the function over the length and stores the values into a
    numpy array. It then uses the built-in numpy expressions to check if it is
    equal to the original sequence.

    INPUT:

    - ``function`` -- sage.expression (default: `0`); The function that
      represents the original sequence.

    - ``original_sequence`` -- list (default: `[]`); The original sequence
      as a numpy array.

    - ``length`` -- integer (default:`0`); The length of original_sequence.

    OUTPUT: True if the function represents the sequence or False otherwise.

    TESTS::

        >>> l = 4
        >>> ops = _deduce([0,0,0,0], [4,2,0,4], l)
        >>> func = _calculate(ops, l)
        >>> equivalence = _check(func, [4,2,0,4], l)
        >>> print(equivalence)
        True

    .. SEEALSO::

        :func:`pfunc`
    """

    # Evaluate the P sums
    comp_seq = _evaluate(function, length)

    # Check if both sequences are equal.
    equivalence = np.array_equiv(original_sequence, comp_seq)

    # Return boolean of equivalence
    return equivalence

###############################################################################


def _pfunc(initial_integers, initial_function, final_integers, length):
    r"""
    Main wrapper for deduce and caluclate functions.

    INPUT:
    - ``initial_integers`` -- list (default:`[]`); The initial starting
      sequence of integers.

    - ``initial_function`` -- sage.expression (default:`0`); The original
      expression that represents the sequence of integers.

    - ``final_integers`` -- list (default:`[]`); The final ending sequence
      of integers.

    - ``length`` -- integer (default:`0`); The length of initial_integers.

    OUTPUT: A sage.expression that represents the final_integers.

    TESTS::

        >>> func = _pfunc([0,0,0,0], 0, [4,2,0,4], 4)
        >>> print(type(func))
        <type 'sage.symbolic.expression.Expression'>
    """

    # Deduce operations
    ops = _deduce(initial_integers, final_integers, length)

    # Calculate the p sums
    func = _calculate(ops, length)

    # Return Combined function
    return initial_function + func


###############################################################################


if __name__ == "__main__":
    import doctest
    doctest.testmod()
