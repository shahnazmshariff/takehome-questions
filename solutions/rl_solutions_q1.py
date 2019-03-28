import unittest
import re

d = {}

def split_string(s, D, output_string = None):
    '''

    :param s: input string
    :param D: input list of words
    :param output_string: intermediate string containing split results
    :return:
    '''

    if output_string is None:
        output_string = ""

    # store intermediate results in a dictionary if end of string is reached
    if len(s) == 0:
        count_spaces = output_string.count(' ')
        if count_spaces > 0:
            d[count_spaces-1] = output_string[1:]
    # recursive call to split suffix
    for i in range(len(s)+1):
        prefix = s[0: i]
        suffix = s[i: len(s)]
        if prefix in D:
            split_string(suffix, D, output_string + " " + prefix)

def return_min_splits(s,D):
    split_string(s, D)
    # if split is possible
    if len(d) > 0:
        # return min split and the corresponding sequence/ split results
        split_results = d[min(d.keys())]
        min_split = min(d.keys())
        d.clear()
        return split_results, min_split
    else:
        return "n/a"

class TestQuestionOne(unittest.TestCase):

    def test_split_count(self):
        self.assertEqual(return_min_splits('abcdefab', ['def', 'abc', 'ab', 'cdefab', 'ef']), ('ab cdefab', 1))
        self.assertEqual(return_min_splits('abcdefghijkl', ['abcd', 'efgh', 'ijkl', 'ab', 'cdefg', 'hij', 'kl']), ('abcd efgh ijkl', 2))
        self.assertEqual(return_min_splits('shahnazmshariff', ['sh', 'naz', 'shah', 'na', 'dev']), "n/a")
        self.assertEqual(return_min_splits('shahnaz', ['sh', 'naz', 'shah', 'na', 'dev']), ('shah naz', 1))
        self.assertEqual(return_min_splits('abcdefabcdef', ['def', 'abc', 'ab', 'cd', 'ef']),
                         ('abc def abc def', 3))
        self.assertEqual(return_min_splits('abcdefab', ['def', 'abc', 'ab', 'cd', 'ef']), ('abc def ab', 2))
        self.assertEqual(return_min_splits('cdab', ['ab', 'cd']), ('cd ab', 1))
        self.assertEqual(return_min_splits("abc", ['ab', 'cd']), "n/a")
        self.assertEqual(return_min_splits("rlsolutions", ['r', 'l', 'solutions']), ('r l solutions', 2))
        self.assertEqual(return_min_splits("rlsolutions", ['rl', 'solutions', 'solution']),
                         ('rl solutions', 1))
        self.assertEqual(return_min_splits('Otorhinolaryngologist',
                                                    ["lo", "i", "s", "t", "ino", "o", "n", "o", "l", "a", "r", "gist",
                                                     "tor", "logist", "aryn", "nolary", "t", "o", "r", "h", "i", "o",
                                                     "l", "o", "g", "Otorh",
                                                     "no", "th", "o", "n", "y", "n", "g", "laryngo"]),
                         ('Otorh ino laryngo logist', 3))
        self.assertEqual(return_min_splits("", ['a']), "n/a")
        self.assertEqual(return_min_splits("verification", []), "n/a")
        self.assertEqual(return_min_splits("abcdefgh", ['a', 'b', 'c', 'ab', 'defgh']), ('ab c defgh', 2))

if __name__ == '__main__':
    # to add a custom test case, update value of s & D
    s = 'abcdefghijkl'
    D = ['abcd', 'efgh', 'ijkl', 'ab', 'cdefg', 'hij', 'kl']
    result = return_min_splits(s, D)
    print result
    # run unittests
    unittest.main()