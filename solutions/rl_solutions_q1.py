import unittest
def split_string(s, D):
    '''
    :param s: input string
    :param D: list of words
    :return: string split based on elements or words in D
    '''
    if s in D:
        return s #if the entire word is already in D, return input string
    # iterate through the input string to check if the prefix (or the first set of characters) is in D
    for i in range(len(s)+1):
        prefix = s[0: i]
        suffix = s[i: len(s)]
        # creating a list of substrings
        l = [s[i:i + j] for j in range(0, len(s)+1)]
        # common elements between list of substrings and D
        matched_substrings = set(l) & set(D)
        # update prefix and suffix based on the matched substrings
        if len(matched_substrings) > 0:
            # the longest matched string is identified as the prefix
            prefix = max(list(matched_substrings))
            suffix = s[len(prefix):len(s)]
        if prefix in D:
            # recursive call to split suffix
            segmented_suffix = split_string(suffix, D)
            if segmented_suffix is not None:
                main_string = prefix + " " + segmented_suffix
                return main_string


def get_min_spaces_from_string(s, D):
    '''

    :param s: input string
    :param D: list of words
    :return: least number of spaces (int)
    '''
    str1 = split_string(s, D)
    if str1 is not None:
        return len(str1.split()) - 1
    else:
        return "n/a"

class TestQuestionOne(unittest.TestCase):

    def test_split_count(self):
        self.assertEqual(get_min_spaces_from_string('shahnazmshariff', ['sh', 'naz', 'shah', 'na', 'dev']), "n/a")
        self.assertEqual(get_min_spaces_from_string('shahnaz', ['sh', 'naz', 'shah', 'na', 'dev']), 1)
        self.assertEqual(get_min_spaces_from_string('abcdefabcdef', ['def', 'abc', 'ab', 'cd', 'ef']), 3)
        self.assertEqual(get_min_spaces_from_string('abcdefab', ['def', 'abc', 'ab', 'cd', 'ef']), 2)
        self.assertEqual(get_min_spaces_from_string('cdab', ['ab', 'cd']), 1)
        self.assertEqual(get_min_spaces_from_string("abc", ['ab', 'cd']), "n/a")
        self.assertEqual(get_min_spaces_from_string("rlsolutions", ['r', 'l','solutions']), 2)
        self.assertEqual(get_min_spaces_from_string("rlsolutions", ['rl', 'solutions', 'solution']), 1)
        self.assertEqual(get_min_spaces_from_string('Otorhinolaryngologist', ["lo","i","s","t","ino","o","n","o","l","a","r","gist","tor","logist","aryn","nolary","t","o","r","h","i","o","l","o","g","Otorh",
"no","th","o","n","y","n","g","laryngo"]), 3)
        self.assertEqual(get_min_spaces_from_string("",['a']), "n/a")
        self.assertEqual(get_min_spaces_from_string("verification", []), "n/a")
        self.assertEqual(get_min_spaces_from_string("abcdefgh", ['a', 'b', 'c', 'defgh']), 3)


if __name__ == '__main__':
    # to add a custom test case, update value of s & D
    s = 'abcdefabcd'
    D = ['def', 'abc', 'ab', 'cd', 'ef']
    result = get_min_spaces_from_string(s, D)
    print result
    # run unittests 
    unittest.main()
