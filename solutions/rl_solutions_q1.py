import unittest
import re
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
        return str1, len(str1.split()) - 1
    else:
        return "n/a"

def split_str_by_position(item, s, split_str, count_split):
    # find item in s
    pos = s.find(item)
    # obtain first and last part of the string (without the current item)
    start = s[:pos]
    end = s[pos + len(item):]
    # update s based on start and end
    s = str(start) + str(end)
    split_str.append(item)
    count_split += 1
    return s, split_str, count_split


def alternate_approach(s, D):
    '''

    :param s: input string
    :param D: input list of words
    :return: list containing the split results and the minimum number of splits
    '''
    # sorting the list in place in descending order based on the length of the string
    D.sort(key=len, reverse=True)
    count_split = 0
    split_str = []
    for item in D:
        # check if string in D (i.e., item) is present in s and then split s based on the start & end position of the item
        if item in s:
            # find all occurences of the item in s and split s that many times
            occurences = re.findall(item, s)
            for i in range(len(occurences)):
                s, split_str, count_split = split_str_by_position(item, s, split_str, count_split)
    # if the string cannot be split even partially or if string still contains some characters,
    # then the list of words (D) cannot be used to form the string (s)
    if count_split-1 <= 0 or len(s) > 0:
        return "n/a"
    split_str.append(count_split-1)
    # returns list containing the split results and the minimum number of splits
    return split_str

class TestQuestionOne(unittest.TestCase):

    def test_alternate_solution(self):
        self.assertItemsEqual(alternate_approach('abcdefab', ['def', 'abc', 'ab', 'cdefab', 'ef']), ['ab', 'cdefab', 1])
        self.assertItemsEqual(alternate_approach('abcdefabcdef', ['def', 'abc', 'ab', 'cd', 'ef']), ['abc', 'def','abc', 'def', 3])
        self.assertEqual(alternate_approach('shahnazmshariff', ['sh', 'naz', 'shah', 'na', 'dev']), "n/a")
        self.assertItemsEqual(alternate_approach('shahnaz', ['sh', 'naz', 'shah', 'na', 'dev']), ['shah', 'naz', 1])
        self.assertItemsEqual(alternate_approach('abcdefab', ['def', 'abc', 'ab', 'cd', 'ef']), ['abc', 'def', 'ab', 2])
        self.assertItemsEqual(alternate_approach('cdab', ['ab', 'cd']), ['cd', 'ab', 1])
        self.assertEqual(alternate_approach("abc", ['ab', 'cd']), "n/a")
        self.assertItemsEqual(alternate_approach("rlsolutions", ['r', 'l', 'solutions']), ['r', 'l', 'solutions', 2])
        self.assertItemsEqual(alternate_approach("rlsolutions", ['rl', 'solutions', 'solution']),
                              ['rl', 'solutions', 1])
        self.assertItemsEqual(alternate_approach('Otorhinolaryngologist',
                                                    ["lo", "i", "s", "t", "ino", "o", "n", "o", "l", "a", "r", "gist",
                                                     "tor", "logist", "aryn", "nolary", "t", "o", "r", "h", "i", "o",
                                                     "l", "o", "g", "Otorh",
                                                     "no", "th", "o", "n", "y", "n", "g", "laryngo"]),
                         ['Otorh', 'ino', 'laryngo', 'logist', 3])
        self.assertEqual(alternate_approach("", ['a']), "n/a")
        self.assertEqual(alternate_approach("verification", []), "n/a")
        self.assertItemsEqual(alternate_approach("abcdefgh", ['a', 'b', 'c', 'defgh']), ['a' ,'b', 'c', 'defgh', 3])


if __name__ == '__main__':
    # to add a custom test case, update value of s & D
    s = 'abcdefab'
    D = ['def', 'abc', 'ab', 'cdefab', 'ef']
    result = alternate_approach(s, D)
    print result
    # run unittests
    unittest.main()