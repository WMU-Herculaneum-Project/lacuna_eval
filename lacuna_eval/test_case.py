# A test case for the Lacuna evaluation
# In JSON format it looks like this:
# {
#     "case_index": 2,
#     "id": "DDbDP/cde.85.247/1/2",
#     "number_alternatives": 1,
#     "max_length": 1,
#     "min_length": 1,
#     "mode_length": 1,
#     "test_case": "ἔτους ἐβδ.μου Ἀ[.]τωνείνου\nΚαίσαξος τοῦ κυρίου Μεχεὶρ κγ\nδιέγραψε Ἀρείῳ ἐγλήμπτορι χειροναξίου \nἙριεῦς Ἁξπαγάθου τοῦ Ἑριέως\nμητρὸς Τανούπιος γέρδιος Σοκνοπαίου \nΝήσου ὑπὲρ χειροναξίου τοῦ\nπέμπτου ἔτους ἐπὶ λόγου ἀργυρίου δραχμὰς \nὀκτώ, γίνονται δραχμαὶ η",
#     "alternatives": [
#       "ν"
#     ]
# }
#
# It is the responsibility of the model to receive the test_case string
# It should return a list of alternatives, which are an ordered list of strings
# The alternatives are ordered from the most likely to the least likely
# OR
# An ordered list of tuples, where the first element is the alternative
# and the second element is the probability, where the probabilities are floats
# between 0 and 1
# Again, these should be ordered from the most likely to the least likely


class LacunaTastCase:
    @classmethod
    def from_json(cls, json_obj):
        return cls(
            json_obj["case_index"],
            json_obj["id"],
            json_obj["test_case"],
            json_obj["alternatives"],
        )

    def __init__(self, case_index, id, test_case, alternatives):
        self.case_index = case_index
        self.id = id
        self.test_case = test_case
        self.alternatives = alternatives

    def __str__(self):
        return f"Case {self.case_index}:\n{self.test_case}\nAlternatives: {self.alternatives}"

    def __repr__(self):
        return self.__str__()

    def alternatives(self):
        return self.alternatives

    def test_case(self):
        return self.test_case

    def id(self):
        return self.id

    def case_index(self):
        return self.case_index

    def number_alternatives(self):
        return len(self.alternatives)
