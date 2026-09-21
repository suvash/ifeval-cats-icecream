import ifeval_cats_icecream as env


def test_hidden_word_variants():
    assert env._hidden_score("I like icecream.", "icecream") == 1.0
    assert env._hidden_score("I like ice cream.", "icecream") == 1.0
    assert env._hidden_score("I like ice-cream.", "icecream") == 1.0
    assert env._hidden_score("I like Ice Cream.", "icecream") == 1.0
    assert env._hidden_score("This is nicecream.", "icecream") == 0.0
    assert env._hidden_score("This is icecreamy.", "icecream") == 0.0


if __name__ == "__main__":
    test_hidden_word_variants()
    print("hidden word asserts passed")
