hindi_suffixes = [
    'ा', 'ी', 'ें', 'ों', 'े', 'ि', 'ू', 'ु',      # Gender/Plural variations
    'ता', 'ते', 'ती', 'ना', 'वाला',               # Verb tense/aspect
    'वाले', 'वाली', 'कर', 'हुए', 'करना',         # Participle forms
    'में', 'से', 'के', 'पर', 'तक', 'की', 'को'    # Postpositions
    ]
    
words=['खेलते','खेलती','खेला']

# Check and remove any suffix found at the end of the word
for word in words:
    for suffix in hindi_suffixes:
        if word.endswith(suffix):
            print(word[:-len(suffix)])
