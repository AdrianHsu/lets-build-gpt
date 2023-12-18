import tiktoken

enc = tiktoken.get_encoding('gpt2')
print(enc.n_vocab)
print(enc.encode("hii there"))
print(enc.decode([71, 4178, 612]))