import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

my_text = "Hello, world! This is Manoj Kumar."
tokens = enc.encode(my_text)

print(f"Token IDs: {tokens}")
# [13225, 11, 2375, 0, 1328, 382, 115942, 73, 70737, 13]
print(f"Number of tokens: {len(tokens)}")

dec = enc.decode(tokens)
print(f"Decoded text: {dec}")
print(f"Original and decoded texts are the same: {my_text == dec}")