def load_hf_token(filepath: str = "C:\\Users\\chris\\PycharmProjects\\hf_token.txt") -> str:  # hehe
    with open(filepath, 'r') as f:
        return f.readline()

