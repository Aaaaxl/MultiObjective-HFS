def print_separator(text, sep="=", total_len=50):
    text = f" {text} "
    side_len = (total_len - len(text)) // 2
    print(sep * side_len + text + sep * (total_len - side_len - len(text)))