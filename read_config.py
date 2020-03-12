config = {}

with open("config", 'r') as f:
    for line in f.readlines():
        key, value = line.split(":", 1)
        config[key.strip()] = value.strip(" \n")