FILENAME = 'data/web-Google-10k.txt'

with open(FILENAME) as file:
    lines = [line.strip().split('\t') for line in file]
    edges = [[int(src), int(dst)] for (src, dst) in lines[4:]]
    nodes = set([node for edge in edges for node in edge])

    links = { node: [] for node in nodes }
    for (src, dst) in edges:
        links[src].append(dst)

    dends = sum(1 for node in nodes if len(links[node]) == 0)
    print(dends)
