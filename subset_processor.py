with open("marco-data/msmarco_passages_subset.tsv") as f:
    keep_ids = {line.strip() for line in f}

with open("marco-data/collection.tsv") as fin, open("marco-data/collection_subset.tsv", "w") as fout:
    for line in fin: 
        pid = line.split("\t", 1)[0]
        if pid in keep_ids:
            fout.write(line)