import os, sys
import pandas as pd
from os.path import join

import pandas as pd

def to_boolean(label):
    label = int(label)
    if label == 1:
        return "Yes"
    elif label == 0:
        return "No"
    else:
        raise Exception(f"Unexpected label in dataset! {label}")

def main(args):
    if len(args) < 2:
        sys.stderr.write("2 required arguments: <input directory> <output directory>\n")
        sys.exit(-1)
    
    # pre-process the data which has columns ID, label, judgement, and text as 4 tab-separated columns. change to just Yes/No and text column for tsv format.
    files = ["in_domain_dev.tsv",
            "in_domain_train.tsv",
            "out_of_domain_dev.tsv"]
    
    for fn in files:
        df = pd.read_csv(join(args[0], fn), sep="\t", header=None, names=['id', 'label', 'star', 'text'])
        df["acceptable"] = df.label.apply(to_boolean)
        df = df[["acceptable", "text"]]
        df.to_csv(join(args[1], fn), 
                  sep="\t",
                 index=False,
                 header=True,
                 escapechar=None)
        
if __name__ == '__main__':
    main(sys.argv[1:])
