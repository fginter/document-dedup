import get_text
import argparse
import dedup
import scipy.sparse
import gzip
import json
import re

parser = argparse.ArgumentParser(description='Index')
parser.add_argument("--section", default=0, type=int, help='Which part to ask 0...N-1 (default %(default)d)')
parser.add_argument("--all", default=10, type=int, help='N (default %(default)d)')
parser.add_argument("--batch", default=100000, type=int, help='batchsize (default %(default)d)')
parser.add_argument("--preverts", default="/home/mjluot/preverts", help='preverts file (default %(default)s)')
parser.add_argument("outfile", help="outfile.batchnum.npz will be the matrices and outfile.batchnum.json.gz will be the data")

args = parser.parse_args()

markup_re=re.compile("""^<doc.*?langdiff=.*?>|<gap chars=".*?" />|<p heading="[0-9]+">|</p>|</doc>""")

for batch_num, (docs,indices) in enumerate(get_text.get_text(args.section,args.all,args.preverts,args.batch)):
    docs_to_index=[]
    for d in docs:
        d=markup_re.sub("",d)
        docs_to_index.append(d)
    sparse_m=dedup.documents2sparse(docs_to_index)
    scipy.sparse.save_npz("{}.batch_{}.npz".format(args.outfile,batch_num),sparse_m)
    with gzip.open("{}.batch_{}.json.gz".format(args.outfile,batch_num),"wt") as f:
        json.dump((docs,indices),f)
    print("Batch {} done".format(batch_num))
              
    

