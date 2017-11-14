import six
assert six.PY3, "Run me with Python3"

import numpy as np
import scipy.sparse
import csr_csc_dot as ccd
import time
import sys
import re
from sklearn.feature_extraction.text import HashingVectorizer

discard_re=re.compile(r"[^a-zA-ZåäöÅÄÖ0-9 ]") #regex to discard characters which are not of our interest
wspace_re=re.compile(r"\s+") #regex to discard characters which are not of our interest

def documents2sparse(documents,ngram=4,n_features=10**7):
    """
    documents: iterable yielding docs
    ngram: how long ngrams to consider for deduplication?
    n_features: used as modulo after hash, so all ngrams willbe mapped to this many features. Cannot be made too large because we will also need a feature x document sparse matrix. 10**6 or 10**7 are good values.

    Returns a sparse matrix. This can be saved with:
    scipy.sparse.save_npz("matrix_file.npz",sparse_m)
    """
    vectorizer=HashingVectorizer(lowercase=True, ngram_range=(ngram,ngram), n_features=n_features,norm=None,dtype=np.float32) #use float32 because this is what I need later
    docs=[]
    for d in documents:
        d=discard_re.sub(" ",d)
        d=wspace_re.sub(" ",d)
        docs.append(d)
    sparse_m=np.abs(vectorizer.transform(docs)) #CSR matrix (one row for document, sparse columns for features)
    return sparse_m


def duplicates_matrix_pair(m1,m2,slice_width=10000,m2_csc=None,cut=0.98):
    """
    Find duplicates between m1 and m2.
    m1: document-by-feature CSR sparse matrix (as produced eg by scikit vectorizers)
    m2: document-by-feature CSR sparse matrix 
    slice: sort of minibatch size - how many m1 rows are compared to m2 at once
    m2_csc: if None, will be calcuated as m2_csc=m2.tocsc() - provide if you happen to have it cached
    """

    # Inspired here
    # https://github.com/scipy/scipy/blob/v0.14.0/scipy/sparse/data.py#L86
    #
    # Diagonal values of m1.dot(m1) and m2.dot(m2) so we know what the maximum is
    diagonal1=m1._with_data(np.square(m1.data),copy=True).sum(-1)
    diagonal2=m2._with_data(np.square(m2.data),copy=True).sum(-1)

    if m2_csc is None:
        m2_csc=m2.tocsc()

    for slice_idx in np.arange(0,m1.shape[0],slice_width):
        out=np.zeros((slice_width,m2.shape[0]),np.float32)
        ccd.csr_csc_dot_f(slice_idx,slice_width,m1,m2_csc,out)
        #Now `out` is filled with the dot products
        out=np.square(out)
        out/=diagonal1
        out/=diagonal2
        #Now any value in `out` which is above something like 0.95 is a near-duplicate
        rows,cols=np.where(out>cut)
        for row,col in zip(rows,cols):
            yield row,col
        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    args = parser.parse_args()
    
    
