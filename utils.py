# -*- coding: utf-8 -*-
from numba import njit, typeof, typed, types, jit
import numba 
import numpy as np
import pandas as pd

alph_dict = {'G':1, 'A':2, 'T':3, 'C':4}
alph_dict_n = typed.Dict.empty(types.int8,types.int8)
for key in alph_dict:
  alph_dict_n[ord(key)] = alph_dict[key]




@njit()
def get_neighbours(kmer, m):
    alph=[1,2,3,4]
    candidates = [ (0, 0) ]
    for letter in kmer:
        num_candidates = len(candidates)
        for i in range(num_candidates):
            candidate, mismatches = candidates.pop(0)
            if mismatches < m :
                for a in alph:
                    if a == letter :
                        candidates.append( (candidate*5 + a, mismatches) )
                    else:
                        candidates.append( (candidate*5 + a, mismatches + 1))

            if mismatches == m:
                candidates.append( (candidate*5 + letter, mismatches))

    return [b for b, m in candidates]
  

@njit
def encode_kmer(kmer):
    weights = np.array([5**i for i in range(len(kmer))])
    return np.sum(weights*np.array(kmer))

@njit
def encode(kmer, alph_dict_n):
    return [ alph_dict_n[ord(x)] for x in kmer]

@njit
def get_all_kmers(X, k, alph_dict_n ):
    d = len(X[0])
    last_idx = 0
    kmer_set = typed.Dict.empty(types.int64,types.int64)
    print("Populating kmer set")
    for x in X:
        for j in range(d - k + 1):
            kmer = encode(x[j: j + k], alph_dict_n)
            k_code =  encode_kmer(kmer)
            #print(k_code, kmer, x[j: j + k])
            if k_code not in kmer_set:
                kmer_set[k_code] = last_idx
                last_idx +=1
    return kmer_set, last_idx 

#@njit       
def mismatch_preprocess(X, k, m, kmer_set, last_idx, alph_dict_n):
    n = len(X)
    d = len(X[0])
    precomputed = typed.Dict.empty(types.int64,numba.int64[:])
    embedding = [typed.Dict.empty(types.int64,types.int64) for x in X]
    print("Generating mismatch embedding")
    for i,x in enumerate(X):
        for j in range(d - k + 1):
            kmer = encode(x[j: j + k], alph_dict_n)
            k_code =  encode_kmer(kmer)
            if k_code not in precomputed:
                Mneighborhood = get_neighbours(kmer, m)
                precomputed[k_code] = np.array([], dtype = np.int64) 
                for neighbor in Mneighborhood:
                  if neighbor not in kmer_set:
                    kmer_set[neighbor] = last_idx
                    last_idx +=1  
                  precomputed[k_code] = np.append(precomputed[k_code], kmer_set[neighbor])

            for idx in precomputed[k_code]:
                if idx in embedding[i]:
                    embedding[i][idx] += 1
                else:
                    embedding[i][idx] = 1

    return embedding, kmer_set, last_idx


@njit       
def get_embeddings(X, k, m, kmer_set, last_idx, alph_dict_n):
    n = len(X)
    d = len(X[0])
    precomputed = [ 2 for i in range(0) ]
    neighbours = [ [2 for i in range(0)] for j in range(last_idx) ]
    embedding = [typed.Dict.empty(types.int64,types.int64) for x in X]
    print("Generating mismatch embedding")
    for i,x in enumerate(X):
        for j in range(d - k + 1):
            kmer = encode(x[j: j + k], alph_dict_n)
            k_code =  encode_kmer(kmer)
            idx_code = kmer_set[k_code] 
            if k_code in precomputed:
              pass
            else:
                Mneighborhood = get_neighbours(kmer, m)
                #precomputed[k_code] = np.array([], dtype = np.int64) 
                for neighbor in Mneighborhood:
                  if neighbor in kmer_set:
                    pass
                  else:
                    kmer_set[neighbor] = last_idx
                    last_idx +=1
                    neighbours.append([2 for i in range(0)])
                   
                  #precomputed[k_code] = np.append(precomputed[k_code], kmer_set[neighbor])
                  neighbours[idx_code].append(kmer_set[neighbor])
                  precomputed.append(k_code)
                  

            for idx in neighbours[idx_code]:
                if idx in embedding[i]:
                    embedding[i][idx] += 1
                else:
                    embedding[i][idx] = 1

    return embedding, kmer_set, last_idx




@njit       
def get_embeddings_approximated(X, k, m, kmer_set, last_idx, alph_dict_n):
    n = len(X)
    d = len(X[0])
    precomputed = [ 2 for i in range(0) ]
    neighbours = [ [2 for i in range(0)] for j in range(last_idx) ]
    embedding = [typed.Dict.empty(types.int64,types.int64) for x in X]
    print("Generating mismatch embedding")
    for i,x in enumerate(X):
        for j in range(d - k + 1):
            kmer = encode(x[j: j + k], alph_dict_n)
            k_code =  encode_kmer(kmer)
            idx_code = kmer_set[k_code] 
            if k_code in precomputed:
              pass
            else:
                Mneighborhood = get_neighbours(kmer, m)
                for neighbor in Mneighborhood:
                  if neighbor in kmer_set:
                      neighbours[idx_code].append(kmer_set[neighbor])
                precomputed.append(k_code)
                  

            for idx in neighbours[idx_code]:
                if idx in embedding[i]:
                    embedding[i][idx] += 1
                else:
                    embedding[i][idx] = 1

    return embedding, kmer_set, last_idx


@njit
def get_mismatch(x,y):
    prod_scal = 0
    for idx in x:
        if idx in y:
            prod_scal += x[idx]*y[idx]
    return prod_scal

@njit
def get_gram_mismatch(embedding, show_every):
    n = len(embedding)
    K = np.zeros((n,n))
    size = n*(n+1)/2
    progress = size//show_every
    counter = 0
    for i in range(n):
      for j in range(i,n):
        x = embedding[i]
        y = embedding[j]
        value = get_mismatch(x,y)
        K[i,j] = K[j,i] = value
        counter+=1
        if counter % progress == 0:
          print( (counter/size*10000//1)/100, '%...')
    return K


@njit
def get_pairwise_mismatch(embedding1, embedding2, show_every):
    n = len(embedding1)
    m = len(embedding2)
    K = np.zeros((n,m))
    size = n*m
    progress = size//show_every
    counter = 0
    for i in range(n):
      for j in range(m):
        x = embedding1[i]
        y = embedding2[j]
        value = get_mismatch(x,y)
        K[i,j] = value
        counter+=1
        if counter % progress == 0:
          print( (counter/size*10000//1)/100, '%...')
    return K


@njit
def normalize_K(K):
    """
    Normalize kernel
    :param K: np.array
    :return: np.array
    """
    if K[0, 0] == 1:
        print('Kernel already normalized')
    else:
        n = K.shape[0]
        diag = np.sqrt(np.diag(K))
        for i in range(n):
            d = diag[i]
            for j in range(i+1, n):
                K[i, j] /= (d * diag[j])
                K[j, i] = K[i, j]
        np.fill_diagonal(K, np.ones(n))
    return K



def create_submission(predictions, submission_file_name):
    data = np.where(predictions == 1, 1, 0)
    data_frame = pd.DataFrame(data, columns = ['Bound'])
    data_frame.index.name = 'Id'
    data_frame.to_csv(submission_file_name)

