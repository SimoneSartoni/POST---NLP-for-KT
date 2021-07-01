import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def identity_tokenizer(text):
    return text


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')


def tf_idf(texts):
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=identity_tokenizer,
        preprocessor=identity_tokenizer,
        token_pattern=None,
        use_idf = True)
    tfidf_vectorizer_vectors_existing_words_only = tfidf_vectorizer.fit_transform(texts)
    df_tf_idf = pd.DataFrame.sparse.from_spmatrix(tfidf_vectorizer_vectors_existing_words_only)
    sparse_tf_idf = tfidf_vectorizer
    dataframe_tf_idf = df_tf_idf
    words_unique = tfidf_vectorizer_existing_words_only.get_feature_names()
    # Save sparse matrix in current directory
    data_folder = './'

    sps.save_npz(os.path.join(data_folder, 'pro_words.npz'), sparse_tf_idf)

    words_dict = dict({})
    for i in range(0, len(words_unique)):
        words_dict[str(i)] = words_unique[i]
    write_txt(os.path.join(data_folder, 'words_set.txt'), words_unique)
    pro_num = dataframe_tf_idf.shape[0]
    words_num = dataframe_tf_idf.shape[1]
    shrink = 10
    topK = 100
    normalize = True
    similarity = "cosine"
    similarity_matrix = Compute_Similarity(sparse_tf_idf.T, shrink=shrink, topK=topK, normalize=normalize,
                                           similarity=similarity).compute_similarity()
    sps.save_npz(os.path.join(data_folder, 'TF_IDF_pro_pro_'+str(shrink)+str(topK)+str(normalize)+'.npz'), similarity_matrix)



