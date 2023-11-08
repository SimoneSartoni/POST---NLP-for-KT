# Prediction Oriented Self-attention for Knowledge Tracing (POST), NLP-enhanced DKT, NLP-enhanced POST and Hybrid NLP-DKT
Repository for Master Degree thesis on "A study of possible improvements in Knowledge Tracing with Natural Language Processing and self-attention".

In this research work, we studied three directions to improve Knowledge Tracing (KT):
- Proposing Prediction Oriented Self-attentive knowledge Tracing (POST), a new state-of-the-art model for KT outperforming both DKT and SAINT+ architectures.
- Using six Natural Language Processing methods (CountVectorizer, word2vec, doc2vec, DistilBERT, Sentence Transformer and BERTopic) to create embeddings from exercises' texts. Then developing "NLP-enhanced" versions of DKT and POST, able to use as additional information the created textual embeddings. We propose two methods to create hybrid models too, able to sue at the same time the exercise embeddings created by multiple NLP methods.


## Here POST and "NLP-enhanced" POST architectures:


- ![POST architecture](https://github.com/SimoneSartoni/POST---NLP-for-KT/blob/main/Knowledge_Tracing/analysis_and_results/images/proposed%20models/post.png)
- ![NLP-POST architecture](https://github.com/SimoneSartoni/POST---NLP-for-KT/blob/main/Knowledge_Tracing/analysis_and_results/images/proposed%20models/nlp_post.png)


## RESULTS
All our models have provided improvements to KT. Here the results of our best models on the four datasets used for evaluation:


![Results of our models on ASSISTments 2009 dataset](https://github.com/SimoneSartoni/POST---NLP-for-KT/blob/main/Knowledge_Tracing/analysis_and_results/images/results/2009_best_models.png)
![Results of our models on ASSISTments 2012 dataset](https://github.com/SimoneSartoni/POST---NLP-for-KT/blob/main/Knowledge_Tracing/analysis_and_results/images/results/2012_best_models.png)
![Results of our models on Cloud Academy dataset](https://github.com/SimoneSartoni/POST---NLP-for-KT/blob/main/Knowledge_Tracing/analysis_and_results/images/results/Cloud_Academy_best_models.png)
![Results of our models on Peking Online Judge dataset](https://github.com/SimoneSartoni/POST---NLP-for-KT/blob/main/Knowledge_Tracing/analysis_and_results/images/results/Peking_Online_Judge_best_models.png)

## CITE THIS REPOSITORY
If you make use of any part of code from this repository, please cite this repository in your work.

@software{ 
  SARTONI_POST---NLP-for-KT_2022,
  author = {SARTONI, SIMONE},
  month = apr,
  title = {{POST---NLP-for-KT}},
  url = {https://github.com/SimoneSartoni/POST---NLP-for-KT},
  version = {1.0.0},
  year = {2022}
}
