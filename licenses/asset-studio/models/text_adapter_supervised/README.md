---
library_name: peft
license: mit
language:
- en
pipeline_tag: sentence-similarity
tags:
- text-embedding
- embeddings
- information-retrieval
- beir
- text-classification
- language-model
- text-clustering
- text-semantic-similarity
- text-evaluation
- text-reranking
- feature-extraction
- sentence-similarity
- Sentence Similarity
- natural_questions
- ms_marco
- fever
- hotpot_qa
- mteb
model-index:
- name: LLM2Vec-Meta-Llama-3-supervised
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en)
      config: en
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 79.94029850746269
    - type: ap
      value: 44.93223506764482
    - type: f1
      value: 74.30328994013465
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_polarity
      name: MTEB AmazonPolarityClassification
      config: default
      split: test
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
    metrics:
    - type: accuracy
      value: 86.06680000000001
    - type: ap
      value: 81.97124658709345
    - type: f1
      value: 86.00558036874241
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (en)
      config: en
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 46.836
    - type: f1
      value: 46.05094679201488
  - task:
      type: Retrieval
    dataset:
      type: arguana
      name: MTEB ArguAna
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 37.980000000000004
    - type: map_at_10
      value: 54.167
    - type: map_at_100
      value: 54.735
    - type: map_at_1000
      value: 54.738
    - type: map_at_3
      value: 49.384
    - type: map_at_5
      value: 52.285000000000004
    - type: mrr_at_1
      value: 38.549
    - type: mrr_at_10
      value: 54.351000000000006
    - type: mrr_at_100
      value: 54.932
    - type: mrr_at_1000
      value: 54.935
    - type: mrr_at_3
      value: 49.585
    - type: mrr_at_5
      value: 52.469
    - type: ndcg_at_1
      value: 37.980000000000004
    - type: ndcg_at_10
      value: 62.778999999999996
    - type: ndcg_at_100
      value: 64.986
    - type: ndcg_at_1000
      value: 65.036
    - type: ndcg_at_3
      value: 53.086999999999996
    - type: ndcg_at_5
      value: 58.263
    - type: precision_at_1
      value: 37.980000000000004
    - type: precision_at_10
      value: 9.011
    - type: precision_at_100
      value: 0.993
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 21.266
    - type: precision_at_5
      value: 15.248999999999999
    - type: recall_at_1
      value: 37.980000000000004
    - type: recall_at_10
      value: 90.114
    - type: recall_at_100
      value: 99.289
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 63.798
    - type: recall_at_5
      value: 76.24499999999999
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-p2p
      name: MTEB ArxivClusteringP2P
      config: default
      split: test
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
    metrics:
    - type: v_measure
      value: 44.27081216556421
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-s2s
      name: MTEB ArxivClusteringS2S
      config: default
      split: test
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
    metrics:
    - type: v_measure
      value: 46.8490872532913
  - task:
      type: Reranking
    dataset:
      type: mteb/askubuntudupquestions-reranking
      name: MTEB AskUbuntuDupQuestions
      config: default
      split: test
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
    metrics:
    - type: map
      value: 65.18525400430678
    - type: mrr
      value: 78.80149936244119
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_spearman
      value: 84.92301936595548
  - task:
      type: Classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77Classification
      config: default
      split: test
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
    metrics:
    - type: accuracy
      value: 88.0487012987013
    - type: f1
      value: 88.00953788281542
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-p2p
      name: MTEB BiorxivClusteringP2P
      config: default
      split: test
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
    metrics:
    - type: v_measure
      value: 32.34687321141145
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-s2s
      name: MTEB BiorxivClusteringS2S
      config: default
      split: test
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
    metrics:
    - type: v_measure
      value: 36.69881680534123
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/android
      name: MTEB CQADupstackAndroidRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 37.742
    - type: map_at_10
      value: 51.803
    - type: map_at_100
      value: 53.556000000000004
    - type: map_at_1000
      value: 53.652
    - type: map_at_3
      value: 47.286
    - type: map_at_5
      value: 50.126000000000005
    - type: mrr_at_1
      value: 46.924
    - type: mrr_at_10
      value: 57.857
    - type: mrr_at_100
      value: 58.592
    - type: mrr_at_1000
      value: 58.619
    - type: mrr_at_3
      value: 55.340999999999994
    - type: mrr_at_5
      value: 57.150999999999996
    - type: ndcg_at_1
      value: 46.924
    - type: ndcg_at_10
      value: 58.733999999999995
    - type: ndcg_at_100
      value: 63.771
    - type: ndcg_at_1000
      value: 64.934
    - type: ndcg_at_3
      value: 53.189
    - type: ndcg_at_5
      value: 56.381
    - type: precision_at_1
      value: 46.924
    - type: precision_at_10
      value: 11.431
    - type: precision_at_100
      value: 1.73
    - type: precision_at_1000
      value: 0.213
    - type: precision_at_3
      value: 25.942
    - type: precision_at_5
      value: 19.113
    - type: recall_at_1
      value: 37.742
    - type: recall_at_10
      value: 71.34
    - type: recall_at_100
      value: 91.523
    - type: recall_at_1000
      value: 98.494
    - type: recall_at_3
      value: 55.443
    - type: recall_at_5
      value: 64.122
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/english
      name: MTEB CQADupstackEnglishRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 34.183
    - type: map_at_10
      value: 46.837
    - type: map_at_100
      value: 48.126000000000005
    - type: map_at_1000
      value: 48.25
    - type: map_at_3
      value: 43.171
    - type: map_at_5
      value: 45.318999999999996
    - type: mrr_at_1
      value: 43.376
    - type: mrr_at_10
      value: 52.859
    - type: mrr_at_100
      value: 53.422000000000004
    - type: mrr_at_1000
      value: 53.456
    - type: mrr_at_3
      value: 50.434999999999995
    - type: mrr_at_5
      value: 51.861999999999995
    - type: ndcg_at_1
      value: 43.376
    - type: ndcg_at_10
      value: 53.223
    - type: ndcg_at_100
      value: 57.175
    - type: ndcg_at_1000
      value: 58.86900000000001
    - type: ndcg_at_3
      value: 48.417
    - type: ndcg_at_5
      value: 50.77
    - type: precision_at_1
      value: 43.376
    - type: precision_at_10
      value: 10.236
    - type: precision_at_100
      value: 1.5730000000000002
    - type: precision_at_1000
      value: 0.203
    - type: precision_at_3
      value: 23.97
    - type: precision_at_5
      value: 17.134
    - type: recall_at_1
      value: 34.183
    - type: recall_at_10
      value: 64.866
    - type: recall_at_100
      value: 81.26100000000001
    - type: recall_at_1000
      value: 91.412
    - type: recall_at_3
      value: 50.080000000000005
    - type: recall_at_5
      value: 56.871
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/gaming
      name: MTEB CQADupstackGamingRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 44.878
    - type: map_at_10
      value: 58.656
    - type: map_at_100
      value: 59.668
    - type: map_at_1000
      value: 59.704
    - type: map_at_3
      value: 54.891
    - type: map_at_5
      value: 57.050999999999995
    - type: mrr_at_1
      value: 51.975
    - type: mrr_at_10
      value: 62.357
    - type: mrr_at_100
      value: 62.907999999999994
    - type: mrr_at_1000
      value: 62.925
    - type: mrr_at_3
      value: 59.801
    - type: mrr_at_5
      value: 61.278
    - type: ndcg_at_1
      value: 51.975
    - type: ndcg_at_10
      value: 64.95100000000001
    - type: ndcg_at_100
      value: 68.414
    - type: ndcg_at_1000
      value: 69.077
    - type: ndcg_at_3
      value: 58.897999999999996
    - type: ndcg_at_5
      value: 61.866
    - type: precision_at_1
      value: 51.975
    - type: precision_at_10
      value: 10.502
    - type: precision_at_100
      value: 1.31
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 26.290000000000003
    - type: precision_at_5
      value: 18.093999999999998
    - type: recall_at_1
      value: 44.878
    - type: recall_at_10
      value: 79.746
    - type: recall_at_100
      value: 94.17
    - type: recall_at_1000
      value: 98.80499999999999
    - type: recall_at_3
      value: 63.70099999999999
    - type: recall_at_5
      value: 70.878
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/gis
      name: MTEB CQADupstackGisRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 28.807
    - type: map_at_10
      value: 39.431
    - type: map_at_100
      value: 40.56
    - type: map_at_1000
      value: 40.617999999999995
    - type: map_at_3
      value: 36.436
    - type: map_at_5
      value: 37.955
    - type: mrr_at_1
      value: 31.186000000000003
    - type: mrr_at_10
      value: 41.654
    - type: mrr_at_100
      value: 42.58
    - type: mrr_at_1000
      value: 42.623
    - type: mrr_at_3
      value: 38.983000000000004
    - type: mrr_at_5
      value: 40.35
    - type: ndcg_at_1
      value: 31.186000000000003
    - type: ndcg_at_10
      value: 45.297
    - type: ndcg_at_100
      value: 50.515
    - type: ndcg_at_1000
      value: 52.005
    - type: ndcg_at_3
      value: 39.602
    - type: ndcg_at_5
      value: 42.027
    - type: precision_at_1
      value: 31.186000000000003
    - type: precision_at_10
      value: 7.073
    - type: precision_at_100
      value: 1.0210000000000001
    - type: precision_at_1000
      value: 0.11900000000000001
    - type: precision_at_3
      value: 17.1
    - type: precision_at_5
      value: 11.729000000000001
    - type: recall_at_1
      value: 28.807
    - type: recall_at_10
      value: 61.138999999999996
    - type: recall_at_100
      value: 84.491
    - type: recall_at_1000
      value: 95.651
    - type: recall_at_3
      value: 45.652
    - type: recall_at_5
      value: 51.522
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/mathematica
      name: MTEB CQADupstackMathematicaRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 20.607
    - type: map_at_10
      value: 31.944
    - type: map_at_100
      value: 33.317
    - type: map_at_1000
      value: 33.428000000000004
    - type: map_at_3
      value: 28.508
    - type: map_at_5
      value: 30.348999999999997
    - type: mrr_at_1
      value: 25.622
    - type: mrr_at_10
      value: 36.726
    - type: mrr_at_100
      value: 37.707
    - type: mrr_at_1000
      value: 37.761
    - type: mrr_at_3
      value: 33.934
    - type: mrr_at_5
      value: 35.452
    - type: ndcg_at_1
      value: 25.622
    - type: ndcg_at_10
      value: 38.462
    - type: ndcg_at_100
      value: 44.327
    - type: ndcg_at_1000
      value: 46.623
    - type: ndcg_at_3
      value: 32.583
    - type: ndcg_at_5
      value: 35.175
    - type: precision_at_1
      value: 25.622
    - type: precision_at_10
      value: 7.425
    - type: precision_at_100
      value: 1.173
    - type: precision_at_1000
      value: 0.149
    - type: precision_at_3
      value: 16.418
    - type: precision_at_5
      value: 11.866
    - type: recall_at_1
      value: 20.607
    - type: recall_at_10
      value: 53.337
    - type: recall_at_100
      value: 78.133
    - type: recall_at_1000
      value: 94.151
    - type: recall_at_3
      value: 37.088
    - type: recall_at_5
      value: 43.627
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/physics
      name: MTEB CQADupstackPhysicsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 33.814
    - type: map_at_10
      value: 47.609
    - type: map_at_100
      value: 48.972
    - type: map_at_1000
      value: 49.061
    - type: map_at_3
      value: 43.397999999999996
    - type: map_at_5
      value: 45.839
    - type: mrr_at_1
      value: 42.059999999999995
    - type: mrr_at_10
      value: 53.074
    - type: mrr_at_100
      value: 53.76800000000001
    - type: mrr_at_1000
      value: 53.794
    - type: mrr_at_3
      value: 50.241
    - type: mrr_at_5
      value: 51.805
    - type: ndcg_at_1
      value: 42.059999999999995
    - type: ndcg_at_10
      value: 54.419
    - type: ndcg_at_100
      value: 59.508
    - type: ndcg_at_1000
      value: 60.858000000000004
    - type: ndcg_at_3
      value: 48.296
    - type: ndcg_at_5
      value: 51.28
    - type: precision_at_1
      value: 42.059999999999995
    - type: precision_at_10
      value: 10.231
    - type: precision_at_100
      value: 1.4789999999999999
    - type: precision_at_1000
      value: 0.17700000000000002
    - type: precision_at_3
      value: 23.419999999999998
    - type: precision_at_5
      value: 16.843
    - type: recall_at_1
      value: 33.814
    - type: recall_at_10
      value: 68.88
    - type: recall_at_100
      value: 89.794
    - type: recall_at_1000
      value: 98.058
    - type: recall_at_3
      value: 51.915
    - type: recall_at_5
      value: 59.704
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/programmers
      name: MTEB CQADupstackProgrammersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 29.668
    - type: map_at_10
      value: 43.032
    - type: map_at_100
      value: 44.48
    - type: map_at_1000
      value: 44.574000000000005
    - type: map_at_3
      value: 38.609
    - type: map_at_5
      value: 41.164
    - type: mrr_at_1
      value: 37.785000000000004
    - type: mrr_at_10
      value: 48.898
    - type: mrr_at_100
      value: 49.728
    - type: mrr_at_1000
      value: 49.769000000000005
    - type: mrr_at_3
      value: 45.909
    - type: mrr_at_5
      value: 47.61
    - type: ndcg_at_1
      value: 37.785000000000004
    - type: ndcg_at_10
      value: 50.21099999999999
    - type: ndcg_at_100
      value: 55.657999999999994
    - type: ndcg_at_1000
      value: 57.172
    - type: ndcg_at_3
      value: 43.726
    - type: ndcg_at_5
      value: 46.758
    - type: precision_at_1
      value: 37.785000000000004
    - type: precision_at_10
      value: 9.669
    - type: precision_at_100
      value: 1.4409999999999998
    - type: precision_at_1000
      value: 0.174
    - type: precision_at_3
      value: 21.651
    - type: precision_at_5
      value: 15.822
    - type: recall_at_1
      value: 29.668
    - type: recall_at_10
      value: 65.575
    - type: recall_at_100
      value: 87.977
    - type: recall_at_1000
      value: 97.615
    - type: recall_at_3
      value: 47.251
    - type: recall_at_5
      value: 55.359
  - task:
      type: Retrieval
    dataset:
      type: mteb/cqadupstack
      name: MTEB CQADupstackRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 30.29925
    - type: map_at_10
      value: 41.98708333333333
    - type: map_at_100
      value: 43.306916666666666
    - type: map_at_1000
      value: 43.40716666666667
    - type: map_at_3
      value: 38.431666666666665
    - type: map_at_5
      value: 40.4195
    - type: mrr_at_1
      value: 36.24483333333334
    - type: mrr_at_10
      value: 46.32666666666667
    - type: mrr_at_100
      value: 47.13983333333333
    - type: mrr_at_1000
      value: 47.18058333333334
    - type: mrr_at_3
      value: 43.66799999999999
    - type: mrr_at_5
      value: 45.163666666666664
    - type: ndcg_at_1
      value: 36.24483333333334
    - type: ndcg_at_10
      value: 48.251916666666666
    - type: ndcg_at_100
      value: 53.3555
    - type: ndcg_at_1000
      value: 55.024249999999995
    - type: ndcg_at_3
      value: 42.599583333333335
    - type: ndcg_at_5
      value: 45.24166666666666
    - type: precision_at_1
      value: 36.24483333333334
    - type: precision_at_10
      value: 8.666833333333333
    - type: precision_at_100
      value: 1.3214166666666665
    - type: precision_at_1000
      value: 0.16475
    - type: precision_at_3
      value: 19.9955
    - type: precision_at_5
      value: 14.271999999999998
    - type: recall_at_1
      value: 30.29925
    - type: recall_at_10
      value: 62.232333333333344
    - type: recall_at_100
      value: 84.151
    - type: recall_at_1000
      value: 95.37333333333333
    - type: recall_at_3
      value: 46.45541666666667
    - type: recall_at_5
      value: 53.264
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/stats
      name: MTEB CQADupstackStatsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 28.996
    - type: map_at_10
      value: 38.047
    - type: map_at_100
      value: 39.121
    - type: map_at_1000
      value: 39.202999999999996
    - type: map_at_3
      value: 35.376000000000005
    - type: map_at_5
      value: 36.763
    - type: mrr_at_1
      value: 32.362
    - type: mrr_at_10
      value: 40.717999999999996
    - type: mrr_at_100
      value: 41.586
    - type: mrr_at_1000
      value: 41.641
    - type: mrr_at_3
      value: 38.292
    - type: mrr_at_5
      value: 39.657
    - type: ndcg_at_1
      value: 32.362
    - type: ndcg_at_10
      value: 43.105
    - type: ndcg_at_100
      value: 48.026
    - type: ndcg_at_1000
      value: 49.998
    - type: ndcg_at_3
      value: 38.147999999999996
    - type: ndcg_at_5
      value: 40.385
    - type: precision_at_1
      value: 32.362
    - type: precision_at_10
      value: 6.7940000000000005
    - type: precision_at_100
      value: 1.0170000000000001
    - type: precision_at_1000
      value: 0.125
    - type: precision_at_3
      value: 16.411
    - type: precision_at_5
      value: 11.35
    - type: recall_at_1
      value: 28.996
    - type: recall_at_10
      value: 55.955
    - type: recall_at_100
      value: 77.744
    - type: recall_at_1000
      value: 92.196
    - type: recall_at_3
      value: 42.254999999999995
    - type: recall_at_5
      value: 47.776
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/tex
      name: MTEB CQADupstackTexRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 20.029
    - type: map_at_10
      value: 29.188
    - type: map_at_100
      value: 30.484
    - type: map_at_1000
      value: 30.608
    - type: map_at_3
      value: 26.195
    - type: map_at_5
      value: 27.866999999999997
    - type: mrr_at_1
      value: 24.57
    - type: mrr_at_10
      value: 33.461
    - type: mrr_at_100
      value: 34.398
    - type: mrr_at_1000
      value: 34.464
    - type: mrr_at_3
      value: 30.856
    - type: mrr_at_5
      value: 32.322
    - type: ndcg_at_1
      value: 24.57
    - type: ndcg_at_10
      value: 34.846
    - type: ndcg_at_100
      value: 40.544000000000004
    - type: ndcg_at_1000
      value: 43.019
    - type: ndcg_at_3
      value: 29.683999999999997
    - type: ndcg_at_5
      value: 32.11
    - type: precision_at_1
      value: 24.57
    - type: precision_at_10
      value: 6.535
    - type: precision_at_100
      value: 1.11
    - type: precision_at_1000
      value: 0.149
    - type: precision_at_3
      value: 14.338000000000001
    - type: precision_at_5
      value: 10.496
    - type: recall_at_1
      value: 20.029
    - type: recall_at_10
      value: 47.509
    - type: recall_at_100
      value: 72.61999999999999
    - type: recall_at_1000
      value: 89.778
    - type: recall_at_3
      value: 33.031
    - type: recall_at_5
      value: 39.306000000000004
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/unix
      name: MTEB CQADupstackUnixRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 31.753999999999998
    - type: map_at_10
      value: 43.814
    - type: map_at_100
      value: 45.072
    - type: map_at_1000
      value: 45.155
    - type: map_at_3
      value: 40.316
    - type: map_at_5
      value: 42.15
    - type: mrr_at_1
      value: 38.06
    - type: mrr_at_10
      value: 48.311
    - type: mrr_at_100
      value: 49.145
    - type: mrr_at_1000
      value: 49.181000000000004
    - type: mrr_at_3
      value: 45.678000000000004
    - type: mrr_at_5
      value: 47.072
    - type: ndcg_at_1
      value: 38.06
    - type: ndcg_at_10
      value: 50.083
    - type: ndcg_at_100
      value: 55.342
    - type: ndcg_at_1000
      value: 56.87
    - type: ndcg_at_3
      value: 44.513999999999996
    - type: ndcg_at_5
      value: 46.886
    - type: precision_at_1
      value: 38.06
    - type: precision_at_10
      value: 8.638
    - type: precision_at_100
      value: 1.253
    - type: precision_at_1000
      value: 0.149
    - type: precision_at_3
      value: 20.709
    - type: precision_at_5
      value: 14.44
    - type: recall_at_1
      value: 31.753999999999998
    - type: recall_at_10
      value: 64.473
    - type: recall_at_100
      value: 86.832
    - type: recall_at_1000
      value: 96.706
    - type: recall_at_3
      value: 48.937000000000005
    - type: recall_at_5
      value: 55.214
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/webmasters
      name: MTEB CQADupstackWebmastersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 28.815
    - type: map_at_10
      value: 40.595
    - type: map_at_100
      value: 42.337
    - type: map_at_1000
      value: 42.559000000000005
    - type: map_at_3
      value: 37.120999999999995
    - type: map_at_5
      value: 38.912
    - type: mrr_at_1
      value: 34.585
    - type: mrr_at_10
      value: 45.068000000000005
    - type: mrr_at_100
      value: 45.93
    - type: mrr_at_1000
      value: 45.974
    - type: mrr_at_3
      value: 42.26
    - type: mrr_at_5
      value: 43.742
    - type: ndcg_at_1
      value: 34.585
    - type: ndcg_at_10
      value: 47.519
    - type: ndcg_at_100
      value: 53.102000000000004
    - type: ndcg_at_1000
      value: 54.949999999999996
    - type: ndcg_at_3
      value: 41.719
    - type: ndcg_at_5
      value: 44.17
    - type: precision_at_1
      value: 34.585
    - type: precision_at_10
      value: 9.368
    - type: precision_at_100
      value: 1.7870000000000001
    - type: precision_at_1000
      value: 0.254
    - type: precision_at_3
      value: 19.895
    - type: precision_at_5
      value: 14.506
    - type: recall_at_1
      value: 28.815
    - type: recall_at_10
      value: 61.414
    - type: recall_at_100
      value: 85.922
    - type: recall_at_1000
      value: 97.15
    - type: recall_at_3
      value: 45.076
    - type: recall_at_5
      value: 51.271
  - task:
      type: Retrieval
    dataset:
      type: cqadupstack/wordpress
      name: MTEB CQADupstackWordpressRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 24.298000000000002
    - type: map_at_10
      value: 32.889
    - type: map_at_100
      value: 33.989999999999995
    - type: map_at_1000
      value: 34.074
    - type: map_at_3
      value: 29.873
    - type: map_at_5
      value: 31.539
    - type: mrr_at_1
      value: 26.433
    - type: mrr_at_10
      value: 34.937000000000005
    - type: mrr_at_100
      value: 35.914
    - type: mrr_at_1000
      value: 35.96
    - type: mrr_at_3
      value: 32.286
    - type: mrr_at_5
      value: 33.663
    - type: ndcg_at_1
      value: 26.433
    - type: ndcg_at_10
      value: 38.173
    - type: ndcg_at_100
      value: 43.884
    - type: ndcg_at_1000
      value: 45.916000000000004
    - type: ndcg_at_3
      value: 32.419
    - type: ndcg_at_5
      value: 35.092
    - type: precision_at_1
      value: 26.433
    - type: precision_at_10
      value: 6.1
    - type: precision_at_100
      value: 0.963
    - type: precision_at_1000
      value: 0.126
    - type: precision_at_3
      value: 13.802
    - type: precision_at_5
      value: 9.871
    - type: recall_at_1
      value: 24.298000000000002
    - type: recall_at_10
      value: 52.554
    - type: recall_at_100
      value: 79.345
    - type: recall_at_1000
      value: 94.464
    - type: recall_at_3
      value: 37.036
    - type: recall_at_5
      value: 43.518
  - task:
      type: Retrieval
    dataset:
      type: climate-fever
      name: MTEB ClimateFEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 14.194999999999999
    - type: map_at_10
      value: 24.563
    - type: map_at_100
      value: 26.775
    - type: map_at_1000
      value: 26.965
    - type: map_at_3
      value: 19.983999999999998
    - type: map_at_5
      value: 22.24
    - type: mrr_at_1
      value: 31.661
    - type: mrr_at_10
      value: 44.804
    - type: mrr_at_100
      value: 45.655
    - type: mrr_at_1000
      value: 45.678000000000004
    - type: mrr_at_3
      value: 41.292
    - type: mrr_at_5
      value: 43.468
    - type: ndcg_at_1
      value: 31.661
    - type: ndcg_at_10
      value: 34.271
    - type: ndcg_at_100
      value: 42.04
    - type: ndcg_at_1000
      value: 45.101
    - type: ndcg_at_3
      value: 27.529999999999998
    - type: ndcg_at_5
      value: 29.862
    - type: precision_at_1
      value: 31.661
    - type: precision_at_10
      value: 10.925
    - type: precision_at_100
      value: 1.92
    - type: precision_at_1000
      value: 0.25
    - type: precision_at_3
      value: 20.456
    - type: precision_at_5
      value: 16.012999999999998
    - type: recall_at_1
      value: 14.194999999999999
    - type: recall_at_10
      value: 41.388999999999996
    - type: recall_at_100
      value: 67.58800000000001
    - type: recall_at_1000
      value: 84.283
    - type: recall_at_3
      value: 25.089
    - type: recall_at_5
      value: 31.642
  - task:
      type: Retrieval
    dataset:
      type: dbpedia-entity
      name: MTEB DBPedia
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 9.898
    - type: map_at_10
      value: 23.226
    - type: map_at_100
      value: 33.372
    - type: map_at_1000
      value: 35.407
    - type: map_at_3
      value: 15.892999999999999
    - type: map_at_5
      value: 18.747
    - type: mrr_at_1
      value: 73.5
    - type: mrr_at_10
      value: 80.404
    - type: mrr_at_100
      value: 80.671
    - type: mrr_at_1000
      value: 80.676
    - type: mrr_at_3
      value: 78.958
    - type: mrr_at_5
      value: 79.683
    - type: ndcg_at_1
      value: 62.0
    - type: ndcg_at_10
      value: 48.337
    - type: ndcg_at_100
      value: 53.474
    - type: ndcg_at_1000
      value: 60.999
    - type: ndcg_at_3
      value: 52.538
    - type: ndcg_at_5
      value: 49.659
    - type: precision_at_1
      value: 73.5
    - type: precision_at_10
      value: 39.25
    - type: precision_at_100
      value: 12.4
    - type: precision_at_1000
      value: 2.4459999999999997
    - type: precision_at_3
      value: 56.333
    - type: precision_at_5
      value: 48.15
    - type: recall_at_1
      value: 9.898
    - type: recall_at_10
      value: 29.511
    - type: recall_at_100
      value: 60.45700000000001
    - type: recall_at_1000
      value: 84.47200000000001
    - type: recall_at_3
      value: 17.064
    - type: recall_at_5
      value: 21.258
  - task:
      type: Classification
    dataset:
      type: mteb/emotion
      name: MTEB EmotionClassification
      config: default
      split: test
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
    metrics:
    - type: accuracy
      value: 51.19999999999999
    - type: f1
      value: 46.23854137552949
  - task:
      type: Retrieval
    dataset:
      type: fever
      name: MTEB FEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 80.093
    - type: map_at_10
      value: 87.139
    - type: map_at_100
      value: 87.333
    - type: map_at_1000
      value: 87.344
    - type: map_at_3
      value: 86.395
    - type: map_at_5
      value: 86.866
    - type: mrr_at_1
      value: 86.36399999999999
    - type: mrr_at_10
      value: 91.867
    - type: mrr_at_100
      value: 91.906
    - type: mrr_at_1000
      value: 91.90700000000001
    - type: mrr_at_3
      value: 91.484
    - type: mrr_at_5
      value: 91.759
    - type: ndcg_at_1
      value: 86.36399999999999
    - type: ndcg_at_10
      value: 90.197
    - type: ndcg_at_100
      value: 90.819
    - type: ndcg_at_1000
      value: 91.01599999999999
    - type: ndcg_at_3
      value: 89.166
    - type: ndcg_at_5
      value: 89.74
    - type: precision_at_1
      value: 86.36399999999999
    - type: precision_at_10
      value: 10.537
    - type: precision_at_100
      value: 1.106
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 33.608
    - type: precision_at_5
      value: 20.618
    - type: recall_at_1
      value: 80.093
    - type: recall_at_10
      value: 95.003
    - type: recall_at_100
      value: 97.328
    - type: recall_at_1000
      value: 98.485
    - type: recall_at_3
      value: 92.072
    - type: recall_at_5
      value: 93.661
  - task:
      type: Retrieval
    dataset:
      type: fiqa
      name: MTEB FiQA2018
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 29.063
    - type: map_at_10
      value: 47.113
    - type: map_at_100
      value: 49.294
    - type: map_at_1000
      value: 49.422
    - type: map_at_3
      value: 40.955000000000005
    - type: map_at_5
      value: 44.5
    - type: mrr_at_1
      value: 55.401
    - type: mrr_at_10
      value: 62.99400000000001
    - type: mrr_at_100
      value: 63.63999999999999
    - type: mrr_at_1000
      value: 63.661
    - type: mrr_at_3
      value: 61.034
    - type: mrr_at_5
      value: 62.253
    - type: ndcg_at_1
      value: 55.401
    - type: ndcg_at_10
      value: 55.332
    - type: ndcg_at_100
      value: 61.931000000000004
    - type: ndcg_at_1000
      value: 63.841
    - type: ndcg_at_3
      value: 50.92
    - type: ndcg_at_5
      value: 52.525
    - type: precision_at_1
      value: 55.401
    - type: precision_at_10
      value: 15.262
    - type: precision_at_100
      value: 2.231
    - type: precision_at_1000
      value: 0.256
    - type: precision_at_3
      value: 33.848
    - type: precision_at_5
      value: 25.031
    - type: recall_at_1
      value: 29.063
    - type: recall_at_10
      value: 62.498
    - type: recall_at_100
      value: 85.86
    - type: recall_at_1000
      value: 97.409
    - type: recall_at_3
      value: 45.472
    - type: recall_at_5
      value: 53.344
  - task:
      type: Retrieval
    dataset:
      type: hotpotqa
      name: MTEB HotpotQA
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 37.205
    - type: map_at_10
      value: 64.19399999999999
    - type: map_at_100
      value: 65.183
    - type: map_at_1000
      value: 65.23299999999999
    - type: map_at_3
      value: 60.239
    - type: map_at_5
      value: 62.695
    - type: mrr_at_1
      value: 74.409
    - type: mrr_at_10
      value: 80.84
    - type: mrr_at_100
      value: 81.10199999999999
    - type: mrr_at_1000
      value: 81.109
    - type: mrr_at_3
      value: 79.739
    - type: mrr_at_5
      value: 80.46600000000001
    - type: ndcg_at_1
      value: 74.409
    - type: ndcg_at_10
      value: 71.757
    - type: ndcg_at_100
      value: 75.152
    - type: ndcg_at_1000
      value: 76.098
    - type: ndcg_at_3
      value: 66.174
    - type: ndcg_at_5
      value: 69.283
    - type: precision_at_1
      value: 74.409
    - type: precision_at_10
      value: 15.503
    - type: precision_at_100
      value: 1.8110000000000002
    - type: precision_at_1000
      value: 0.194
    - type: precision_at_3
      value: 43.457
    - type: precision_at_5
      value: 28.532000000000004
    - type: recall_at_1
      value: 37.205
    - type: recall_at_10
      value: 77.515
    - type: recall_at_100
      value: 90.56
    - type: recall_at_1000
      value: 96.759
    - type: recall_at_3
      value: 65.18599999999999
    - type: recall_at_5
      value: 71.33
  - task:
      type: Classification
    dataset:
      type: mteb/imdb
      name: MTEB ImdbClassification
      config: default
      split: test
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
    metrics:
    - type: accuracy
      value: 82.9448
    - type: ap
      value: 78.25923353099166
    - type: f1
      value: 82.86422040179993
  - task:
      type: Retrieval
    dataset:
      type: msmarco
      name: MTEB MSMARCO
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 22.834
    - type: map_at_10
      value: 35.85
    - type: map_at_100
      value: 37.013
    - type: map_at_1000
      value: 37.056
    - type: map_at_3
      value: 31.613000000000003
    - type: map_at_5
      value: 34.113
    - type: mrr_at_1
      value: 23.424
    - type: mrr_at_10
      value: 36.398
    - type: mrr_at_100
      value: 37.498
    - type: mrr_at_1000
      value: 37.534
    - type: mrr_at_3
      value: 32.275999999999996
    - type: mrr_at_5
      value: 34.705000000000005
    - type: ndcg_at_1
      value: 23.424
    - type: ndcg_at_10
      value: 43.236999999999995
    - type: ndcg_at_100
      value: 48.776
    - type: ndcg_at_1000
      value: 49.778
    - type: ndcg_at_3
      value: 34.692
    - type: ndcg_at_5
      value: 39.119
    - type: precision_at_1
      value: 23.424
    - type: precision_at_10
      value: 6.918
    - type: precision_at_100
      value: 0.9690000000000001
    - type: precision_at_1000
      value: 0.105
    - type: precision_at_3
      value: 14.881
    - type: precision_at_5
      value: 11.183
    - type: recall_at_1
      value: 22.834
    - type: recall_at_10
      value: 66.03999999999999
    - type: recall_at_100
      value: 91.532
    - type: recall_at_1000
      value: 99.068
    - type: recall_at_3
      value: 42.936
    - type: recall_at_5
      value: 53.539
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (en)
      config: en
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 96.1377108983128
    - type: f1
      value: 95.87034720246666
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (en)
      config: en
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 86.10579115367078
    - type: f1
      value: 70.20810321445228
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (en)
      config: en
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 79.80497646267652
    - type: f1
      value: 77.32475274059293
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (en)
      config: en
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 81.52320107599192
    - type: f1
      value: 81.22312939311655
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-p2p
      name: MTEB MedrxivClusteringP2P
      config: default
      split: test
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
    metrics:
    - type: v_measure
      value: 30.709106678767018
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-s2s
      name: MTEB MedrxivClusteringS2S
      config: default
      split: test
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
    metrics:
    - type: v_measure
      value: 32.95879128399585
  - task:
      type: Reranking
    dataset:
      type: mteb/mind_small
      name: MTEB MindSmallReranking
      config: default
      split: test
      revision: 3bdac13927fdc888b903db93b2ffdbd90b295a69
    metrics:
    - type: map
      value: 32.67476691128679
    - type: mrr
      value: 33.921654478513986
  - task:
      type: Retrieval
    dataset:
      type: nfcorpus
      name: MTEB NFCorpus
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 7.223
    - type: map_at_10
      value: 15.992999999999999
    - type: map_at_100
      value: 21.09
    - type: map_at_1000
      value: 22.822
    - type: map_at_3
      value: 11.475
    - type: map_at_5
      value: 13.501
    - type: mrr_at_1
      value: 53.251000000000005
    - type: mrr_at_10
      value: 61.878
    - type: mrr_at_100
      value: 62.307
    - type: mrr_at_1000
      value: 62.342
    - type: mrr_at_3
      value: 60.01
    - type: mrr_at_5
      value: 61.202
    - type: ndcg_at_1
      value: 51.702999999999996
    - type: ndcg_at_10
      value: 41.833999999999996
    - type: ndcg_at_100
      value: 39.061
    - type: ndcg_at_1000
      value: 47.397
    - type: ndcg_at_3
      value: 47.083000000000006
    - type: ndcg_at_5
      value: 44.722
    - type: precision_at_1
      value: 53.251000000000005
    - type: precision_at_10
      value: 31.3
    - type: precision_at_100
      value: 10.254000000000001
    - type: precision_at_1000
      value: 2.338
    - type: precision_at_3
      value: 43.756
    - type: precision_at_5
      value: 38.824
    - type: recall_at_1
      value: 7.223
    - type: recall_at_10
      value: 20.529
    - type: recall_at_100
      value: 39.818
    - type: recall_at_1000
      value: 70.152
    - type: recall_at_3
      value: 12.666
    - type: recall_at_5
      value: 15.798000000000002
  - task:
      type: Retrieval
    dataset:
      type: nq
      name: MTEB NQ
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 38.847
    - type: map_at_10
      value: 56.255
    - type: map_at_100
      value: 57.019
    - type: map_at_1000
      value: 57.03
    - type: map_at_3
      value: 51.665000000000006
    - type: map_at_5
      value: 54.543
    - type: mrr_at_1
      value: 43.801
    - type: mrr_at_10
      value: 58.733999999999995
    - type: mrr_at_100
      value: 59.206
    - type: mrr_at_1000
      value: 59.21300000000001
    - type: mrr_at_3
      value: 55.266999999999996
    - type: mrr_at_5
      value: 57.449
    - type: ndcg_at_1
      value: 43.772
    - type: ndcg_at_10
      value: 64.213
    - type: ndcg_at_100
      value: 67.13
    - type: ndcg_at_1000
      value: 67.368
    - type: ndcg_at_3
      value: 55.977
    - type: ndcg_at_5
      value: 60.597
    - type: precision_at_1
      value: 43.772
    - type: precision_at_10
      value: 10.272
    - type: precision_at_100
      value: 1.193
    - type: precision_at_1000
      value: 0.121
    - type: precision_at_3
      value: 25.261
    - type: precision_at_5
      value: 17.885
    - type: recall_at_1
      value: 38.847
    - type: recall_at_10
      value: 85.76700000000001
    - type: recall_at_100
      value: 98.054
    - type: recall_at_1000
      value: 99.812
    - type: recall_at_3
      value: 64.82
    - type: recall_at_5
      value: 75.381
  - task:
      type: Retrieval
    dataset:
      type: quora
      name: MTEB QuoraRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 68.77
    - type: map_at_10
      value: 83.195
    - type: map_at_100
      value: 83.869
    - type: map_at_1000
      value: 83.883
    - type: map_at_3
      value: 80.04599999999999
    - type: map_at_5
      value: 82.011
    - type: mrr_at_1
      value: 79.2
    - type: mrr_at_10
      value: 85.942
    - type: mrr_at_100
      value: 86.063
    - type: mrr_at_1000
      value: 86.064
    - type: mrr_at_3
      value: 84.82
    - type: mrr_at_5
      value: 85.56899999999999
    - type: ndcg_at_1
      value: 79.17999999999999
    - type: ndcg_at_10
      value: 87.161
    - type: ndcg_at_100
      value: 88.465
    - type: ndcg_at_1000
      value: 88.553
    - type: ndcg_at_3
      value: 83.958
    - type: ndcg_at_5
      value: 85.699
    - type: precision_at_1
      value: 79.17999999999999
    - type: precision_at_10
      value: 13.401
    - type: precision_at_100
      value: 1.54
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 36.903000000000006
    - type: precision_at_5
      value: 24.404
    - type: recall_at_1
      value: 68.77
    - type: recall_at_10
      value: 95.132
    - type: recall_at_100
      value: 99.58200000000001
    - type: recall_at_1000
      value: 99.997
    - type: recall_at_3
      value: 86.119
    - type: recall_at_5
      value: 90.932
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering
      name: MTEB RedditClustering
      config: default
      split: test
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
    metrics:
    - type: v_measure
      value: 61.7204049654583
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering-p2p
      name: MTEB RedditClusteringP2P
      config: default
      split: test
      revision: 282350215ef01743dc01b456c7f5241fa8937f16
    metrics:
    - type: v_measure
      value: 63.98164986883849
  - task:
      type: Retrieval
    dataset:
      type: scidocs
      name: MTEB SCIDOCS
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 5.443
    - type: map_at_10
      value: 13.86
    - type: map_at_100
      value: 16.496
    - type: map_at_1000
      value: 16.836000000000002
    - type: map_at_3
      value: 9.661
    - type: map_at_5
      value: 11.745
    - type: mrr_at_1
      value: 26.8
    - type: mrr_at_10
      value: 37.777
    - type: mrr_at_100
      value: 38.928000000000004
    - type: mrr_at_1000
      value: 38.967
    - type: mrr_at_3
      value: 34.083000000000006
    - type: mrr_at_5
      value: 36.308
    - type: ndcg_at_1
      value: 26.8
    - type: ndcg_at_10
      value: 22.961000000000002
    - type: ndcg_at_100
      value: 32.582
    - type: ndcg_at_1000
      value: 37.972
    - type: ndcg_at_3
      value: 21.292
    - type: ndcg_at_5
      value: 18.945999999999998
    - type: precision_at_1
      value: 26.8
    - type: precision_at_10
      value: 12.06
    - type: precision_at_100
      value: 2.593
    - type: precision_at_1000
      value: 0.388
    - type: precision_at_3
      value: 19.900000000000002
    - type: precision_at_5
      value: 16.84
    - type: recall_at_1
      value: 5.443
    - type: recall_at_10
      value: 24.445
    - type: recall_at_100
      value: 52.602000000000004
    - type: recall_at_1000
      value: 78.767
    - type: recall_at_3
      value: 12.098
    - type: recall_at_5
      value: 17.077
  - task:
      type: STS
    dataset:
      type: mteb/sickr-sts
      name: MTEB SICK-R
      config: default
      split: test
      revision: a6ea5a8cab320b040a23452cc28066d9beae2cee
    metrics:
    - type: cos_sim_spearman
      value: 83.9379272617096
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_spearman
      value: 79.26752176661364
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_spearman
      value: 84.8327309083665
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_spearman
      value: 82.9394255552954
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_spearman
      value: 88.08995363382608
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_spearman
      value: 86.53522220099619
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_spearman
      value: 89.57796559847532
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (en)
      config: en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_spearman
      value: 67.66598855577894
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_spearman
      value: 88.0472708354572
  - task:
      type: Reranking
    dataset:
      type: mteb/scidocs-reranking
      name: MTEB SciDocsRR
      config: default
      split: test
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
    metrics:
    - type: map
      value: 86.04689157650684
    - type: mrr
      value: 96.51889958262507
  - task:
      type: Retrieval
    dataset:
      type: scifact
      name: MTEB SciFact
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 62.827999999999996
    - type: map_at_10
      value: 73.54899999999999
    - type: map_at_100
      value: 73.892
    - type: map_at_1000
      value: 73.901
    - type: map_at_3
      value: 70.663
    - type: map_at_5
      value: 72.449
    - type: mrr_at_1
      value: 66.0
    - type: mrr_at_10
      value: 74.554
    - type: mrr_at_100
      value: 74.81700000000001
    - type: mrr_at_1000
      value: 74.82600000000001
    - type: mrr_at_3
      value: 72.667
    - type: mrr_at_5
      value: 73.717
    - type: ndcg_at_1
      value: 66.0
    - type: ndcg_at_10
      value: 78.218
    - type: ndcg_at_100
      value: 79.706
    - type: ndcg_at_1000
      value: 79.925
    - type: ndcg_at_3
      value: 73.629
    - type: ndcg_at_5
      value: 75.89
    - type: precision_at_1
      value: 66.0
    - type: precision_at_10
      value: 10.333
    - type: precision_at_100
      value: 1.113
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 28.889
    - type: precision_at_5
      value: 19.067
    - type: recall_at_1
      value: 62.827999999999996
    - type: recall_at_10
      value: 91.533
    - type: recall_at_100
      value: 98.333
    - type: recall_at_1000
      value: 100.0
    - type: recall_at_3
      value: 79.0
    - type: recall_at_5
      value: 84.68900000000001
  - task:
      type: PairClassification
    dataset:
      type: mteb/sprintduplicatequestions-pairclassification
      name: MTEB SprintDuplicateQuestions
      config: default
      split: test
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
    metrics:
    - type: cos_sim_accuracy
      value: 99.8019801980198
    - type: cos_sim_ap
      value: 95.09301057928796
    - type: cos_sim_f1
      value: 89.71193415637859
    - type: cos_sim_precision
      value: 92.37288135593221
    - type: cos_sim_recall
      value: 87.2
    - type: dot_accuracy
      value: 99.72079207920792
    - type: dot_ap
      value: 92.77707970155015
    - type: dot_f1
      value: 85.88588588588588
    - type: dot_precision
      value: 85.97194388777555
    - type: dot_recall
      value: 85.8
    - type: euclidean_accuracy
      value: 99.7980198019802
    - type: euclidean_ap
      value: 95.04124481520121
    - type: euclidean_f1
      value: 89.61693548387096
    - type: euclidean_precision
      value: 90.34552845528455
    - type: euclidean_recall
      value: 88.9
    - type: manhattan_accuracy
      value: 99.7960396039604
    - type: manhattan_ap
      value: 95.02691504694813
    - type: manhattan_f1
      value: 89.60321446509292
    - type: manhattan_precision
      value: 90.0100908173562
    - type: manhattan_recall
      value: 89.2
    - type: max_accuracy
      value: 99.8019801980198
    - type: max_ap
      value: 95.09301057928796
    - type: max_f1
      value: 89.71193415637859
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering
      name: MTEB StackExchangeClustering
      config: default
      split: test
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
    metrics:
    - type: v_measure
      value: 72.74124969197169
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering-p2p
      name: MTEB StackExchangeClusteringP2P
      config: default
      split: test
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
    metrics:
    - type: v_measure
      value: 32.262798307863996
  - task:
      type: Reranking
    dataset:
      type: mteb/stackoverflowdupquestions-reranking
      name: MTEB StackOverflowDupQuestions
      config: default
      split: test
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
    metrics:
    - type: map
      value: 54.823414217790464
    - type: mrr
      value: 55.557133838383834
  - task:
      type: Summarization
    dataset:
      type: mteb/summeval
      name: MTEB SummEval
      config: default
      split: test
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
    metrics:
    - type: cos_sim_pearson
      value: 31.01226930465494
    - type: cos_sim_spearman
      value: 30.9368445798007
    - type: dot_pearson
      value: 30.204833368654533
    - type: dot_spearman
      value: 30.438900411966618
  - task:
      type: Retrieval
    dataset:
      type: trec-covid
      name: MTEB TRECCOVID
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 0.22699999999999998
    - type: map_at_10
      value: 2.0420000000000003
    - type: map_at_100
      value: 13.33
    - type: map_at_1000
      value: 33.627
    - type: map_at_3
      value: 0.639
    - type: map_at_5
      value: 1.056
    - type: mrr_at_1
      value: 84.0
    - type: mrr_at_10
      value: 91.167
    - type: mrr_at_100
      value: 91.167
    - type: mrr_at_1000
      value: 91.167
    - type: mrr_at_3
      value: 90.667
    - type: mrr_at_5
      value: 91.167
    - type: ndcg_at_1
      value: 82.0
    - type: ndcg_at_10
      value: 80.337
    - type: ndcg_at_100
      value: 65.852
    - type: ndcg_at_1000
      value: 59.821000000000005
    - type: ndcg_at_3
      value: 81.061
    - type: ndcg_at_5
      value: 81.396
    - type: precision_at_1
      value: 84.0
    - type: precision_at_10
      value: 85.0
    - type: precision_at_100
      value: 67.75999999999999
    - type: precision_at_1000
      value: 26.272000000000002
    - type: precision_at_3
      value: 85.333
    - type: precision_at_5
      value: 86.4
    - type: recall_at_1
      value: 0.22699999999999998
    - type: recall_at_10
      value: 2.241
    - type: recall_at_100
      value: 16.478
    - type: recall_at_1000
      value: 56.442
    - type: recall_at_3
      value: 0.672
    - type: recall_at_5
      value: 1.143
  - task:
      type: Retrieval
    dataset:
      type: webis-touche2020
      name: MTEB Touche2020
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 1.836
    - type: map_at_10
      value: 8.536000000000001
    - type: map_at_100
      value: 14.184
    - type: map_at_1000
      value: 15.885
    - type: map_at_3
      value: 3.7359999999999998
    - type: map_at_5
      value: 5.253
    - type: mrr_at_1
      value: 22.448999999999998
    - type: mrr_at_10
      value: 34.77
    - type: mrr_at_100
      value: 36.18
    - type: mrr_at_1000
      value: 36.18
    - type: mrr_at_3
      value: 30.612000000000002
    - type: mrr_at_5
      value: 32.449
    - type: ndcg_at_1
      value: 20.408
    - type: ndcg_at_10
      value: 20.498
    - type: ndcg_at_100
      value: 33.354
    - type: ndcg_at_1000
      value: 45.699
    - type: ndcg_at_3
      value: 19.292
    - type: ndcg_at_5
      value: 19.541
    - type: precision_at_1
      value: 22.448999999999998
    - type: precision_at_10
      value: 19.387999999999998
    - type: precision_at_100
      value: 7.163
    - type: precision_at_1000
      value: 1.541
    - type: precision_at_3
      value: 19.728
    - type: precision_at_5
      value: 20.0
    - type: recall_at_1
      value: 1.836
    - type: recall_at_10
      value: 15.212
    - type: recall_at_100
      value: 45.364
    - type: recall_at_1000
      value: 83.64
    - type: recall_at_3
      value: 4.651000000000001
    - type: recall_at_5
      value: 7.736
  - task:
      type: Classification
    dataset:
      type: mteb/toxic_conversations_50k
      name: MTEB ToxicConversationsClassification
      config: default
      split: test
      revision: d7c0de2777da35d6aae2200a62c6e0e5af397c4c
    metrics:
    - type: accuracy
      value: 70.5856
    - type: ap
      value: 14.297836125608864
    - type: f1
      value: 54.45458507465688
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_extraction
      name: MTEB TweetSentimentExtractionClassification
      config: default
      split: test
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
    metrics:
    - type: accuracy
      value: 61.89869835880024
    - type: f1
      value: 62.15163526419782
  - task:
      type: Clustering
    dataset:
      type: mteb/twentynewsgroups-clustering
      name: MTEB TwentyNewsgroupsClustering
      config: default
      split: test
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
    metrics:
    - type: v_measure
      value: 56.408998393035446
  - task:
      type: PairClassification
    dataset:
      type: mteb/twittersemeval2015-pairclassification
      name: MTEB TwitterSemEval2015
      config: default
      split: test
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
    metrics:
    - type: cos_sim_accuracy
      value: 88.78822197055493
    - type: cos_sim_ap
      value: 81.73234934293887
    - type: cos_sim_f1
      value: 74.16373812312898
    - type: cos_sim_precision
      value: 73.18263549961469
    - type: cos_sim_recall
      value: 75.17150395778364
    - type: dot_accuracy
      value: 87.85837754068069
    - type: dot_ap
      value: 79.69812660365871
    - type: dot_f1
      value: 72.52999744702579
    - type: dot_precision
      value: 70.25222551928783
    - type: dot_recall
      value: 74.96042216358839
    - type: euclidean_accuracy
      value: 88.74649818203493
    - type: euclidean_ap
      value: 81.47777928110055
    - type: euclidean_f1
      value: 74.1248097412481
    - type: euclidean_precision
      value: 71.37274059599413
    - type: euclidean_recall
      value: 77.0976253298153
    - type: manhattan_accuracy
      value: 88.7286165583835
    - type: manhattan_ap
      value: 81.47766386927232
    - type: manhattan_f1
      value: 74.16730231375541
    - type: manhattan_precision
      value: 71.56526005888125
    - type: manhattan_recall
      value: 76.96569920844327
    - type: max_accuracy
      value: 88.78822197055493
    - type: max_ap
      value: 81.73234934293887
    - type: max_f1
      value: 74.16730231375541
  - task:
      type: PairClassification
    dataset:
      type: mteb/twitterurlcorpus-pairclassification
      name: MTEB TwitterURLCorpus
      config: default
      split: test
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
    metrics:
    - type: cos_sim_accuracy
      value: 89.30026778437536
    - type: cos_sim_ap
      value: 86.56353001037664
    - type: cos_sim_f1
      value: 79.359197907585
    - type: cos_sim_precision
      value: 75.12379642365887
    - type: cos_sim_recall
      value: 84.10070834616569
    - type: dot_accuracy
      value: 88.8539604921023
    - type: dot_ap
      value: 85.44601003294055
    - type: dot_f1
      value: 78.20008094484713
    - type: dot_precision
      value: 74.88549080403072
    - type: dot_recall
      value: 81.82168155220204
    - type: euclidean_accuracy
      value: 89.25369658865992
    - type: euclidean_ap
      value: 86.46965679550075
    - type: euclidean_f1
      value: 79.16785612332285
    - type: euclidean_precision
      value: 73.77627028465017
    - type: euclidean_recall
      value: 85.4096088697259
    - type: manhattan_accuracy
      value: 89.26727985407692
    - type: manhattan_ap
      value: 86.46460344566123
    - type: manhattan_f1
      value: 79.1723543358
    - type: manhattan_precision
      value: 74.20875420875421
    - type: manhattan_recall
      value: 84.84755158607946
    - type: max_accuracy
      value: 89.30026778437536
    - type: max_ap
      value: 86.56353001037664
    - type: max_f1
      value: 79.359197907585
---

# LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders

> LLM2Vec is a simple recipe to convert decoder-only LLMs into text encoders. It consists of 3 simple steps: 1) enabling bidirectional attention, 2) masked next token prediction, and 3) unsupervised contrastive learning. The model can be further fine-tuned to achieve state-of-the-art performance.
- **Repository:** https://github.com/McGill-NLP/llm2vec
- **Paper:** https://arxiv.org/abs/2404.05961


## Installation
```bash
pip install llm2vec
```

## Usage
```python
from llm2vec import LLM2Vec

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
)
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
)
model = model.merge_and_unload()  # This can take several minutes on cpu

# Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
)

# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

# Encoding queries using instructions
instruction = (
    "Given a web search query, retrieve relevant passages that answer the query:"
)
queries = [
    [instruction, "how much protein should a female eat"],
    [instruction, "summit define"],
]
q_reps = l2v.encode(queries)

# Encoding documents. Instruction are not required for documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]
d_reps = l2v.encode(documents)

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
"""
tensor([[0.6470, 0.1619],
        [0.0786, 0.5844]])
"""
```

## Questions
If you have any question about the code, feel free to email Parishad (`parishad.behnamghader@mila.quebec`) and Vaibhav (`vaibhav.adlakha@mila.quebec`).