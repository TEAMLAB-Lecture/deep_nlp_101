# Natural Language Process with Deep Learning

## Course overview
Todays, Natural Language Processing (NLP) plays a significant role in building intelligent information systems. Traditionally, applications of NLP are everywhere in a variety of areas including web searching, email processing, e-commerce, translation, and automatic generation of reports.  Human languages are complex and unstructured. NLP is recognized as a tough field because various preprocessing is required for a computer to understand human words.

In recent years, the explosion of text data and advancement of Deep Learning technology have resulted in a dramatic increase in the performance of existing NLP applications. In particular, neural networks, unlike traditional models, find the appropriate features for text information on their own, minimizing human involvement. Besides, by developing appropriate models for the features, we have seen dramatically increasing performance and practicality.

In this course, students will take the advanced learning to develop NLP applications with cutting-edge deep learning techniques. You will study the design, implementation, debugging, and visualization techniques of neural network models to handle textual information. Through the final project, students will also have the opportunity to organize and train their neural net for text processing problems in specific areas they want to handle. It will be an arduous journey, but I wish you an enjoyable walk with your friends.

## Course history
- [2018] [NLP with DL](https://github.com/TEAMLAB-Lecture/deep_nlp_101/tree/master/2018), graduate cource, IME at Gachon University

## Course coverage
- 본 과정에서 주료 다루는 NLP 기법들은 아래와 같습니다.
  - Language modeling
    - Word embeddings
    - Document embeddings
    - Text simliarity
  - Memory and Attention Models
  - Text classification
  - Sentiment Analysis
  - Parsing & Tagging
  - Neural machine translation
  - Conversation modeling / Dialog
    - Chatbot modeling
    - Visual question and answering


## Prerequisites
- 파이썬 코딩 능력
  - 데이터 과학을 위한 파이썬 입문(https://www.inflearn.com/course/python-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9E%85%EB%AC%B8-%EA%B0%95%EC%A2%8C/)
- 머신 러닝 기초 이해
  - 밑바닥 부터 시작하는 머신러닝 입문 (https://www.inflearn.com/course/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%9E%85%EB%AC%B8-%EA%B0%95%EC%A2%8C/)
- 딥러닝 기초 이해
  - 모두를 위한 머신러닝과 딥러닝[http://hunkim.github.io/ml/]

## Resources
- Courses
  - CS224n [link](http://web.stanford.edu/class/cs224n/)
  - Neural Networks for NLP from Carnegie Mellon University [link](http://phontron.com/class/nn4nlp2017/)
  - Deep Learning for Natural Language Processing from University of Oxford and DeepMind [link](https://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/)
- [paperswithcode](https://paperswithcode.com/)
- [Awesome-dl4nlp](https://github.com/brianspiering/awesome-dl4nlp)

## Papers
### Vector for language modeling

#### word embeddings
- \[[NLM_2003]()\] Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. Journal of machine learning research, 3(Feb), 1137-1155. Available - http://www.jmlr.org/papers/v3/bengio03a.html
- \[[WORD2VEC_2013]()\] Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. "Distributed representations of words and phrases and their compositionality." In Advances in neural information processing systems, pp. 3111-3119. 2013.  Available - https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
- \[[GLOVE_2014]()\] Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532-1543. 2014. Available - http://www.aclweb.org/anthology/D14-1162

#### sentece or paragraph embeddings
- \[[DOC2VEC_2014]()\] Le, Quoc, and Tomas Mikolov. "Distributed representations of sentences and documents." In International Conference on Machine Learning, pp. 1188-1196. 2014. Available - https://cs.stanford.edu/~quocle/paragraph_vector.pdf
- \[[SIM_SEN_2016]()\] Arora, S., Liang, Y. and Ma, T., 2016. A simple but tough-to-beat baseline for sentence embeddings. Available - https://openreview.net/forum?id=SyK00v5xx
- \[[DEEP_SEN_2016]()\] Palangi, H., Deng, L., Shen, Y., Gao, J., He, X., Chen, J., Song, X. and Ward, R., 2016. Deep sentence embedding using long short-term memory networks: Analysis and application to information retrieval. IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), 24(4), pp.694-707. Available - https://dl.acm.org/citation.cfm?id=2992457



### Text Classification
- \[[CHAR_CNN_2015]()\] Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level convolutional networks for text classification." In Advances in neural information processing systems, pp. 649-657. 2015. Aaailalble at: http://papers.nips.cc/paper/5782-character-level-convolutional-networks-fo
- \[[FASTTEXT_2016()]()\]Joulin, Armand, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. "Bag of tricks for efficient text classification." arXiv preprint arXiv:1607.01759 (2016). Available at - https://arxiv.org/abs/1607.01759

### Network Architecture
- \[SEQ2SEQ_2014\]Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." In Advances in neural information processing systems, pp. 3104-3112. 2014. Available at - https://arxiv.org/abs/1409.3215


http://www.sciencedirect.com/science/article/pii/S0893608005001206
