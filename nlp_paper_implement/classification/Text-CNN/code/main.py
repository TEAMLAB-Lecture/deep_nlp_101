from model import KoreanTextCnn

def main():
    textcnn = KoreanTextCnn('nonstatic')
    #textcnn = KoreanTextCnn('rand')
    #textcnn = KoreanTextCnn('static')
    #textcnn = KoreanTextCnn('multi')

    textcnn.dictionary_for_static()
    textcnn.static_word2vec_dataset()
    textcnn.datas_to_docs()
    textcnn.static_make_word2vec()
    textcnn.static_docs_to_index()
    textcnn.static_cnn_method()

if __name__ == '__main__':
    main()