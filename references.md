http://www.fastcampus.co.kr/data_camp_dtm/

Part 1. 텍스트 마이닝 기법
1	텍스트 마이닝 개요 : 텍스트 분석을 위한 기본적인 내용을 소개
- 텍스트 마이닝 프레임워크
- Logistic Regression을 이용한 문서 판별
2	비지도학습 기반의 한국어 자연어 처리 모듈 개발
: 사전(dictionary) 또는 말뭉치(corpus)를 이용하지 않는 데이터 기반 한국어 처리 방법 소개
- 단어 추출
- 토크나이저
- 명사 추출
3	텍스트 데이터에 분류 모델 적용: 분류기(classifier)를 텍스트 데이터에 적용하는 방법 소개
- 분류기를 이용한 키워드 추출
- 오탈자 교정
- 띄어쓰기 오류 수정
4	단어 / 문서의 임베딩 및 시각화
: classification/ clustering에 적합하도록 단어/ 문서의 벡터를 변환하는 임베딩 방법 소개
- 단어/ 문서의 임베딩: Word2Vec, Doc2Vec, CloVe, FastText
- 시각화를 위한 임베딩: t-SNE, LLE, ISOMAP, MDS, PCA
5	문서 군집화 및 벡터 인덱서
: 비슷한 문서를 하나의 군집으로 추출하는 군집화 기술 및 대량 텍스트 처리를 위한 벡터 인덱서 소개
- 군집화 모델 개요 및 텍스트 데이터에의 적용
- 벡터 인덱서: LSH

Part 2. Neural Network 기법과 응용
6	Feed-Forward Network & 인공신경망 학습 방법
: Feed-Forward Network에 대해 이해하고 이를 기반으로 한 인공신경망 모델의 학습 방법 소개
- Feed-Forward Network
- Backpropagation, Gradient Descent
- TensorFlow basic
7	비지도학습 기반의 인공신경망 & CNN 1
: 데이터의 새로운 표현을 생성하는 비지도학습 인공신경망의 텍스트 데이터 적용 방법 및 CNN 개요 소개
- 딥러닝의 학습 기술들 : Dropout, Batch Normalization, various optimization methods, Weight Initialization
- Autoencoder and its variations
- CNN(Convolutional Neural Network)의 구조와 학습 방법
8	CNN 2 : CNN을 이용한 텍스트 분류 모델 및 관련 최신 연구 동향 소개
- Deep Learning for Textual Data
- CNN을 이용한 문서/ 문장 분류
- Character-Level CNN
- CNN 관련 최신 연구 동향 소개
9	RNN 1 : RNN의 이론, 기계번역에서 좋은 성능을 보인 seq2seq 모델 및 관련 최신 연구 동향 소개
- RNN(Recurrent Neural Network)의 구조와 학습 방법
- seq2seq(Sequence-to-Sequence) model
- RNN 관련 최신 연구 동향 소개
10	RNN 2 & 챗봇 구성 요소 개발 : RNN 기반의 언어 모델 및 강의 내용을 이용한 챗봇 구성 요소 개발
- 문장 생성을 위한 RNN based language model
- 챗봇의 개요
- 챗봇 구성 요소 개발


http://www.fastcampus.co.kr/data_camp_textmining/
파이썬을 이용한 텍스트 데이터의 구조화
4	형태소 분석과 후처리
- 형태소와 형태소 분석의 개념
- 형태소 분석 라이브러리의 설치와 사용
- 형태소 분석 결과의 구조화와 저장
- 형태소 분석 결과의 후처리
5	문서 군집
- 문서 군집의 개념
- 문서의 속성과 유사도 측정
- 계층적 군집 분석
- 비계층적 군집 분석
6	문서 분류
- 문서 분류의 개념
- 나이브 베이즈 모델을 이용한 문서 분류
- 문서 분류의 성능 평가
- 그리드 탐색에 의한 문서 분류 파라미터 최적화

파이썬을 이용한 텍스트 마이닝 응용
7	키워드 분석
- 형태소 빈도 계수
- 형태소 빈도의 시각화
- 용어 빈도와 문헌 빈도
- 분류 사전의 활용
8	어휘 공기 분석
- 어휘 공기의 개념
- 어휘 공기의 추출과 계수
- 어휘 공기 행렬의 구성
- 어휘 공기 네트워크의 생성
9	토픽 모델링과 워드 임베딩
- 토픽 모델링의 개념
- 토픽 모델링을 이용한 주제 분석
- 워드 임베딩의 개념
- 워드 임베딩을 이용한 유사도 분석
10	감성 분석
- 감성 분석의 개념과 방법
- 문서 분류 기법을 이용한 긍/부정 분석
- 감성어 사전 기반 세부 감성 분석
- 개인별 텍스트 마이닝 프로젝트 경험 나누기
