## Backronym generator
- 논문이나 문서를 입력해주면 그럴듯한 acronym으로 상품명 정해주기 ㅋ
- 무슨무슨 단어는 꼭 포함되도록하고 그럴듯한 acronym 생성하는것도 ㅋㅋ
- 데이터셋 구축을 주된 컨트리뷰션으로 하고, 적당한 베이스라인 만들어서 내면 어떨까 싶은 ㅋ
- 사람들도 보통 처음본 약어가 있으면 자기가 아는 언어로 뜻을 풀어보려고 하자나요

## Author2Vec for name disambiguation
특허나 웹, 논문 같은데서 Author의 Name disambiguation이 이슈중 하나인데.
1) 이름 같은 사람을 모두 다른 개체로 본다
2) 그 이름과 같이 나온 단어, 다른 저자, 문서 정보등을 같이 묶어서 Embedding Representation을 찾는다
3) 결과로 나온 개체별 embedding representation을 Clustering하여 유사한 값은 같은걸로 묶어준다
이렇게 하면 Embedding을 사용하여 Name ambiguation문제를 해결할 수 있지 않을까요?
