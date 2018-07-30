# Pytorch Enviroment Setup

이번 챕터에서는 이번 강의에서 사용하는 Pytorch를 실행하기 위한 다양한 환경들을 세팅하는 방법에 대해 설명합니다. 기본적으로 딥 러닝 환경을 구성하기 위해서는 두 가지 환경위에서 실행이 가능합니다.

1. 개인 로컬 PC
2. 클라우드 서비스

일반적으로 처음 시작하는 사람은 개인의 로컬 PC 환경에서 딥러닝 환경을 구축할 때가 많습니다. 그러나 개인의 로컬 PC에는 고가의 GPU를 확보하지 못하거나, 메모리의 부족 문제 등이 일어날 수 있습니다. 이럴 경우는 우리는 클라우드 서비스를 사용합니다.

대표적인 클라우드 서비스는 아마존 AWS, 구글 클라우드, MS의 애저 등이 있습니다. 딥러닝이나 머신러닝에만 특화된 서비스로 구글의 Colab이 있습니다. 국내에도 대표적인 분산처리 머신러닝 서비스 스타트업 레블업의 [backend.AI](https://cloud.backend.ai/#/)와 네이버의 NSML 등이 있습니다.

본 강의에서는 가장 대중적으로 많이 쓰이는 AWS와 구글 클라우드의 클라우드 머신러닝 서비스와 무료 클라우드 플랫폼인 Colab 그리고 국내 스타트업 서비스인 [backend.AI](https://cloud.backend.ai/#/)에서 간단히 클라우드 서비스를 사용하는 방법에 대해서 설명드립니다.

각 서비스의 장단점은 아래표로 정리되어 있습니다.

|서비스명   | 개요 | 장점   | 단점  | 가격  |
|-----|-----|-----|-----|-----|
|[AWS AMI](https://aws.amazon.com/ko/machine-learning/amis/)   |- 기존 AWS AMI에 딥러닝 프레임워크를 모두 쓸 수 있도록  세팅 해둔 AMI </br> - 가장 대중적인 딥러닝 서비스 모델</br>- 가장 대중적인 [p2 인스턴스](https://aws.amazon.com/ko/ec2/instance-types/p2/)는 192GB의 GPU 메모리 제공</br>-    |- 사용자가 원할 경우 spot instance 등으로 가격을 낮출 수 있음</br>- 클라우드 기반 분산 학습의 사용이 용이해짐</br>- 관련 문서를 찾는데 용이함   |- 개인 PC GPU 대비 높은 가격</br>TensorFlow의 TPU 사용불가  |- $0.023 to $41.944/hr</br>-권장 p2.xlarge $0.900/hr |   
|[AWS SageMaker]()   |    |  |   | |   
|[Google VM 인스턴스]()   |    |- TensorFlow의 [TPU](https://cloud.google.com/tpu/) 사용가능  |   | |   
|[Google Cloud ML 엔진](https://cloud.google.com/ml-engine/docs/tensorflow/technical-overview)   |    |- TensorFlow의 [TPU](https://cloud.google.com/tpu/) 사용가능  |   | |  
|[Google Colab](https://colab.research.google.com/)  |    |  |   | |   
| [backend.AI](https://cloud.backend.ai/#/)|   |   |   |   |   |
