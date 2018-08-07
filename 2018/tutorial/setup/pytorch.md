# Pytorch on Local machine

### 1. Python miniconda 설치
- [영상](https://www.youtube.com/watch?v=OMuHLDvmQl4&index=5&list=PLBHVuYlKEkUJvRVv9_je9j3BpHwGHSZHz)


### 2. 가상환경 구축 설치
- conda 환경 세팅
```bash
conda create -n torch python=3.6
```

- 환경 실행
```bash
source activate torch
```

- pytorch 설치
```bash
conda install pytorch torchvision -c pytorch
```


### 3. pytorch 설치 확인
```python
>>> import torch
>>> print(torch.rand(3,3).cuda())
tensor([[0.7395, 0.2452, 0.6946],
        [0.5746, 0.8971, 0.9563],
        [0.6575, 0.7767, 0.4529]], device='cuda:0')
>>> torh.cuda.is_availabe()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'torh' is not defined
>>> torch.cuda.is_availabe()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'torch.cuda' has no attribute 'is_availabe'
>>> torch.cuda.is_available()
True

```
