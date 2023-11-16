# AI_assignment_2023

## 과제 2

pytorch, tensorflow, keras를 이용하지 않고 numpy만으로 구현한 CNN  
MNIST 데이터셋 이용에만 tensorflow 사용  
accuracy 10% 대의 결과로, 성공하지는 못함  
gpu 병렬 연산을 이용하지 못해 시간이 오래걸려 실험을 많이 하지 못함

https://hanbit.co.kr/store/books/look.php?p_code=B7818450418  
https://drive.google.com/drive/folders/1yByDjj6r352YGqUPKHpFFNA9yqu5n4vn  
파이토치 첫걸음 책의 CNN 예제 코드 참고

## 과제 1

numpy 등 라이브러리를 이용하지 않고 파이썬 기본 내장함수 만으로 구현한 인공 신경망  
하나의 hidden layer를 가짐  
EMNIST 데이터셋 이용 학습  
시간이 오래걸려 테스트 데이터 일부(10000개)로 실험  
test 데이터로는 직접 쓴 손글씨 (t u v w x y z)  
16/20 의 accuracy로 적당히 잘 학습된 것 확인

\+ C언어로 구현 예정

https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part2_neural_network_mnist_data.ipynb  
신경망 첫걸음 책의 예제 코드 참고  
https://www.kaggle.com/datasets/crawford/emnist
