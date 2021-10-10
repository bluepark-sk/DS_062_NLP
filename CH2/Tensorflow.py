import tensorflow as tf

## tf.keras.layers.Dense

# 0. 변수 선언 및 직접 곱하고 더하기
W = tf.Variable(tf.random_uniform([5, 10], -1.0, 1.0))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(W, x) + b
# 1-1. 객체 생성 후 다시 호출하면서 입력값 설정
dense = tf.keras.layers.Dense(...)
output = dense(input)
# 1-2. 객체 생성 시 입력값 설정
output = tf.keras.layers.Dense(...)(input)
# 2. Option
__init__( # Dense층 객체를 만들 때 지정할 수 있는 인자
    units, # units: 출력 값의 크기
    activation=None, # activation: 활성화 함수
    use_bias=True, # use_bias: 편향을 사용할지 여부
    kernel_initializer='glorot_uniform', # kernel_initializer: 가중치 초기화 함수
    bias_initializer='zeros', # bias_initializer: 편향 초기화 함수
    kernel_regularizer=None, # kernel_regularizer: 가중치 정규화 방법
    bias_regularizer=None, # bias_regularizer: 편향 정규화 방법
    activity_regularizer=None, # activity_regularizer: 출력 값 정규화 방법
    kernel_constraint=None, # kernel_constraint: Optimizer에 의해 업데이트된 이후에 가중치에 적용되는 부가적인 제약 함수
    bias_constraint=None # bias_constraint: Optimizer에 의해 업데이트된 이후에 편향에 적용되는 부가적인 제약 함수
)
# 3-1. (Ex) Fully Connected Layer, Sigmoid, 10 Outputs
INPUT_SIZE = (20, 1)
inputs = tf.keras.layers.Input(shape=INPUT_SIZE)
output = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(inputs)
# 3-2. (Ex) 10 Hiddens, 2 Outputs
INPUT_SIZE = (20, 1)
inputs = tf.keras.layers.Input(shape=INPUT_SIZE)
hidden = tf.keras.layer.Dense(units=10, activation=tf.nn.sigmoid)(inputs)
output = tf.keras.layer.Dense(units=2, activation=tf.nn.sigmoid)(hidden)

## tf.keras.layers.Dropout

# 1-1. 객체 생성 후 다시 호출하면서 입력값 설정
dropout = tf.keras.layers.Dropout(...)
output = dropout(input)
# 1-2. 객체 생성 시 입력값 설정
output = tf.keras.layers.Dropout(...)(input)
# 2. Option
__init__( # 드롭아웃층 객체를 만들 때 지정할 수 있는 인자
    rate, # 드롭아웃을 적용할 확률을 지정 0~1
    noise_shape=None, # 정수형의 1D-tensor 값으로, 지정한 값만 드롭아웃을 적용할 수 있음
    seed=None # 드롭아웃의 경우 지정된 확률 값을 바탕으로 무작위로 드롭아웃을 적용하는데, 임의의 선택을 위한 시드 값
)
# 3. (Ex)
INPUT_SIZE = (20, 1)
inputs = tf.keras.layers.Input(shape=INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.2)(inputs)
hidden = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(dropout)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(hidden)

# tf.keras.layers.Conv1D

# 0. 합성곱 구분 기준: 합성곱의 방향, 합성곱 결과(출력값)
# 1-1. 객체 생성 후 다시 호출하면서 입력값 설정
conv1d = tf.keras.layers.Conv1D(...)
output = conv1d(input)
# 1-2. 객체 생성 시 입력값 설정
output = tf.keras.layers.Conv1D(...)(input)
# 2. Option
__init__( # 합성곱층 객체를 만들 때 지정할 수 있는 인자
    filters, # 필터의 개수 (출력의 차원수)
    kernel_size, # 필터의 크기 (합성곱이 적용되는 window 길이)
    strides=1, # 적용할 스트라이드 값
    padding='valid', # 패딩 방법 (valid or same)
    data_format='channels_last', # 데이터의 표현 방법 (channel_last or channel_first) -> (batch, length, channels) or (batch, channels, length)
    dilation_rate=1, # dilation 합성곱 사용 시 적용할 dilation 값
    activation=None, # 활성화 함수
    use_bias=True, # 편향을 사용할지 여부
    kernel_initializer='glorot_uniform', # 가중치 초기화 함수
    bias_initializer='zeros', # 편향 초기화 함수
    kernel_regularizer=None, # 가중치 정규화 방법
    bias_regularizer=None, # 편향 정규화 방법
    activity_regularizer=None, # 출력 값 정규화 방법
    kernel_constraint=None, # Optimizer에 의해 업데이트된 이후에 가중치에 적용되는 부가적인 제약 함수
    bias_constraint=None # Optimizer에 의해 업데이트된 이후에 편향에 적용되는 부가적인 제약 함수
)
# 3-1. (Ex) 10 filters, 3 kernel_size, same, relu
INPUT_SIZE = (1, 28, 28)
inputs = tf.keras.layers.Input(shape=INPUT_SIZE)
output = tf.keras.layers.Conv1D(
    filters=10,
    kernel_size=3,
    padding='same',
    activation=tf.nn.relu
)(inputs)
# 3-2. (Ex) 입력값에 드롭아웃 적용
INPUT_SIZE = (1, 28, 28)
inputs = tf.keras.layers.Input(shape=INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.2)(inputs)
output = tf.keras.layers.Conv1D(
    filters=10,
    kernel_size=3,
    padding='same',
    activation=tf.nn.relu
)(dropout)

## tf.