# 量子演算子

量子演算子は量子状態に作用する数学的な演算対象であり、一般に物理的な観測可能量 (エネルギーなど)や状態変換を記述するために用いられます。

Scaluqにおいて、量子演算子は{class}`Operator <scaluq.default.f64.Operator>`で表されます。
これは1つ以上の{class}`PauliOperator <scaluq.default.f64.PauliOperator>`インスタンスにより構成されます。

## PauliOperator

{class}`PauliOperator <scaluq.default.f64.PauliOperator>`は、特定の量子ビットに作用するパウリ演算子 (I,X,Y,Z)のテンソル積に、複素係数 (coef)を乗じたものを表します。

パウリ演算子は文字列、リスト、ビットマスクを使って、柔軟に定義できます。

```py
# 1. 文字列により初期化
# 量子ビットのインデックス0にX、量子ビットのインデックス2にYを適用する。
p1 = PauliOperator("X 0 Y 2", coef=1.0)

# 2. ターゲット量子ビットとパウリIDリストにより初期化
# IDs: 0=I, 1=X, 2=Y, 3=Z
p2 = PauliOperator(target_qubit_list=[0, 1, 2], pauli_id_list=[1, 0, 2], coef=1.0)

# 3. ビットマスクにより初期化
# bit_flip_mask: Xの位置, phase_flip_mask: Zの位置
p3 = PauliOperator(bit_flip_mask=0b101, phase_flip_mask=0b010, coef=1.0)
```

## Operator

{class}`Operator <scaluq.default.f64.Operator>`は複数の{class}`PauliOperator <scaluq.default.f64.PauliOperator>`の項の和で定義される一般的な演算子 (ハミルトニアンなど)を表します。

```py
from scaluq.default.f64 import Operator, PauliOperator

#prepare two pauli
pauli1 = PauliOperator("Z 0 Z 1", coef=0.5) # (0.5 + 0.0j) Z0 Z1
pauli2 = PauliOperator("Z 0 Z 1", coef=0.3) # (0.3 + 0.0j) Z0 Z1
terms = [pauli1, pauli2]

op1 = Operator(terms)
print(op1) # (0.5 + 0.0j) Z0 Z1，(0.3 + 0.0j) Z0 Z1

# 同じパウリ文字列の項を統合することで演算子を最適化できます。
op1.optimize()
print(op1) # (0.8 + 0.0j) Z0 Z1
```

## 期待値の計算と遷移振幅

定義した演算子を用いて、{class}`StateVector <scaluq.default.f64.StateVector>`から物理的な情報を取得できます。

```py
from scaluq.default.f64 import Operator, PauliOperator, StateVector

# ハール測度を用いてランダムな状態ベクトルに初期化
state = StateVector.Haar_random_state(2)

# 1. 期待値: <state|op|state>　を取得する
pauli = PauliOperator("Z 0 X 1", coef=1.0)
op = Operator([pauli])
exp_val = op.get_expectation_value(state)
print(f"Expectation value: {exp_val}") # e.g., (0.16471708718595834+0j)

# 2. 遷移振幅: <target_state|op|state>　を取得する
# 1番目の引数は "Bra" 、2番目の引数は "Ket" 。
target_state = StateVector.Haar_random_state(2) 
trans_amp = op.get_transition_amplitude(target_state, state) 
print(f"Transition amplitude: {trans_amp}") # e.g., (0.2589420813971237+0.355705558096908j)
```
