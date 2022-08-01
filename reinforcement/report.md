# 強化学習による迷路の探索

## 問題設定

迷路の自動探索を行うエージェントを $\epsilon$-Greedy TD(0) アルゴリズムにより訓練する．

- 壁方向には移動できないものとする．
- 現在の位置から移動できるマスを $(x', y')$ としたとき，瞬時報酬 $r(x', y')$ は，$(x', y')$ がゴールならば 1, そうでないなら 0 とする．

迷路は以下のものを使用する．赤色のマスがスタート，青色のマスがゴールに対応している．
<img src="field.png" width=300 style="padding: 10px">

## 手法

1. パラメータ $\epsilon, \gamma, \alpha$ を初期化
2. 状態価値関数を0に初期化
3. 以下の学習エピソードを繰り返す．「エピソード数が50以上500未満」かつ「状態価値関数の更新幅の絶対値の和が適当に定めた閾値よりも小さい」を停止条件として設定する．
   1. エージェントをスタート位置に置く
   2. $\epsilon$-Greedy$ 方策に基づき行動を選択
        - 確率 $1-\epsilon$ で $\argmax_{x', y'}(r(x', y') + \gamma V(x', y'))$ となる行動を選択し，確率 $\epsilon$ でランダムに行動を選択する．

4. 状態価値関数を以下の式に基づき更新する．
        - $V(x, y) := V(x, y) + \alpha(r + \gamma V(x', y') - V(x, y))$
5. $(x', y')$ がゴール出なければ，$(x, y) \leftarrow (x', y')$ として行動選択のステップまで戻る．

## 実装の詳細

- Agentクラス
  - 迷路の探索を行うAgentのクラス．
  - 自身の現在地と，今までの移動ログを保持する．

- TDクラス
  - Agentクラスのインスタンスと，ndarrayの迷路をフィールドとして保持し，探索アルゴリズムを実行・管理するクラス．
  - 迷路は txt 形式として読み込む．壁は#, ゴールはG, スタートはS, それ以外は半角スペースとして対応付けている．

## 実験１： 学習率を変える

$\gamma=0.95,\,\, \alpha=0.2$ として初期化し，$\epsilon$ の値を変えながら実際に探索を行っていく．なお掲載している各図は学習曲線(上)，最終エピソード時にエージェントが進んだ経路(左下)，各マスの状態価値関数(赤色が濃いほど価値が高い)(右下)を表す．

- $\epsilon=0.1$
  <img src="TD_learning_01.png" width=600 style="padding: 0px">
  <img src="TD_log_01.png" width=300 style="padding: 0px"> <img src="TD_values_01.png" width=300 style="padding: 0px">
</br>

- $\epsilon=0.3$
  <img src="TD_learning_03.png" width=600 style="padding: 0px">
  <img src="TD_log_03.png" width=300 style="padding: 0px"> <img src="TD_values_03.png" width=300 style="padding: 0px">
</br>

- $\epsilon=0.5$
  <img src="TD_learning_05.png" width=600 style="padding: 0px">
  <img src="TD_log_05.png" width=300 style="padding: 0px"> <img src="TD_values_05.png" width=300 style="padding: 0px">

### 考察

$\epsilon$ が大きくするほどランダムに探索することができるため学習曲線が早くに収束する一方で，学習が収束していても最適なルートを通らず無駄な寄り道をしてしまう可能性が高くなる．そのため，学習が進むにつれて $\epsilon$ を徐々に小さくしていくことが最適だと考えられる．これは学習の初期段階で大雑把に良いルートが掴み，徐々に最適なルートを通るように制限をかけていく（＝行動選択のランダム性を排除する）ことに対応している．

## 実験２： Q学習との比較

実験１と同様 に$\gamma=0.95,\,\, \alpha=0.2$ として初期化し，$\epsilon$ の値を変えながらQ学習により探索を行っていく．なお掲載している各図は学習曲線(上)，最終エピソード時にエージェントが進んだ経路(左下)，各マスの行動価値関数(緑色が濃いほど価値が高い)(右下)を表す．右下の図は各マスを3x3に分割し，左右上下の位置がそのマスにおける各行動の行動価値に対応している．

- $\epsilon=0.1$
  <img src="Q_learning_01.png" width=600 style="padding: 0px">
  <img src="Q_log_01.png" width=300 style="padding: 0px"> <img src="Q_values_01.png" width=300 style="padding: 0px">
</br>

- $\epsilon=0.3$
  <img src="Q_learning_03.png" width=600 style="padding: 0px">
  <img src="Q_log_03.png" width=300 style="padding: 0px"> <img src="Q_values_03.png" width=300 style="padding: 0px">
</br>

- $\epsilon=0.5$
  <img src="Q_learning_05.png" width=600 style="padding: 0px">
  <img src="Q_log_05.png" width=300 style="padding: 0px"> <img src="Q_values_05.png" width=300 style="padding: 0px">

### 考察

TD学習ほど $\epsilon$ の値を変えても収束スピードには影響しなかった．また，どのような $\epsilon$ の値においても学習の初期段階におけるゴールに辿り着くまでのステップ数がTD学習に比べて少なかった．

## 実験３： 複雑な迷路における探索

- $\epsilon=0.1$
  - TD学習
  <img src="TD_learning_01_2.png" width=600 style="padding: 0px">
  <img src="TD_log_01_2.png" width=500 style="padding: 0px"> <img src="TD_values_01_2.png" width=500 style="padding: 0px">
  - Q学習
  <img src="Q_learning_01_2.png" width=600 style="padding: 0px">
  <img src="Q_log_01_2.png" width=500 style="padding: 0px"> <img src="Q_values_01_2.png" width=500 style="padding: 0px">
</br>

- $\epsilon=0.3$
  - TD学習
  <img src="TD_learning_03_2.png" width=600 style="padding: 0px">
  <img src="TD_log_03_2.png" width=500 style="padding: 0px"> <img src="TD_values_03_2.png" width=500 style="padding: 0px">
  - Q学習
  <img src="Q_learning_03_2.png" width=600 style="padding: 0px">
  <img src="Q_log_03_2.png" width=500 style="padding: 0px"> <img src="Q_values_03_2.png" width=500 style="padding: 0px">
</br>

- $\epsilon=0.5$
  - TD学習
  <img src="TD_learning_05_2.png" width=600 style="padding: 0px">
  <img src="TD_log_05_2.png" width=500 style="padding: 0px"> <img src="TD_values_05_2.png" width=500 style="padding: 0px">
  - Q学習
  <img src="Q_learning_05_2.png" width=600 style="padding: 0px">
  <img src="Q_log_05_2.png" width=500 style="padding: 0px"> <img src="Q_values_05_2.png" width=500 style="padding: 0px">
</br>

### 考察

迷路が単純な場合と同様に，Q学習のほうがTD学習よりも学習の初期においてゴールにたどり着くまでのステップ数が少なかった．
