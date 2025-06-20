[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# A Trading Model Utilizing a Dynamic Weighting and Aggregate Scoring System with LSTM Networks

A regression model and trading strategy for  [FreqAI](https://www.freqtrade.io/en/stable/freqai/) module
from [freqtrade](https://github.com/freqtrade/freqtrade), a crypto trading bot.


## Overview

This project aims to develop a trading model that utilizes a dynamic weighting and aggregate scoring system to make more informed trading decisions. The model was initially built using TensorFlow and the Keras API, but has been ported to PyTorch to take advantage of its better GPU support across platforms and faster development process.

## Quick Start

0. install freqtrade
```shell
pip install -r requirements-hyperopt.txt

```

2. Clone the repository

```shell
cd /opt/raid0/ft/
git clone https://github.com/coffeecloudgit/freqailstm.git
```
2. Copy the files to the freqtrade directory

```shell 
#!/bin/bash
# 定义频率交易相关目录变量（请替换为实际路径）
FREQT_TRADE_SRC_DIR="/opt/raid0/ft/freqtrade"
FREQT_TRADE_DATA_DIR="/opt/raid0/ft/gpu001"

mkdir FREQT_TRADE_DATA_DIR

# 确保变量已定义
if [ -z "$FREQT_TRADE_SRC_DIR" ] || [ -z "$FREQT_TRADE_DATA_DIR" ]; then
    echo "错误：请先定义FREQT_TRADE_SRC_DIR和FREQT_TRADE_DATA_DIR变量"
    exit 1
fi

# 生成user_data基础目录结构（若不存在）
echo "检查并生成user_data目录结构..."
cd "$FREQT_TRADE_DATA_DIR" || { echo "错误：数据目录不存在"; exit 1; }
if [ ! -d "user_data" ]; then
    cd FREQT_TRADE_DATA_DIR
    freqtrade create-userdir --userdir user_data
    if [ $? -ne 0 ]; then
        echo "警告：无法通过freqtrade命令创建目录，将手动创建..."
        mkdir -p user_data/{strategies,freqaimodels}
    fi
else
    echo "user_data目录已存在，跳过创建"
fi

# 打印执行信息
echo "开始复制文件到Freqtrade目录..."
cd cd /opt/raid0/ft/freqailstm
# 复制Torch模型相关文件到源码目录
cp torch/BasePyTorchModel.py "$FREQT_TRADE_SRC_DIR/freqtrade/freqai/base_models/"
cp torch/PyTorchLSTMModel.py "$FREQT_TRADE_SRC_DIR/freqtrade/freqai/torch/"
cp torch/PyTorchModelTrainer.py "$FREQT_TRADE_SRC_DIR/freqtrade/freqai/torch/"

# 复制Torch模型和配置到用户数据目录
cp torch/PyTorchLSTMRegressor.py "$FREQT_TRADE_DATA_DIR/user_data/freqaimodels/"

# 复制策略相关文件到策略目录
cp V8/1HOUR/AlexStrategyFinalV8.py "$FREQT_TRADE_DATA_DIR/user_data/strategies/"
cp V8/1HOUR/AlexStrategyFinalV8Hyper.py "$FREQT_TRADE_DATA_DIR/user_data/strategies/"
cp V8/1HOUR/config-torch.json "$FREQT_TRADE_DATA_DIR/user_data/"

# 检查复制结果
if [ $? -eq 0 ]; then
    echo "文件复制完成！"
    echo "源码目录: $FREQT_TRADE_SRC_DIR"
    echo "数据目录: $FREQT_TRADE_DATA_DIR"
else
    echo "文件复制过程中出现错误，请检查路径和文件权限。"
    exit 1
fi

```
3. Download the data minimum 1 Month!
```shell
freqtrade download-data -c user_data/config-torch.json --timerange 20230101-20250611 --timeframe 15m 1h 2h 4h 1d --erase


freqtrade hyperopt -s AlexStrategyFinalV11 --freqaimodel PyTorchLSTMRegressor -c user_data/config-torch.json --timerange 20250301-20250620 --hyperopt-loss SharpeHyperOptLoss --spaces "buy sell" 2>&1 | tee user_data/hyperopt001.txt


freqtrade backtesting -c user_data/config-torch.json --breakdown day week month --timerange 20250301-20250620 2>&1 | tee user_data/backtest_res001.txt

freqtrade trade --strategy AlexStrategyFinalV11 --config user_data/config-torch.json --freqaimodel PyTorchLSTMRegressor 2>&1 | tee user_data/trade001.log
```

4. Edit "freqtrade/configuration/config_validation.py"
```python
...
def _validate_freqai_include_timeframes()
...
    if freqai_enabled:
        main_tf = conf.get('timeframe', '5m') -> change to '15m/30m/1h' or the **min** timeframe of your choosing
```
5. Make sure your package is edible after the the changes
```shell
pip install -e .
```

7. Run the backtest chosse a timeframe minimum 2-3months
```shell
freqtrade backtesting -c user_data/config-torch.json --breakdown day week month --timerange 20240601-20240801 
````

## Quick Start with docker

1. Clone the repository

```shell
git clone [https://github.com/AlexCryptoKing/freqailstm.git]
```
2. Build local docker images

```shell
cd freqailstm
docker build -f torch/Dockerfile  -t freqai .
```
3. Download data and Run the backtest
```
docker run -v ./data:/freqtrade/user_data/data  -it freqai  download-data -c user_data/config-torch.json --timerange 2020601-2024081 --timeframe 15m 30m 1h 2h 4h 8h 1d --erase

docker run -v ./data:/freqtrade/user_data/data  -it freqai  backtesting -c user_data/config-torch.json --breakdown day week month --timerange 20240701-20240801 
```

## Model Architecture
config.json 

The core of the model is a Long Short-Term Memory (LSTM) network, which is a type of recurrent neural network that excels at handling sequential data and capturing long-term dependencies.

The LSTM model (PyTorchLSTMModel) has the following architecture:

1. The input data is passed through a series of LSTM layers (the number of layers is configurable via the `num_lstm_layers` parameter.). Each LSTM layer is followed by a Batch Normalization layer and a Dropout layer for regularization.
2. The output from the last LSTM layer is then passed through a fully connected layer with ReLU activation.
3. An Alpha Dropout layer is applied for additional regularization.
4. Finally, the output is passed through another fully connected layer to produce the final predictions.

The model's hyperparameters, such as the number of LSTM layers, hidden dimensions, dropout rates, and others, can be easily configured through the `model_kwargs` parameter in the `model_training_parameters` section of the configuration file.

Here's an example of how the model_training_parameters can be set up:

```json
"model_training_parameters": {
    "learning_rate": 3e-3,
    "trainer_kwargs": {
    "n_steps": null,
    "batch_size": 32,
    "n_epochs": 10,
    },
    "model_kwargs": {
    "num_lstm_layers": 3,
    "hidden_dim": 128,
    "dropout_percent": 0.4,
    "window_size": 5
    }
}
```
Let's go through each of these parameters:

- `learning_rate`: This is the learning rate used by the optimizer during training. It controls the step size at which the model's weights are updated in response to the estimated error each time the model weights are updated.
- `trainer_kwargs`: These are keyword arguments passed to the `PyTorchLSTMTrainer` which is located in PyTorchModelTrainer.
    - `n_steps`: The number of training iterations. If set to null, the number of epochs (n_epochs) will be used instead.
    - `batch_size`: The number of samples per gradient update.
    -  `n_epochs`: The number of times to iterate over the dataset.
- `model_kwargs`: These are keyword arguments passed to the `PyTorchLSTMModel`.
    - `num_lstm_layers`: The number of LSTM layers in the model.
    - `hidden_dim`: The dimensionality of the output space (i.e., the number of hidden units) in each LSTM layer.
    - `dropout_percent`: The dropout rate for regularization. Dropout is a technique used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
    - `window_size`: The number of time steps (or data points in the above case) to look back when making predictions.


## The Strategy

At its core, this strategy is all about making smart trading decisions by looking at the market from different angles. It's like having a team of experts, each focusing on a specific aspect of the market, and then combining their insights to make a well-informed decision.

Here's how it works:

1. **Indicators**: The strategy calculates a bunch of technical indicators, which are like different lenses to view the market. These indicators help identify trends, momentum, volatility, and other important market characteristics.

2. **Normalization**: To make sure all the indicators are on the same page. it normalizes them by calculating the z-score. This step ensures that the indicators are comparable and can be weighted appropriately.

3. **Dynamic Weighting**: The strategy is adaptable and can adjust the importance of different indicators based on market conditions.

4. **Aggregate Score**: All the normalized indicators are combined into a single score, which represents the overall market sentiment. Just like taking a vote among the experts to reach a consensus.

5. **Market Regime Filters**: The strategy considers the current market regime, whether it's bullish, bearish, or neutral. Looking up the weather before deciding on an outfit. 🌞🌧️?

6. **Volatility Adjustments**: It takes into account the market's volatility and adjusts the target score accordingly. We want to be cautious when the market is choppy and more aggressive when it's calm.

7. **Final Target Score**: All these factors are combined into a final target score, which is like a concise and informative signal for the LSTM model to learn from. It's like giving the model a clear and focused task to work on.

8. **Entry and Exit Signals**: we use the predicted target score and set thresholds to determine when to enter or exit a trade.

## Contributing
Contacts: 

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/alex15_08)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/vfJQ5pftwX)

Contributions to the project are welcome! If you find any issues or have suggestions for improvements, please open an
issue or submit a pull request on the [GitHub repository](https://github.com/AlexCryptoKing/freqailstm.git).


```python
Best result:

    35/100:   9827 trades. 6399/12/3416 Wins/Draws/Losses. Avg profit   0.67%. Median profit   1.50%. Total profit 4296.91988925 USDT ( 429.69%). Avg duration 1:46:00 min. Objective: -131
.40184


    # Buy hyperspace params:
    buy_params = {
        "confidence_threshold_multiplier": 0.51115,
        "dynamic_long_threshold_multiplier": 0.94728,
        "dynamic_short_threshold_multiplier": 0.9477,
        "rolling_trend_threshold_multiplier": 1.01779,
        "stake_scaling_factor": 1.5,
    }

    # Sell hyperspace params:
    sell_params = {
        "atr_multiplier": 1.81397,
        "base_risk": 0.04342,
        "dynamic_exit_threshold_multiplier": 1.03184,
        "exit_trend_threshold_multiplier": 0.38391,
        "max_risk_per_trade_multiplier": 0.01, 
        "max_trade_duration_long": 6,
        "max_trade_duration_short": 2,
        "timed_exit_long_threshold": 26,
        "timed_exit_short_threshold": 15,
    }

    # Stoploss:
    stoploss = -1.0  # value loaded from strategy

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.001  # value loaded from strategy
    trailing_stop_positive_offset = 0.0139  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy
     

    # Max Open Trades:
    max_open_trades = 20  # value loaded from strategy
```
     


