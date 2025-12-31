# Intraday Bounce Recovery Predictor

Machine learning system to predict whether intraday stock bounces will recover to opening price by end of day.

## 🎯 Performance

- **Recall:** 56% (catches over half of recoveries)
- **Precision:** 37% (1 in 3 signals are correct)
- **Training Data:** 161 bounce recoveries across 5 years
- **Model:** Random Forest with balanced class weights

## 🔧 Features

- Vectorized bounce detection (processes 40k bounces in <60 seconds)
- Technical indicators (RSI, volume analysis)
- Market context features (SPY comparison)
- Hyperparameter optimization with GridSearchCV

## 📊 Key Insights

- **First bounces** perform best (0 prior bounces avg)
- **Early timing** critical (67 min avg for recoveries)
- **Volume ratio** strong predictor (2.0x avg for recoveries)

## 🚀 Tech Stack

- Python 3.12
- scikit-learn (Random Forest)
- pandas (vectorized processing)
- Professional-grade data (TradeStation 5-year history)

## 📈 Future Work

- Expand to 15 high-volatility stocks (1,690 Label 1 target)
- Live trading system with real-time monitoring
- Ensemble with neural network
