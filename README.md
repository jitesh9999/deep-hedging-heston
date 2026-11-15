# Deep Hedging under Heston with Transaction Costs (NumPy)

From-scratch implementation of Buehler et al. (_Deep Hedging_, 2019). A multi-layer perceptron learns to hedge a short at-the-money call option under Heston stochastic volatility and proportional transaction costs, optimized directly on a tail-risk objective ($\text{CVaR}_{95}$).

## Implementation Details

- **Pure NumPy:** MLP, backpropagation, Adam optimizer, CVaR objective, and analytical gradients implemented by hand. Verified via finite-difference checks (`tests/test_gradients.py`).
- **Market Dynamics:** Full-truncation Euler Heston simulation ($v_0 = \theta = 0.04, \xi = 0.5, \rho = -0.7$), daily rebalancing, 10 bp proportional transaction costs, and terminal liquidation.
- **Observable Inputs Only:** Features restricted to log-moneyness, time-to-maturity, and a historical EWMA volatility estimator (latent variance is hidden from the policy).
- **Baselines:** Compared against Black-Scholes delta hedging under both flat 20% volatility and time-varying EWMA volatility.

## Out-of-Sample Performance (50,000 Paths, ATM Call Premium = 0.0399)

### Training Regime

| Strategy                | Mean P&L | Std    | CVaR95 loss | CVaR99 loss | Avg cost | Turnover |
| ----------------------- | -------- | ------ | ----------- | ----------- | -------- | -------- |
| Unhedged                | 0.0020   | 0.0488 | 0.1272      | 0.1744      | 0.0000   | 0.00     |
| BS delta (flat 20% vol) | -0.0015  | 0.0105 | 0.0278      | 0.0410      | 0.0034   | 3.41     |
| BS delta (EWMA vol)     | -0.0016  | 0.0104 | 0.0258      | 0.0348      | 0.0035   | 3.45     |
| **Deep hedge (MLP)**    | -0.0008  | 0.0121 | **0.0217**  | **0.0307**  | 0.0028   | 2.73     |

_Note: $\text{CVaR}_{95}$ is 16% lower than EWMA-vol Black-Scholes (paired bootstrap 95% CI for difference: 0.0039 to 0.0043) with ~20% lower turnover._

### Stress Test (Unseen Parameters: $v_0 = \theta = 0.06, \xi = 0.8, \rho = -0.5$)

| Strategy                | CVaR95 loss | CVaR99 loss |
| ----------------------- | ----------- | ----------- |
| BS delta (flat 20% vol) | 0.0605      | 0.0884      |
| BS delta (EWMA vol)     | 0.0497      | 0.0655      |
| Deep hedge (MLP)        | 0.0521      | 0.0792      |

_Limitation:_ Policy is tuned to the training regime; under severe stress shifts, it underperforms EWMA-vol Black-Scholes without domain randomization.

## Usage

```bash
pip install -r requirements.txt
python tests/test_gradients.py                                   # Gradient verification
python -m deephedge.train --steps 1200 --batch 2048 --out results   # Training
python -m deephedge.evaluate --out results                       # Evaluation & metrics
```
