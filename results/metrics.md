
### Base Heston (training dynamics)

| Strategy | Mean P&L | Std | CVaR95 loss | CVaR99 loss | Avg cost | Turnover |
|---|---|---|---|---|---|---|
| Unhedged | 0.0020 | 0.0488 | 0.1272 | 0.1744 | 0.0000 | 0.00 |
| BS delta (flat 20% vol) | -0.0015 | 0.0105 | 0.0278 | 0.0410 | 0.0034 | 3.41 |
| BS delta (EWMA vol) | -0.0016 | 0.0104 | 0.0258 | 0.0348 | 0.0035 | 3.45 |
| Deep hedge (MLP) | -0.0008 | 0.0121 | 0.0217 | 0.0307 | 0.0028 | 2.73 |

### Stressed Heston (unseen: higher vol, vol-of-vol)

| Strategy | Mean P&L | Std | CVaR95 loss | CVaR99 loss | Avg cost | Turnover |
|---|---|---|---|---|---|---|
| Unhedged | -0.0052 | 0.0636 | 0.1899 | 0.2828 | 0.0000 | 0.00 |
| BS delta (flat 20% vol) | -0.0090 | 0.0185 | 0.0605 | 0.0884 | 0.0035 | 3.50 |
| BS delta (EWMA vol) | -0.0087 | 0.0172 | 0.0497 | 0.0655 | 0.0034 | 3.33 |
| Deep hedge (MLP) | -0.0084 | 0.0181 | 0.0521 | 0.0792 | 0.0029 | 2.90 |
