# Results

| Train → test | Model | Threshold | Precision | Recall | F1 | Avg. precision | Words caught | False triggers / min |
|---|---|---|---|---|---|---|---|---|
| Train2 → Train1 | always 'no' | 1.01 | 0.00 | 0.00 | 0.00 | 0.06 | 0/6 | 0.00 |
| Train2 → Train1 | logistic regression (mel statistics) | 0.42 | 0.08 | 0.56 | 0.15 | 0.09 | 6/6 | 16.01 |
| Train2 → Train1 | CNN (log-mel + augmentation) | 0.31 | 0.08 | 0.62 | 0.14 | 0.06 | 6/6 | 17.27 |
| Train1 → Train2 | always 'no' | 1.01 | 0.00 | 0.00 | 0.00 | 0.27 | 0/3 | 0.00 |
| Train1 → Train2 | logistic regression (mel statistics) | 0.12 | 0.30 | 0.65 | 0.41 | 0.32 | 3/3 | 19.49 |
| Train1 → Train2 | CNN (log-mel + augmentation) | 0.05 | 0.20 | 0.70 | 0.31 | 0.19 | 3/3 | 25.05 |
