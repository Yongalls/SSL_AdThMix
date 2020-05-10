## Adaptive Threshold
Our final adaptive threshold algorithm is implemented here. 
We used current train accuracy as a percentile for unlabeled data. 
Also, we set 0.5 as minimum value for pseudo label's confidence. 
Also, loss term for unlabeled data has been changed to cross entropy.

### Excution

```
nsml run -d fashion_eval -e main.py -a "--unlabeled_loss CEE --min_threshold 0.5"
```

If you want to use MSE for computing unlabeled loss, change the argument 'unlabeled_loss' to MSE. You may also change 'min_threshold' as you want.
