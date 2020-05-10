## MixMatch_basic

### Excution

```
nsml run -d fashion_eval -e main.py -a "--unlabeled_loss CEE --min_threshold 0.5"
```

If you want to use MSE for computing unlabeled loss, change the argument 'unlabeled_loss' to MSE. You may also change 'min_threshold' as you want.
