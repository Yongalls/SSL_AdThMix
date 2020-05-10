## Fixed Threshold
MixMatch with Fixed Threshold applied for pseudo labels of unlabeled data.
In this algorith, one fixed value of threshold is applied for overall training phase. 

### Excution

```
nsml run -d fashion_eval -e main.py -a "--threshold 0.8"
```

You can change the value of threshold using above argument. 
