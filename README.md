# Train & Pruning with PyTorch
by shi wenqi & hou-yz, based on `kuangliu/pytorch-cifar` & `jacobgil/pytorch-pruning`

improve inference speed and reduce intermediate feature sizes to favor distributed inference (local device compute half of the model and upload the feature for further computing on stronger devices or cloud).

- pruning stage-1: prune the whole model to increase inference speed and slightly reduce intermediate feature sizes.

- pruning stage-2: (based on step-1's model) for each split-point (where the intermediate feature is transferred to another device for further computation), specifically prune the layer just before the split-point to reduce intermediate feature sizes even more.

only support python3 with pytorch = 0.3.1;
model trained on cifar-10, tested on vgg-16 & alexnet.

also added auto-logging and auto chart-drawing.

## usage
### training:
```lua
python main.py --train          # train from scratch
python main.py --resume         # resume training
```

### 2-step pruning:

first, in step-1, you can prune the whole model by
```lua 
python main.py --prune          # prune the whole model
```

once you finished step-1, you can then prune each layer (step-2) individually for minimum bandwidth requirement with 
``` lua
python main.py --prune_layer    # prune layers and save models separately
```
### test 2-step pruning
``` lua
python main.py --prune_layer_test_accuracy    #prune layers until remain 1 filter
```

### comparison with feature coding + fine-tuning
``` lua
python main.py --test_encode
```

### chart drawing:

for logging and excel chart drawing, try 
```lua
python maim.py --test_pruned    # test the pruned model and save *.json logs
python draw_chart.py
```
which automatically generate the `chart.xlsx` file.


## updates
- added pruning features;
- added 2-stage pruning method: --prune & --prune_layer
- added draw_chart with `openpyxl` (open in excel);
- added cpu-only support and windows support.

## updates by shiwenqi
- optimized second stage pruing
- added comparison with feature coding + fine-tuning
- added charts/curves

## doing
- adding spport of ResNet
- adding comparison with early-exit branchynet

