This is a repository of the code used to generate the experiments in **FlipOut : Uncovering redundant weights via sign flipping**. It contains the implementation of our proposed method as well as for the baselines. A Conda virtual environment can be created from ```environment.yml```.

For the Imagenette experiments, please download the dataset (https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz) and place them in a folder named ```data``` in the root directory.
Some of the methods perform pruning periodically and can have their final sparsity determined by the pruning rate (how many parameters are removed each time, in percentages) and frequency (how often we prune, in epochs). Below we include a reference table for the sparsities we used in our experiments and the pruning frequencies, assuming 350 epochs of training and a pruning rate of 50%.


| Sparsity | Prune frequency |
| --- | --- |
| 75% | 117 |
| 93.75% | 70 |
| 98.44% | 50 |
| 99.61% | 39 |
| 99.9% | 32 |

For SNIP, the sparsity can be directly selected. For Hoyer-Square, it is a function of the regularization term as well as the pruning threshold. 

Below, we provide a table of results as well as example commands used for generating the runs on VGG19.

**VGG-19:**
| Method | Sparsity (%)| Accuracy (%)|
| --- | --- | --- |
| FlipOut | 99.9 | **87.39 ± 0.23** |
| GlobalMagnitude | 99.9 | 82.89 ± 2.00 |
| Random | 99.9 | 0.10 ± 0 |
| SNIP | 99.9 | 0.10 ± 0 |
| Hoyer-Square | 99.89 | 82.78 |


**FlipOut (@99.9%, lambda=1) :**
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion flipout --prune_freq 32 --prune_rate 0.5 \
                --noise --noise_scale_factor 1 \
                --comment="test flipout" \
                --logdir=resnet18/
```
**Global Magnitude (@99.9%):**
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion global_magnitude --prune_freq 32 --prune_rate 0.5 \
                --comment="test global magnitude" \
                --logdir=resnet18/
```
**Random (@99.9%):**
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion random --prune_freq 32 --prune_rate 0.5 \
                --comment="test random" \
                --logdir=resnet18/
```
**SNIP (@99.9%)**:
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 350 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --prune_criterion snip --snip_sparsity 0.999 \
                --comment="test snip" \
                --logdir=resnet18/
```
**Hoyer-Square (lambda=6e-5, threshold=1e-4 )**:
```
python main.py --model vgg19 --dataset cifar10 -bs 128 -e 500 -lr 0.1 \
                --opt sgd --momentum 0.9 --reg_type wdecay --lambda 5e-4 --use_scheduler \
                --milestones 150 250 \
                --add_hs --hoyer_lambda 6e-5 \
                --prune_criterion threshold --magnitude_threshold 1e-4 \
                --prune_freq 350 --stop_hoyer_at 350 \
                --comment="test hoyer-square" \
                --logdir=resnet18/
```

The runs are saved in the directory specified by ```logdir``` with the filename ```comment``` and can be inspected with Tensorboard.

To run on different model/dataset combinations, simply replace the ```-m``` and ```-d``` arguments, i.e. ```-m vgg19 -d cifar10``` or ```-m densenet121 -d imagenette```. For other levels of sparsity, please refer to the table at the top of the page.
