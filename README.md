
# Lexical-AT 
This is the codes of our IJCAI-21 paper: Long-term, Short-term and Sudden Event: Trading Volume Movement Prediction with Graph-based Multi-view Modeling.
![The architecture of our CGM](https://github.com/lancopku/CGM/blob/master/model.png)
# structure of source code 
```
src_classification/regression 
    -graph 
        -correlation.py 
        -file.py 
        -file_overnight.py (processing data)
        -utils.py 
    -models 
        -attention.py 
        -glstm.py 
        -cgm.py (our model)
        -slstm.py 
        -transformer.py 
    -criterion.py 
    -Data.py 
    -dcca.py
    -file.py 
    -lr_scheduler.py 
    -optims.py 
    -train.py (main process)
    -utils.py 
```
# shell scripts for training 

## for volume movement classification
```
CUDA_VISIBLE_DEVICES=0 python3 src_classification/train.py -config config_classification.yaml -verbose -log graph_dcca_classification
```
## for volume movement Regression 
```
CUDA_VISIBLE_DEVICES=0 python3 src_regression/train.py -config config_regression.yaml -verbose -log graph_dcca_regression
```
# Citation
If you use the above codes for your research, please cite our paper:
```
```