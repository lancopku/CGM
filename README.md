
# CGM
This is the code of our IJCAI-21 paper: Long-term, Short-term and Sudden Event: Trading Volume Movement Prediction with Graph-based Multi-view Modeling.
![The architecture of our CGM](https://github.com/lancopku/CGM/blob/master/model.png). The original news data is shared at https://pan.baidu.com/s/1zTHKy54DOu0p9_fa4s-JVg The extraction code is：elj8. 
# Structure of the source code 
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
# Shell scripts for training CGM
## for volume movement classification task
```
CUDA_VISIBLE_DEVICES=0 python3 src_classification/train.py -config config_classification.yaml -verbose -log graph_dcca_classification
```
## for volume movement regression task
```
CUDA_VISIBLE_DEVICES=0 python3 src_regression/train.py -config config_regression.yaml -verbose -log graph_dcca_regression
```
# Citation
If you use the above code for your research, please cite our paper:
```
@inproceedings{zhao2021longterm,
  title={Long-term, Short-term and Sudden Event: Trading Volume Movement Prediction with Graph-based Multi-view Modeling},
  author={Liang Zhao, Wei Li, Ruihan Bao, Keiko Harimoto, Yunfang Wu and Xu Sun},
  booktitle={Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI} 2021, Montreal, Canada, August 21-26, 2021},
  year={2021}}
```
