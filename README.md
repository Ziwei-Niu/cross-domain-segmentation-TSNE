# cross-domain-segmentation-TSNE
This code is for cross-domain segmentation tasks, which can plot T-sne of domains and classes
## Installation Requirements (Three ways to choose)
### 1. Use tsne from sklearn
```
pip install scikit-learn
```


### 2. Use tsnecuda：(only for linux)
```
pip install tsnecuda
```
### 3. Use Multicore-TSNE
```
pip install Multicore-TSNE
```
If the installation is not successful, you can compile from the source code, referring to [CSDN: MulticoreTSNE安装及测试](https://blog.csdn.net/qq_45759229/article/details/120434387?ydreferer=aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzNzU3Mzk4L2FydGljbGUvZGV0YWlscy8xMjk5NzkzMjQ%2Fc3BtPTEwMDEuMjAxNC4zMDAxLjU1MDY%3D)
```
pip install cmake==3.18.4
git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git
cd Multicore-TSNE/
pip install .
```

## How to Run
1. Modify the *setup_loader.py* according to your own dataset and requirements.
2. Modify the parameters in the *TSNE.py*, such as *self.num_class*, *self.num_neighbors*.
3. Modify the domian and class information in *tsne_draw.py*.
4. Run the *tsne_draw.py*.


## Acknowledgments
The tsne code is heavily refer to the [PintheMemory](https://github.com/Genie-Kim/PintheMemory) and [CSDN用于语义分割模型的t-SNE可视化](https://blog.csdn.net/qq_33757398/article/details/129979324?spm=1001.2014.3001.5506).
Thanks to the RobustNet and TSMLDG implementations.

## T-sne plots of [PintheMemory](https://github.com/Genie-Kim/PintheMemory)
In the left part, different colors represent different classes, and in the right part, different colors represent different domains. As shown, the method in this paper is well at learning domain-invariant representation.
<p align="center">
  <img src="assets/tsneplot(GSCB).png" width="500"/>
</p>
