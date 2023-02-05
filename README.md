## Semi Supervised HDBSCAN*
Here we provide an implementation of the base semi-supervised classification model built around HDBSCAN* as per Gertrudes et al. 2019. It can be loaded and used as follows:
```python
from ss_hdscan import HDBScanSSC
clf = HDBSCanSSC(min_cluster_size=5)
clf.fit(X=dataset.data, y=dataset.target)
```
Where *dataset** is a *Scikit-Learn bunch dataset*. The ```dataset.target``` contains the class labels, use the label ```-1``` to specify unlabeled, as per the standard in Scikit-Learn semi-supervised learning implementations.

After creating the classifier instance, and fitting it with data by `clf.fit()`. The fitting function triggers transduction. You can obtain the transductive predictions by referring to the `clf.transduction_`
