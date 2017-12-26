## Implementation of [ACKTR](https://arxiv.org/abs/1708.05144) algorithm.

The code is organized as follows:
* **models.py**: implementations of A2C and A3C
* **kfac.py**: implementation of K-FAC optimizer
* **ac\_nets.py**: neural network architectures of actor and critic for different environments
* **storage.py**: implementation of data structure to efficiently store information during learning
* **utils.py**: utils for models and optimizer

See demo [here](https://github.com/nikishin-evg/acktr_pytorch/blob/master/example_of_usage.ipynb).


------------------------------------------

Code is developed and supported by:
* Iurii Kemaev [hbq1](https://github.com/hbq1) (*y.kemaev@gmail.com*)
* Maxim Kuznetsov [binom16](https://github.com/binom16) (*binom16@gmail.com*)
* Eugenii Nikishin [nikishin-evg](https://github.com/nikishin-evg) (*nikishin.evg@gmail.com*)
