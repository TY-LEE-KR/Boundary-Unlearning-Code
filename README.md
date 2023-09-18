# Boundary Unlearning

## Requirements
Name and versions of Libraries
* python 3.8
* Pytorch 1.10.0
* torchvision 0.11.1
* Numpy 1.21.2
* Matplotlib 3.4.3 



## Datasets
We use the following two datasets in our experiments:
* CIFAR-10: this whole dataset will be downloaded automatically when you run the code in the first time. 
* Vggface2: this dataset can be downloaded from https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/. In our experiments, we randomly select 10 celebrities to compose a new dataset for
convenience.

The path of dataset and checkpoints will be set by the arguments 'dataset_dir' and 'checkpoint_dir'.

## Training

You need to train original model and retrain model from scratch for the first time running this program.


Some key arguments are listed below:
* `--method`: This should be either `boundary_shrink` or `boundary_expanding`. The former represents the **Boundary Shrink**, and the latter represents **Boundary Expanding**.
* `--forget_class`: This represents the target class to be unlearned. It should be between 0 and 9.
* `--train`: This determines whether the model is trained from scratch. You should use it if this program is run for the first time.
* `--evaluation`: If used, unlearned model will be evaluated. It will return accuracy and confusion matrix.
* `--extra_exp`: Optional two improvements for **Enhanced Boundary Unlearning**, which contains 'curv' and 'weight_assign' representing curvature regularization and reweighting scheme, respectively.

One example for the first run:
```bash
python main.py --method boundary_shrink --train --evaluation
```

For Boundary Shrink, there are 3 hyperparameters that can be adjusted for further experiments.
* `bound`: This determines the noise bound in the neighbor searching process of Boundary Shrink, which controls the magnitude of noise used to push samples across boundaries.
* `step`: This represents the step size of each movement to push samples across boundaries. The finer the step size, the easier it is to find the position crossing the boundary, but it will also increase the calculation cost.
* `iter`: This represents the number of iterations for each sample to cross the boundary.

The above three hyperparameters have a synergistic effect and need to be adjusted at the same time to achieve the best results.

In the Enhanced Boundary Unlearning methods, we set the weight of curvature regularization to 0.7. The adjustment factors $s$ and $t$ in reweighting scheme are used to transfer the distances as right-size weight coefficients, and we set them to $0.5$ and $5$, respectively. 


