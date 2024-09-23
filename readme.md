### How to Run : 

Pre-requisites : 

Install the dependencies in a separate virtual enviroment using the steps mentioned in : `<insert link here>`


1.) `datasets.py` 

- Insert your custom links to the train, test and validation files for the particular configuration of Quark-Gluon jets you wish to classify in the variables `train_link`, `test_link`
and `val_link`.

- Uncomment all the indicated lines in the `main()` function and run it.
  This will download the `train.h5`, `test.h5` and `val.h5` files for the Quark-Gluon Tagging task.

- From the home directory, run :

```> python3 datasets.py                                                 ```

- Doing so will access the data in chunks from the raw data files, perform the required pre-processing on the 4-momenta `(p_x, p_y, p_z, E)` and extract shifted and boosted variables such as 
`(η, φ, θ, x, y, z)` in a highly memory-efficient manner using `dask` and `awkward` arrays and save them in chunked `.parquet` files.

- These parquet files are stored in the directory `downloads > converted` by default under the corresponding directory names. You can change the location by changing the destination directory in the
file `datasets.py`.

2.) `data_arrange.py`

- Under the `awkward_structure_parser()` block, in the `event_data` dictionary, choose all the variables you wish to keep for training purposes. A good set to keep could be just the cartesian coordinates, `(x, y, z)` or
  like `(y, φ)` and `(η, y, φ)`.
  
- Run the `main()` block of `data_arrange.py` file. You can set the names and destination addresses to the `.pkl` files that shall be created in the `main()` block itself. By default, theset too,
  are saved in the `downloads > processed` directory.

```>  python3 data_arrange.py `

- These structures that are stored within these files are highly efficient in terms of memory and space because they use `awkward arrays`.


3.) `coordinates_extract.py`

- Running the `main()` block of this file will create the simplices upto a dimension of `max_dimension = 2` and upto a filtration value of `np.inf` (by default).
- I am computing a `Rips Complex` using the `Gudhi` library. I am also passing `sparsity` parameter.
- You can edit this file as per different configurations to see what works the best. However, Message-Passing will only work when the following is upheld :
    * The Laplacian for the Complex must not be "very sparse" (value `k` takes care of this in `scnn > chebyshev.normalize`.)
    * The Complex must have atleast 1 simplex as its subset, whose rank is equal to `max_dimension`. For example, in our case, each simplex must have a triangular face.
    * `Pruning` can help our cause by removing unnecessary faces having high filtration values. However, use it with care. To specify upto what value you want to prune the simplicial
      complexes that shall be produced, specify `filtration_val` parameter in the `.compute_lapl_and_bounds()` method.


4.) `model.py`




### Topological Invariance of Lorentz Boosted Jets

My original motivation to induce non-locality in GNNs for jet classification was related to finding the `global picture`
in a local context, i.e. giving some mathematical language to the `global picture` of particle jets.

I used Algebraic Topology as a tool to push forth my ideas. Treating a collision event as a point-cloud dataset, each data 
entry in it is a point cloud of its own - or as it can be referred to as, `Particle Cloud`.

Now, my hypothesis was that the `Topological Properties` of this unordered sets of particles must give a global idea of the
nature of the event - if not totally, then partially. 

Hence, I used the newly evolving domain of `Topological Data Analysis` to explore my hypothesis. 

My findings are as below : 

1.) On using different sets of features, like the original 4-momenta `(p_x, p_y, p_z, E)`, or the 
  Cartesian coordinates of the Lorentz Boost `(x, y, z)`, or the pseudorapidity coordinates `(y, φ)` etc.,
  the Computed Topological Features were found to be invariant. 
  I inspected this by using a tool from TDA known as `Persistent Homology` to compute Persistence Diagrams 
  and Betti Numbers for different shifted and boosted systems. The figures are as presented below.

 
![Unknown-6](https://github.com/user-attachments/assets/1ed530d8-4e7d-4dd8-b99e-73851d3bbc5c)

Fig 1 : A plot of an event in the `(y, φ)` plane.

Now, in the following figure, the left column shows the persistence plots of the first 5 events in the `(x, y, z)` plane
while the right column shows the plots in `(y, φ)` plane.

![Unknown](https://github.com/user-attachments/assets/b23c972e-cab3-462a-bd8b-64a366b15d61)

As you can see, the Topological Invariants hold across Lorentz Boosts and shifts, like other Invariants.
The paper by [Dawson et. al](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_176.pdf) reinforeces
my ideas and provides a promising footing to carry on further work.



### Simplicial Message Passing Networks

I am using a modification of the Simplicial Neural Network introduced in the paper by [Ebli et al.](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_176.pdf)
which is a simple version of a generalization of a GNN-like message passing framework to Abstract Simplicial Complexes.

The adjacency information is created using 2 entities -> `Laplacians` and `Boundary Matrices`, which are shown in the following figure : 
<img width="551" alt="Screenshot 2024-09-23 at 7 35 22 PM" src="https://github.com/user-attachments/assets/3fc06d4f-8d7c-4d93-9978-6653a16e9355">

For theoretical details, please refer to this comprehensive [survey paper](https://arxiv.org/abs/2304.10031) by Papillon et. al


Below is an idea of how compressibility can be achieved 
<img width="1125" alt="Screenshot 2024-09-23 at 7 03 03 PM" src="https://github.com/user-attachments/assets/1b91e9fe-2479-459e-b28c-3b640d5c1940">
