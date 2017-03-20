## caffe-cpp-example
This is a C++ example using [caffe](http://caffe.berkeleyvision.org/) to train with a dataset and test using the trained model.

## Steps to run
* Configure [ex_model.prototxt](https://github.com/thushv89/caffe-cpp-example/blob/master/prototxt/ex_model.prototxt) and [ex_solver.protoxt](https://github.com/thushv89/caffe-cpp-example/blob/master/prototxt/ex_solver.prototxt) files 
  * set `net:` `snapshot_prefix:` params of ex_solver.prototxt
  * set `source` for train and test data input layers in the ex_model.prototxt
    * To populate the `data` directory refer this [README.md](https://github.com/thushv89/caffe-cpp-example/blob/master/data/README.md)
* Edit the following variables in ODCNNTrainerTest.cpp
  * set `solver_file`, `train_data_dir` and `test_data_dir`
* Build the project
* Run ODCNNTrainerTest.cpp
