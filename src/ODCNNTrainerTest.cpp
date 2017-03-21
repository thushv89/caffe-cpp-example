#include <string>
#include<ODCNNTrainer.h>

using std::string;

int main() {

    std::string solver_file = "/home/thushv89/gsoc2017/caffe-cpp-example/prototxt/ex_solver.prototxt";
    std::string train_data_dir = "/home/thushv89/gsoc2017/caffe-cpp-example/data/mnist/mnist_train_data";
    std::string test_data_dir = "/home/thushv89/gsoc2017/caffe-cpp-example/data/mnist/mnist_test_data";
    std::string snapshot_dir = "/home/thushv89/gsoc2017/caffe-cpp-example/snapshots/";
    ODCNNTrainer cnnModel(solver_file,train_data_dir,test_data_dir,snapshot_dir);
    cnnModel.train();
}
