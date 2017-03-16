#include <string>
#include<detectors/global2D/detection/ODCNNTrainer.h>

using std::string;

int main() {

    std::string solver_file = "./prototxt/ex_solver.prototxt";
    std::string train_data_dir = "./data/mnist/mnist_train_data";
    std::string test_data_dir = "./data/mnist/mnist_test_data'";

    ODCNNTrainer cnnModel(model_file,train_data_dir,test_data_dir);
    cnnModel.train();
}
