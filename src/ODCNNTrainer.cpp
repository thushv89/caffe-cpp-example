

#include <caffe/caffe.hpp>
#include <ODCNNTrainer.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "boost/algorithm/string.hpp"
#include "caffe/util/signal_handler.h"
#include <gflags/gflags.h>

#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fstream>
#include <iterator>



using namespace std;
using namespace cv;
using namespace caffe;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
using std::string;

DEFINE_string(model, "",
              "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
              "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
             "Optional; network level.");
DEFINE_string(stage, "",
              "Optional; network stages (not to be confused with phase), "
                      "separated by ','.");
DEFINE_string(snapshot, "",
              "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
              "Optional; the pretrained weights to initialize finetuning, "
                      "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
             "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
              "Optional; action to take when a SIGINT signal is received: "
                      "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
              "Optional; action to take when a SIGHUP signal is received: "
                      "snapshot, stop or none.");


  ODCNNTrainer :: ODCNNTrainer(string &solverfile, string &traindatadir, string &testdatadir, string &snapshotdir):
          solverFile(""), trainDataDir(""), testDataDir(""), snapshotDir(""){
    Caffe::set_mode(Caffe::CPU);
    solverFile = solverfile;
    trainDataDir = traindatadir;
    testDataDir = testdatadir;
    snapshotDir = snapshotdir;
  }


  vector<string> get_stages_from_flags() {
      vector<string> stages;
      boost::split(stages, FLAGS_stage, boost::is_any_of(","));
      LOG(INFO) << "Stages: " << FLAGS_stage;
      return stages;
  }

  caffe::SolverAction::Enum GetRequestedAction(
  const std::string& flag_value) {
    if (flag_value == "stop") {
      return caffe::SolverAction::STOP;
    }
    if (flag_value == "snapshot") {
      return caffe::SolverAction::SNAPSHOT;
    }
    if (flag_value == "none") {
      return caffe::SolverAction::NONE;
    }
    LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
  }


  int ODCNNTrainer::train()
  {
    Caffe::set_mode(Caffe::CPU);
    vector<string> stages = get_stages_from_flags();

    caffe::SolverParameter solver_param;
    // read prototxt for solver and save the params to solver_param
    caffe::ReadSolverParamsFromTextFileOrDie(solverFile, &solver_param);

    solver_param.mutable_train_state()->set_level(FLAGS_level);

    for (int i = 0; i < stages.size(); i++) {
      solver_param.mutable_train_state()->add_stage(stages[i]);
    }

      caffe::SignalHandler signal_handler(
      GetRequestedAction(FLAGS_sigint_effect),
      GetRequestedAction(FLAGS_sighup_effect));

    caffe::shared_ptr<caffe::Solver<float> >
            solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

    solver->SetActionFunction(signal_handler.GetActionFunction());

    solver->Solve();

    return EXIT_SUCCESS;
  }

  int ODCNNTrainer::test(){
    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    vector<string> stages = get_stages_from_flags();
  }

