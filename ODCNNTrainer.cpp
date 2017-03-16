
#include <fstream>
#include <algorithm>
#include <iterator>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//binding class for svmlight


using namespace std;
using namespace cv;

namespace od
{
  namespace g2d
  {

    static void storeCursor(void)
    {
      printf("\033[s");
    }

    static void resetCursor(void)
    {
      printf("\033[u");
    }


    double ODHOGTrainer::trainWithSVMLight(string svmModelFile, string svmDescriptorFile, vector<float> &descriptorVector)
    {
      //training takes featurefile as input, produces hitthreshold and vector as output
      printf("Calling %s\n", TRAINHOG_SVM_TO_TRAIN::getInstance()->getSVMName());
      TRAINHOG_SVM_TO_TRAIN::getInstance()->read_problem(const_cast<char *> (featuresFile.c_str()));
      TRAINHOG_SVM_TO_TRAIN::getInstance()->train(); // Call the core libsvm training procedure
      printf("Training done, saving model file!\n");
      TRAINHOG_SVM_TO_TRAIN::getInstance()->saveModelToFile(svmModelFile);

      printf("Generating representative single HOG feature vector using svmlight!\n");

      descriptorVector.resize(0);
      vector<unsigned int> descriptorVectorIndices;
      // Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
      TRAINHOG_SVM_TO_TRAIN::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
      // And save the precious to file system
      saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, svmDescriptorFile);
      // Detector detection tolerance threshold
      return hitThreshold = TRAINHOG_SVM_TO_TRAIN::getInstance()->getThreshold();
    }

    int ODHOGTrainer::train()
    {

      vector<string> positiveTrainingImages;
      vector<string> negativeTrainingImages;
      vector<string> validExtensions;
      validExtensions.push_back(".jpg");
      validExtensions.push_back(".png");
      validExtensions.push_back(".ppm");

      FileUtils::getFilesInDirectoryRec(posSamplesDir, validExtensions, positiveTrainingImages);
      FileUtils::getFilesInDirectoryRec(negSamplesDir, validExtensions, negativeTrainingImages);
      cout << "No of positive Training Files: " << positiveTrainingImages.size() << endl;
      cout << "No of neg Training Files: "<< negativeTrainingImages.size() << endl;


      if( positiveTrainingImages.size() + negativeTrainingImages.size() == 0)
      {
        printf("No training sample files found, nothing to do!\n");
        return EXIT_SUCCESS;
      }

      /// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
      setlocale(LC_ALL, "C"); // Do not use the system locale
      setlocale(LC_NUMERIC, "C");
      setlocale(LC_ALL, "POSIX");


      printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFile.c_str());
      float percent;


      fstream File;
      File.open(featuresFile.c_str(), std::fstream::out);
      if(File.is_open())
      {

        // Iterate over POS IMAGES
        for(unsigned long currentFile = 0; currentFile < positiveTrainingImages.size(); ++currentFile)
        {

          vector<float> featureVector;
          const string currentImageFile = positiveTrainingImages.at(currentFile);


          calculateFeaturesFromInput(currentImageFile, featureVector, hog_);
          if(!featureVector.empty())
          {
            /* Put positive or negative sample class to file,
            * true=positive, false=negative,
            * and convert positive class to +1 and negative class to -1 for SVMlight
            */
            File << "+1";
            // Save feature vector components
            for(unsigned int feature = 0; feature < featureVector.size(); ++feature)
            {
              File << " " << (feature + 1) << ":" << featureVector.at(feature);
            }
            File << endl;
          }
        }


        // Iterate over NEG IMAGES
        for(unsigned long currentFile = 0; currentFile < negativeTrainingImages.size(); ++currentFile)
        {
          const string currentImageFile = negativeTrainingImages.at(currentFile);
          handleNegetivefile(currentImageFile, hog_, File);
        }

        printf("\n");
        File.flush();
        File.close();

      } else
      {
        printf("Error opening file '%s'!\n", featuresFile.c_str());
        return EXIT_FAILURE;
      }


      //train them with SVM
      vector<float> descriptorVector;
      hitThreshold = trainWithSVMLight(svmModelFile, descriptorVectorFile, descriptorVector);

      // Pseudo test our custom detecting vector
      hog_.setSVMDetector(descriptorVector);
      printf(
          "Testing training phase using training set as test set (just to check if training is ok - no detection quality conclusion with this!)\n");
      detectTrainingSetTest(hog_, hitThreshold, positiveTrainingImages, negativeTrainingImages);



      if(train_hard_negetive_)
      {
        cout << "Preparing for training HARD negetive windows" << endl;
        //create hard training examples
        createHardTrainingData(hog_, hitThreshold, negativeTrainingImages);
        //train again
        hitThreshold = trainWithSVMLight(svmModelHard, descriptorVectorHard, descriptorVector);

        // Pseudo test our custom detecting vector
        hog_.setSVMDetector(descriptorVector);
        printf(
            "Testing training phase using training set as test set after HARD EXAMPLES (just to check if training is ok - no detection quality conclusion with this!)\n");
        detectTrainingSetTest(hog_, hitThreshold, positiveTrainingImages, negativeTrainingImages);

      }

      save(getSpecificTrainingDataLocation() + "/odtrained." + TRAINED_DATA_ID_);

      return EXIT_SUCCESS;
    }
