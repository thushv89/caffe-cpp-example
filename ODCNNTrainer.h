/*
Copyright (c) 2017, Thushan Ganegedara
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder(s) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*///
// Created by thushv89 on 10.03.17.
//

#ifndef OPENDETECTION_ODHOGTRAINER_H
#define OPENDETECTION_ODHOGTRAINER_H

#include <opencv2/objdetect.hpp>
#include "common/pipeline/ODTrainer.h"
#include "common/utils/utils.h"




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

    /** \brief Class for training CNN based detector.
     *
     * Use ODCNNDetector after training with this class. This is the training class for training CNN detector. SVMlight is used here on top of the convolution layers for learning (classification/detection).
     *
     * \author Thushan Ganegedara
     */

    class HyperparamFormat
    class ODCNNTrainer : public ODTrainer
    {

	public:
	      ODCNNTrainer(std::string const &training_input_location_ = "", std::string const &trained_data_location_ = "",
	      int inputsize[], HyperparamFormat layerhyperparameters[]):
          ODTrainer(training_input_location_, trained_data_location_),
          inputSize(inputsize), layerHyperparameters(layerhyperparameters)
      {
        TRAINED_LOCATION_IDENTIFIER = "CNN";
        TRAINED_DATA_ID = "cnn.xml";

        FileUtils::createTrainingDir(getSpecificTrainingDataLocation());

      }

      int train();

      void init(){}

      int (&getInputSize())[]
      {
        return inputSize;
      }

      void setInputSize(int (&inputsize)[]){
        ODCNNTrainer::inputSize = inputsize;
      }

      int (&getLayerHyperparameters())[]
      {
        return layerHyperparameters;
      }

      void setLayerHyperparameters(int (&inputsize)[]){
        ODCNNTrainer::layerHyperparameters = layerHyperparameters;
      }

      protected:
        int inputSize[];
        int layerHyperparameters[];

    }
  }
}
