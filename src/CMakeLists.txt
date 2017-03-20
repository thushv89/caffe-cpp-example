add_executable(
    ODCNNTrainerTest
    ODCNNTrainer.cpp
    ODCNNTrainerTest.cpp
)
target_link_libraries(
    ODCNNTrainerTest
    ${Caffe_LIBRARIES}
    #${OpenCV_LIBS}

)
