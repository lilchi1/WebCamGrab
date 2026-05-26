#pragma once
#include <iostream>
#include "openCV/sources/include/opencv2/opencv.hpp"
#include "func.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

const int POSE_PAIRS_MPI[17][2] = {
    {1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10}, {1,11}, {11,12}, {12,13},
    {1,0}, {0,14}, {14,16}, {0,15}, {15,17}
};

const string POSE_NAMES[] = {
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
    "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
    "LEye", "REar", "LEar"
};
/*      задача проекта
*	сделать отслеживание камеры, 
* распознование лица, конечностей,
* жестов, направление куда смотрит 
*     пользователь, мимики
* 
*/