#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono>
#include <mutex>
#include <queue>
#include <thread>
#include "json.hpp"

#include "func.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

const int POSE_PAIRS_MPI[17][2] = {
    {1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10}, {1,11}, {11,12}, {12,13},
    {1,0}, {0,14}, {14,16}, {0,15}, {15,17}
};

// Структура для настроек
struct Config {
    int cameraId = 0;
    int frameWidth = 640;
    int frameHeight = 480;
    int fps = 30;
    bool saveVideo = false;
    string outputPath = "output/";
    string modelPath = "models/";
    string cascadePath = "cascades/";

    void load(const string& filename);
    void save(const string& filename);
};

// Логгер
class Logger {
public:
    enum Level { INFO, WARNING, ERROR, DEBUG };
    static void log(Level level, const string& message);
    static void setLogFile(const string& filename);
private:
    static string levelToString(Level level);
    static ofstream logFile;
    static mutex logMutex;
};

// Многопоточный буфер кадров
template<typename T>
class FrameBuffer {
private:
    queue<T> buffer;
    mutex mtx;
    condition_variable cv;
    size_t maxSize;
public:
    FrameBuffer(size_t max = 30) : maxSize(max) {}
    void push(const T& frame);
    bool pop(T& frame);
    void clear();
};