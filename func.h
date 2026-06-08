#pragma once
#include "headers.h"

// Глобальная переменная для ID камеры
extern int g_cameraId;

void simpleCamera();
void motionDetection();
void colorTracking();
void faceDetection();
void edgeDetection();
void motionRecord();
void camera_selection();
void CAM1();
void body();
int selectCamera();

// Новые функции
void colorTrackingAdvanced();
void faceDetectionDNN();
void motionDetectionThreaded();
void recordWithEffects();