#define _CRT_SECURE_NO_WARNINGS

#include <atomic>
#include <mutex>
#include <condition_variable>
#include "func.h"
#include <thread>
#include <chrono>
#include <fstream>
#include <algorithm>

// Глобальная переменная для ID камеры
int g_cameraId = 0;

// Глобальные переменные для оптимизации
static Mat g_lastSkeletonFrame;
static bool g_hasSkeleton = false;

// ======================== ОСНОВНЫЕ ФУНКЦИИ ========================

void simpleCamera() {
    VideoCapture cap(g_cameraId);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        static int frameCount = 0;
        static double timeElapsed = 0;
        static double fps = 0;
        frameCount++;

        double currentTime = (double)getTickCount();
        if (currentTime - timeElapsed >= getTickFrequency()) {
            fps = frameCount / ((currentTime - timeElapsed) / getTickFrequency());
            frameCount = 0;
            timeElapsed = currentTime;
        }

        string fpsText = "FPS: " + to_string((int)fps);
        putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        imshow("Camera", frame);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

void motionDetection() {
    VideoCapture cap(g_cameraId);
    Mat frame, prevFrame, diff;

    cap >> prevFrame;
    cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);
    GaussianBlur(prevFrame, prevFrame, Size(21, 21), 0);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(21, 21), 0);

        absdiff(prevFrame, gray, diff);
        threshold(diff, diff, 25, 255, THRESH_BINARY);

        int motionArea = countNonZero(diff);
        string info = "Motion area: " + to_string(motionArea) + "px";
        putText(diff, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255), 1);

        imshow("Motion Detection", diff);

        if (waitKey(1) == 27) break;
        prevFrame = gray.clone();
    }
}

void colorTracking() {
    VideoCapture cap(g_cameraId);
    Mat frame, hsv, mask, result;

    Scalar lowerGreen(35, 100, 100);
    Scalar upperGreen(85, 255, 255);
    Scalar lowerRed1(0, 100, 100);
    Scalar upperRed1(10, 255, 255);
    Scalar lowerRed2(160, 100, 100);
    Scalar upperRed2(179, 255, 255);
    Scalar lowerBlue(100, 100, 100);
    Scalar upperBlue(140, 255, 255);

    cout << "Color Tracking Started. Press 'g' for green, 'r' for red, 'b' for blue" << endl;

    int currentColor = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, hsv, COLOR_BGR2HSV);

        switch (currentColor) {
        case 0:
            inRange(hsv, lowerGreen, upperGreen, mask);
            break;
        case 1: {
            Mat mask1, mask2;
            inRange(hsv, lowerRed1, upperRed1, mask1);
            inRange(hsv, lowerRed2, upperRed2, mask2);
            mask = mask1 | mask2;
            break;
        }
        case 2:
            inRange(hsv, lowerBlue, upperBlue, mask);
            break;
        }

        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        frame.copyTo(result);

        if (!contours.empty()) {
            auto largest = max_element(contours.begin(), contours.end(),
                [](const vector<Point>& a, const vector<Point>& b) {
                    return contourArea(a) < contourArea(b);
                });

            if (contourArea(*largest) > 500) {
                Rect rect = boundingRect(*largest);
                rectangle(result, rect, Scalar(0, 0, 255), 2);

                Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
                circle(result, center, 5, Scalar(255, 0, 0), -1);

                string coords = "X:" + to_string(center.x) + " Y:" + to_string(center.y);
                putText(result, coords, Point(rect.x, rect.y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
        }

        string colorName;
        switch (currentColor) {
        case 0: colorName = "GREEN"; break;
        case 1: colorName = "RED"; break;
        case 2: colorName = "BLUE"; break;
        }
        putText(result, "Tracking: " + colorName, Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        imshow("Color Tracking", result);

        char key = waitKey(1);
        if (key == 27) break;
        if (key == 'g') currentColor = 0;
        if (key == 'r') currentColor = 1;
        if (key == 'b') currentColor = 2;
    }
}

void faceDetection() {
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade;

    string faceCascadePath = "cascades/haarcascade_frontalface_default.xml";
    string eyeCascadePath = "cascades/haarcascade_eye.xml";

    if (!faceCascade.load(faceCascadePath)) {
        faceCascadePath = "haarcascade_frontalface_default.xml";
        if (!faceCascade.load(faceCascadePath)) {
            cerr << "Error loading face cascade file!" << endl;
            return;
        }
    }

    eyeCascade.load(eyeCascadePath);

    VideoCapture cap(g_cameraId);
    Mat frame, gray;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(0, 255, 0), 2);

            Mat faceROI = gray(face);
            vector<Rect> eyes;
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0, Size(20, 20));

            for (const auto& eye : eyes) {
                Rect eyeRect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
                rectangle(frame, eyeRect, Scalar(255, 0, 0), 2);
            }
        }

        string info = "Faces detected: " + to_string(faces.size());
        putText(frame, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        imshow("Face Detection", frame);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

void edgeDetection() {
    VideoCapture cap(g_cameraId);
    Mat frame, edges;

    int lowThreshold = 50;
    int highThreshold = 150;

    cout << "Edge Detection. Use UP/DOWN arrows to adjust threshold" << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(5, 5), 1.5);
        Canny(edges, edges, lowThreshold, highThreshold);

        edges = ~edges;

        string info = "Threshold: " + to_string(lowThreshold) + "-" + to_string(highThreshold);
        putText(edges, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255), 1);

        imshow("Edge Detection", edges);

        int key = waitKey(1);
        if (key == 27) break;
        if (key == 2490368) {
            lowThreshold = min(100, lowThreshold + 5);
            highThreshold = min(200, highThreshold + 10);
        }
        if (key == 2621440) {
            lowThreshold = max(10, lowThreshold - 5);
            highThreshold = max(20, highThreshold - 10);
        }
    }
}

void motionRecord() {
    VideoCapture cap(g_cameraId);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 30);

    Mat frame, prevFrame, gray, diff;
    Mat recordingFrame;

    cap >> prevFrame;
    if (prevFrame.empty()) {
        cerr << "Cannot get initial frame!" << endl;
        return;
    }
    cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);
    GaussianBlur(prevFrame, prevFrame, Size(21, 21), 0);

    const int MOTION_THRESHOLD = 25;
    const double MOTION_AREA_RATIO = 0.01;

    bool isRecording = false;
    bool motionDetected = false;
    VideoWriter writer;
    string filename;
    int recordingCounter = 0;
    int framesSinceLastMotion = 0;
    const int STOP_AFTER_FRAMES = 30;

    int frameWidth = 640;
    int frameHeight = 480;
    int totalMotionPixels = 0;

    cout << "Motion Recording Started. Press ESC to exit." << endl;
    cout << "Press 'm' for manual recording, 's' to stop" << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(21, 21), 0);

        absdiff(prevFrame, gray, diff);
        threshold(diff, diff, MOTION_THRESHOLD, 255, THRESH_BINARY);

        totalMotionPixels = countNonZero(diff);
        int frameArea = frameWidth * frameHeight;
        motionDetected = (totalMotionPixels > frameArea * MOTION_AREA_RATIO);

        if (motionDetected && !isRecording) {
            isRecording = true;
            framesSinceLastMotion = 0;

            auto now = chrono::system_clock::now();
            auto time_t_now = chrono::system_clock::to_time_t(now);
            char timestamp[64];
            strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&time_t_now));
            filename = "motion_" + string(timestamp) + "_" + to_string(recordingCounter) + ".mp4";

            int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
            writer.open(filename, fourcc, 20.0, Size(frameWidth, frameHeight), true);

            if (writer.isOpened()) {
                cout << ">>> RECORDING STARTED: " << filename << endl;
            }
            else {
                cerr << "Failed to create video file!" << endl;
                isRecording = false;
            }
        }

        if (isRecording) {
            Mat frameToRecord = frame.clone();

            string recordInfo = "RECORDING - Motion: " + to_string(totalMotionPixels) + "px";
            putText(frameToRecord, recordInfo, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);

            circle(frameToRecord, Point(frameWidth - 20, 20), 10, Scalar(0, 0, 255), -1);

            writer.write(frameToRecord);

            if (!motionDetected) {
                framesSinceLastMotion++;
            }
            else {
                framesSinceLastMotion = 0;
            }

            if (framesSinceLastMotion >= STOP_AFTER_FRAMES) {
                isRecording = false;
                writer.release();
                cout << ">>> RECORDING STOPPED: " << filename << endl;
                recordingCounter++;
            }
        }

        Mat displayFrame = frame.clone();

        string statusText;
        Scalar statusColor;
        if (isRecording) {
            statusText = "● RECORDING";
            statusColor = Scalar(0, 0, 255);
        }
        else if (motionDetected) {
            statusText = "MOTION DETECTED";
            statusColor = Scalar(0, 255, 255);
        }
        else {
            statusText = "Waiting for motion";
            statusColor = Scalar(0, 255, 0);
        }
        putText(displayFrame, statusText, Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);

        int motionPercent = (totalMotionPixels * 100) / frameArea;
        rectangle(displayFrame, Point(10, 60), Point(10 + (motionPercent * 2), 75),
            Scalar(0, 255, 0), -1);
        rectangle(displayFrame, Point(10, 60), Point(210, 75),
            Scalar(255, 255, 255), 1);

        imshow("Motion Recording", displayFrame);

        prevFrame = gray.clone();

        char key = waitKey(1);
        if (key == 27) break;

        if (key == 'm' && !isRecording) {
            isRecording = true;
            framesSinceLastMotion = 0;
            auto now = chrono::system_clock::now();
            auto time_t_now = chrono::system_clock::to_time_t(now);
            char timestamp[64];
            strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&time_t_now));
            filename = "manual_" + string(timestamp) + ".avi";
            writer.open(filename, VideoWriter::fourcc('X', 'V', 'I', 'D'),
                20.0, Size(frameWidth, frameHeight), true);
            cout << ">>> MANUAL RECORDING STARTED" << endl;
        }

        if (key == 's' && isRecording) {
            isRecording = false;
            writer.release();
            cout << ">>> MANUAL RECORDING STOPPED" << endl;
        }
    }

    if (isRecording) {
        writer.release();
    }
    cap.release();
    destroyAllWindows();
}

// ======================== УЛУЧШЕННЫЕ ФУНКЦИИ ========================

void colorTrackingAdvanced() {
    VideoCapture cap(g_cameraId);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    vector<Scalar> lowerBounds = {
        Scalar(35, 100, 100),  // Green
        Scalar(0, 100, 100),   // Red
        Scalar(100, 100, 100)  // Blue
    };
    vector<Scalar> upperBounds = {
        Scalar(85, 255, 255),
        Scalar(10, 255, 255),
        Scalar(140, 255, 255)
    };
    vector<string> colorNames = { "GREEN", "RED", "BLUE" };
    vector<Scalar> drawColors = { Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 0, 0) };

    // Для красного нужна дополнительная маска
    Scalar lowerRed2(160, 100, 100);
    Scalar upperRed2(179, 255, 255);

    int currentColor = 0;
    int minObjectArea = 500;
    int maxObjects = 10;

    cout << "Advanced Color Tracking Started" << endl;
    cout << "Controls: [G] Green [R] Red [B] Blue [ESC] Exit" << endl;
    cout << "Tracking up to " << maxObjects << " objects simultaneously" << endl;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        Mat mask;
        if (currentColor == 1) {
            Mat mask1, mask2;
            inRange(hsv, lowerBounds[1], upperBounds[1], mask1);
            inRange(hsv, lowerRed2, upperRed2, mask2);
            mask = mask1 | mask2;
        }
        else {
            inRange(hsv, lowerBounds[currentColor], upperBounds[currentColor], mask);
        }

        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        sort(contours.begin(), contours.end(),
            [](const vector<Point>& a, const vector<Point>& b) {
                return contourArea(a) > contourArea(b);
            });

        int objectsFound = 0;
        for (const auto& contour : contours) {
            if (objectsFound >= maxObjects) break;

            double area = contourArea(contour);
            if (area < minObjectArea) continue;

            Rect rect = boundingRect(contour);
            Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);

            rectangle(frame, rect, drawColors[currentColor], 2);
            circle(frame, center, 5, drawColors[currentColor], -1);

            string info = "#" + to_string(objectsFound + 1) + " A:" + to_string((int)area);
            putText(frame, info, Point(rect.x, rect.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.4, drawColors[currentColor], 1);

            objectsFound++;
        }

        string stats = "Color: " + colorNames[currentColor] + " | Objects: " + to_string(objectsFound);
        putText(frame, stats, Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        string controls = "[G] Green [R] Red [B] Blue [ESC] Exit";
        putText(frame, controls, Point(10, frame.rows - 10),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

        imshow("Color Tracking (Multi-Object)", frame);

        char key = waitKey(1);
        if (key == 27) break;
        if (key == 'g') currentColor = 0;
        if (key == 'r') currentColor = 1;
        if (key == 'b') currentColor = 2;
    }
}

void faceDetectionDNN() {
    string modelPath = "models/opencv_face_detector_uint8.pb";

    string configPath = "models/opencv_face_detector.pbtxt";

    ifstream modelFile(modelPath);
    if (!modelFile.good()) {
        cerr << "DNN face model not found, using Haar cascade fallback" << endl;
        faceDetection();
        return;
    }

    Net net;
    try {
        net = readNetFromTensorflow(modelPath, configPath);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
    catch (const exception& e) {
        cerr << "Error loading DNN model: " << e.what() << endl;
        faceDetection();
        return;
    }

    VideoCapture cap(g_cameraId);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    cout << "DNN Face Detection started. Press ESC to exit." << endl;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123));
        net.setInput(blob);
        Mat detections = net.forward();

        // ПРОВЕРКА: убедимся что detections не пустая и имеет правильные размеры
        if (detections.empty()) {
            cout << "Detections empty, skipping frame" << endl;
            imshow("Face Detection (DNN)", frame);
            if (waitKey(1) == 27) break;
            continue;
        }

        // Получаем размеры
        int numDetections = detections.size[2];

        for (int i = 0; i < numDetections; i++) {
            // Используем ptr для безопасного доступа
            float* data = detections.ptr<float>(0, 0, i);
            float confidence = data[2];

            if (confidence > 0.5) {
                int x1 = data[3] * frame.cols;
                int y1 = data[4] * frame.rows;
                int x2 = data[5] * frame.cols;
                int y2 = data[6] * frame.rows;

                x1 = max(0, x1);
                y1 = max(0, y1);
                x2 = min(frame.cols - 1, x2);
                y2 = min(frame.rows - 1, y2);

                rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

                string confidenceText = to_string((int)(confidence * 100)) + "%";
                putText(frame, confidenceText, Point(x1, y1 - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }
        }

        string info = "DNN Face Detection | Press ESC to exit";
        putText(frame, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        imshow("Face Detection (DNN)", frame);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

void motionDetectionThreaded() {
    VideoCapture cap(g_cameraId);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_BUFFERSIZE, 2);

    Mat prevFrame;
    cap >> prevFrame;
    cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);
    GaussianBlur(prevFrame, prevFrame, Size(21, 21), 0);

    atomic<bool> running(true);
    Mat currentFrame;
    mutex frameMutex;

    cout << "Threaded Motion Detection started. Press ESC to exit." << endl;

    thread processingThread([&]() {
        while (running) {
            Mat frame, gray, diff;
            {
                lock_guard<mutex> lock(frameMutex);
                if (currentFrame.empty()) {
                    this_thread::sleep_for(chrono::milliseconds(10));
                    continue;
                }
                frame = currentFrame.clone();
            }

            cvtColor(frame, gray, COLOR_BGR2GRAY);
            GaussianBlur(gray, gray, Size(21, 21), 0);

            absdiff(prevFrame, gray, diff);
            threshold(diff, diff, 25, 255, THRESH_BINARY);

            int motionArea = countNonZero(diff);
            string info = "Motion area: " + to_string(motionArea) + "px";
            putText(frame, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);

            Mat motionColor;
            cvtColor(diff, motionColor, COLOR_GRAY2BGR);
            Mat smallMotion;
            resize(motionColor, smallMotion, Size(160, 120));
            smallMotion.copyTo(frame(Rect(10, frame.rows - 130, 160, 120)));
            rectangle(frame, Rect(10, frame.rows - 130, 160, 120), Scalar(255, 255, 255), 1);

            prevFrame = gray;

            {
                lock_guard<mutex> lock(frameMutex);
                currentFrame = frame;
            }
        }
        });

    while (running) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        {
            lock_guard<mutex> lock(frameMutex);
            currentFrame = frame.clone();
        }

        Mat displayFrame;
        {
            lock_guard<mutex> lock(frameMutex);
            if (!currentFrame.empty()) {
                displayFrame = currentFrame.clone();
            }
        }

        if (!displayFrame.empty()) {
            imshow("Motion Detection (Threaded)", displayFrame);
        }

        if (waitKey(1) == 27) {
            running = false;
            break;
        }
    }

    processingThread.join();
    cap.release();
    destroyAllWindows();
}

void recordWithEffects() {
    VideoCapture cap(g_cameraId);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    namespace fs = std::filesystem;
    if (!fs::exists("output")) {
        fs::create_directory("output");
    }

    int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    string outputFile = "output/effect_video_" +
        to_string(chrono::system_clock::now().time_since_epoch().count()) +
        ".avi";

    VideoWriter writer(outputFile, fourcc, 20.0, Size(640, 480));

    int effectType = 0;
    bool recording = false;

    cout << "\n=== RECORD WITH EFFECTS ===" << endl;
    cout << "Controls:" << endl;
    cout << "  [0] Normal" << endl;
    cout << "  [1] Sepia" << endl;
    cout << "  [2] Edge Detection" << endl;
    cout << "  [3] Blur" << endl;
    cout << "  [4] Negative" << endl;
    cout << "  [R] Start/Stop Recording" << endl;
    cout << "  [S] Screenshot" << endl;
    cout << "  [ESC] Exit" << endl;
    cout << "\nRecording: OFF" << endl;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        Mat displayFrame;

        switch (effectType) {
        case 0:
            displayFrame = frame.clone();
            break;
        case 1: {
            Mat sepia;
            cvtColor(frame, sepia, COLOR_BGR2YUV);
            vector<Mat> channels;
            split(sepia, channels);
            channels[1] = channels[1] * 0.5;
            channels[2] = channels[2] * 0.5;
            merge(channels, sepia);
            cvtColor(sepia, displayFrame, COLOR_YUV2BGR);
            break;
        }
        case 2: {
            cvtColor(frame, displayFrame, COLOR_BGR2GRAY);
            Canny(displayFrame, displayFrame, 50, 150);
            cvtColor(displayFrame, displayFrame, COLOR_GRAY2BGR);
            break;
        }
        case 3: {
            GaussianBlur(frame, displayFrame, Size(35, 35), 0);
            break;
        }
        case 4: {
            displayFrame = Scalar(255, 255, 255) - frame;
            break;
        }
        default:
            displayFrame = frame.clone();
            break;
        }

        string effectName;
        switch (effectType) {
        case 0: effectName = "NORMAL"; break;
        case 1: effectName = "SEPIA"; break;
        case 2: effectName = "EDGE"; break;
        case 3: effectName = "BLUR"; break;
        case 4: effectName = "NEGATIVE"; break;
        }

        putText(displayFrame, "Effect: " + effectName, Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        if (recording) {
            circle(displayFrame, Point(displayFrame.cols - 20, 20), 8, Scalar(0, 0, 255), -1);
            putText(displayFrame, "REC", Point(displayFrame.cols - 60, 25),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            writer.write(displayFrame);
        }

        string recStatus = recording ? "RECORDING" : "STOPPED";
        putText(displayFrame, "[" + recStatus + "]", Point(10, 60),
            FONT_HERSHEY_SIMPLEX, 0.5, recording ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 1);

        imshow("Effect Recording", displayFrame);

        int key = waitKey(1);
        if (key == 27) break;
        if (key >= '0' && key <= '4') effectType = key - '0';
        if (key == 'r' || key == 'R') {
            recording = !recording;
            cout << "Recording: " << (recording ? "ON" : "OFF") << endl;
        }
        if (key == 's' || key == 'S') {
            string screenshot = "output/screenshot_" + to_string(time(nullptr)) + ".jpg";
            imwrite(screenshot, displayFrame);
            cout << "Screenshot saved: " << screenshot << endl;
        }
    }

    writer.release();
    cout << "Video saved: " << outputFile << endl;

    cap.release();
    destroyAllWindows();
}

int selectCamera() {
    cout << "Available cameras:" << endl;

    vector<int> availableCams;
    for (int i = 0; i < 10; i++) {
        VideoCapture test(i);
        if (test.isOpened()) {
            availableCams.push_back(i);
            test.release();
            cout << "  Camera " << i << " - Available" << endl;
        }
    }

    if (availableCams.empty()) {
        cerr << "No cameras found!" << endl;
        return -1;
    }

    cout << "Select camera ID (0-" << availableCams.back() << "): ";
    int choice;
    cin >> choice;

    if (find(availableCams.begin(), availableCams.end(), choice) != availableCams.end()) {
        cout << "Camera " << choice << " selected" << endl;
        return choice;
    }

    cout << "Invalid selection, using camera 0" << endl;
    return 0;
}

// ======================== СТАРЫЕ ФУНКЦИИ ДЛЯ СОВМЕСТИМОСТИ ========================

void camera_selection() {
    // Выбор камеры при запуске
    cout << "\n========================================\n";
    cout << "        CAMERA SELECTION               \n";
    cout << "========================================\n";

    vector<int> availableCams;
    for (int i = 0; i < 10; i++) {
        VideoCapture test(i);
        if (test.isOpened()) {
            availableCams.push_back(i);
            test.release();
            cout << "  Camera " << i << " - Available" << endl;
        }
    }

    if (availableCams.empty()) {
        cerr << "No cameras found!" << endl;
        return;
    }

    cout << "Select camera ID (0-" << availableCams.back() << "): ";
    cin >> g_cameraId;

    if (find(availableCams.begin(), availableCams.end(), g_cameraId) == availableCams.end()) {
        cout << "Invalid selection, using camera 0" << endl;
        g_cameraId = 0;
    }
}
void CAM1() {
    
    // camera_selection();

    cout << "\n========================================\n";
    cout << "        CAMERA MODE SELECTOR            \n";
    cout << "========================================\n";
    cout << " 1 - Simple Camera                      \n";
    cout << " 2 - Motion Detection                   \n";
    cout << " 3 - Color Tracking (Single Object)     \n";
    cout << " 4 - Color Tracking (Multi-Object)      \n";
    cout << " 5 - Edge Detection                     \n";
    cout << " 6 - Face Detection (DNN)               \n";
    cout << " 7 - Motion Detection (Threaded)        \n";
    cout << " 8 - Record with Effects                \n";
    cout << " 9 - Motion Recording                   \n";
    cout << " 0 - Body Skeleton                      \n";
    cout << "========================================\n";
    cout << "Choice: ";

    int choice;
    cin >> choice;

    switch (choice) {
    case 1: simpleCamera(); break;
    case 2: motionDetection(); break;
    case 3: colorTracking(); break;
    case 4: colorTrackingAdvanced(); break;
    case 5: edgeDetection(); break;
    case 6: faceDetectionDNN(); break;
    case 7: motionDetectionThreaded(); break;
    case 8: recordWithEffects(); break;
    case 9: motionRecord(); break;
    case 0: body(); break;
    default:
        cout << "Invalid choice, starting simple camera..." << endl;
        simpleCamera();
        break;
    }
}


void body() {
    cout << "OpenCV version: " << CV_VERSION << endl;

    ifstream prototxtFile("models/pose_deploy_linevec_faster_4_stages.prototxt");
    ifstream caffemodelFile("models/pose_iter_160000.caffemodel");

    if (!prototxtFile.good() || !caffemodelFile.good()) {
        cerr << "ERROR: Model files not found!" << endl;
        cerr << "Required files:" << endl;
        cerr << "  - pose_deploy_linevec_faster_4_stages.prototxt" << endl;
        cerr << "  - pose_iter_160000.caffemodel" << endl;
        cerr << "Download from: https://github.com/CMU-Perceptual-Computing-Lab/openpose" << endl;
        return;
    }

    prototxtFile.close();
    caffemodelFile.close();

    int inWidth = 256;
    int inHeight = 256;
    float thresh = 0.1f;
    const int SKELETON_DRAW_INTERVAL = 5;

    string prototxt = "models/pose_deploy_linevec_faster_4_stages.prototxt";
    string caffemodel = "models/pose_iter_160000.caffemodel";

    Net net;
    try {
        net = readNetFromCaffe(prototxt, caffemodel);
        if (net.empty()) {
            cerr << "Cannot load model!" << endl;
            return;
        }
    }
    catch (const exception& e) {
        cerr << "Error loading model: " << e.what() << endl;
        return;
    }

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    int frameWidth = 320;
    int frameHeight = 240;

    VideoCapture cap(g_cameraId);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    cap.set(CAP_PROP_BUFFERSIZE, 1);
    cap.set(CAP_PROP_FRAME_WIDTH, frameWidth);
    cap.set(CAP_PROP_FRAME_HEIGHT, frameHeight);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    cout << "Body tracker started. Press ESC to exit." << endl;
    cout << "Resolution: " << frameWidth << "x" << frameHeight << endl;
    cout << "Controls: [R] Reset counter [T] Change threshold" << endl;

    Mat frame;
    int frameCounter = 0;
    vector<Point> points(18);

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        frameCounter++;
        Mat displayFrame = frame.clone();

        if (frameCounter % SKELETON_DRAW_INTERVAL == 0) {
            Mat inputBlob = blobFromImage(frame, 1.0 / 255.0,
                Size(inWidth, inHeight),
                Scalar(0, 0, 0), true, false);

            net.setInput(inputBlob);
            Mat output = net.forward();

            int H = output.size[2];
            int W = output.size[3];

            fill(points.begin(), points.end(), Point(-1, -1));

            for (int i = 0; i < 16; i++) {
                Mat heatMap(H, W, CV_32F, output.ptr<float>(0, i));
                double conf;
                Point maxLoc;
                minMaxLoc(heatMap, nullptr, &conf, nullptr, &maxLoc);

                if (conf > thresh) {
                    points[i] = Point(
                        maxLoc.x * frame.cols / W,
                        maxLoc.y * frame.rows / H
                    );
                }
            }

            for (int i = 0; i < 17; i++) {
                int from = POSE_PAIRS_MPI[i][0];
                int to = POSE_PAIRS_MPI[i][1];

                if (from < 16 && to < 16 &&
                    points[from].x > 0 && points[to].x > 0 &&
                    points[from].y > 0 && points[to].y > 0) {

                    line(displayFrame, points[from], points[to], Scalar(0, 255, 0), 2);
                }
            }

            for (int i = 0; i < 16; i++) {
                if (points[i].x > 0 && points[i].y > 0) {
                    circle(displayFrame, points[i], 4, Scalar(0, 0, 255), -1);
                }
            }

            g_lastSkeletonFrame = displayFrame.clone();
            g_hasSkeleton = true;
        }
        else if (g_hasSkeleton) {
            displayFrame = g_lastSkeletonFrame.clone();
        }

        string info = "Frame: " + to_string(frameCounter) + " | Update: " +
            to_string(SKELETON_DRAW_INTERVAL) + " | Thresh: " + to_string(thresh);
        putText(displayFrame, info, Point(5, 20),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);

        imshow("Body Skeleton", displayFrame);

        int key = waitKey(1);
        if (key == 27) break;
        if (key == 'r') {
            frameCounter = 0;
            cout << "Frame counter reset!" << endl;
        }
        if (key == 't') {
            thresh = (thresh == 0.1f) ? 0.2f : 0.1f;
            cout << "Threshold: " << thresh << endl;
        }
    }

    cap.release();
    destroyAllWindows();
}