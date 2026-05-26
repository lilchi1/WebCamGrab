#include "func.h"
#include <thread>
#include <chrono>
#include <fstream>

void simpleCamera() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        imshow("Camera", frame);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

void motionDetection() {
    VideoCapture cap(0);
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

        imshow("Motion Detection", diff);

        if (waitKey(1) == 27) break;
        prevFrame = gray.clone();
    }
}

void colorTracking() {
    VideoCapture cap(0);
    Mat frame, hsv, mask, result;

    // Диапазон зеленого цвета в HSV
    Scalar lowerGreen(35, 100, 100);
    Scalar upperGreen(85, 255, 255);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, lowerGreen, upperGreen, mask);

        // Поиск контуров
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        frame.copyTo(result);
        for (auto& contour : contours) {
            if (contourArea(contour) > 500) {
                Rect rect = boundingRect(contour);
                rectangle(result, rect, Scalar(0, 0, 255), 2);
            }
        }

        imshow("Color Tracking", result);
        if (waitKey(1) == 27) break;
    }
}

void faceDetection() {
    // Укажите полный путь к файлу
    CascadeClassifier faceCascade;
    string cascadePath = "openCV\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml";

    if (!faceCascade.load(cascadePath)) {
        cerr << "Error loading cascade file: " << cascadePath << endl;
        cerr << "Check if file exists!" << endl;
        return;
    }

    VideoCapture cap(0);  // Исправлено: cap, а не can
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

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
        }

        imshow("Face Detection", frame);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

void edgeDetection() {
    VideoCapture cap(0);
    Mat frame, edges;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(5, 5), 1.5);
        Canny(edges, edges, 50, 150);

        imshow("Edge Detection", edges);
        if (waitKey(1) == 27) break;
    }
}

void CAM1() {
    cout << "Select mode:\n";
    cout << "1 - Simple Camera\n";
    cout << "2 - Motion Detection\n";
    cout << "3 - Color Tracking\n";
    cout << "4 - Edge Detection\n";
    cout << "5 - Face Detection\n";

    int choice;
    cin >> choice;

    switch (choice) {
    case 1: simpleCamera(); break;
    case 2: motionDetection(); break;
    case 3: colorTracking(); break;
    case 4: edgeDetection(); break;
    case 5: faceDetection(); break;
    default: simpleCamera(); break;
    }

}
void _cam() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    cap.set(CAP_PROP_BUFFERSIZE, 1);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(CAP_PROP_FPS, 30);

    Mat frame;
    int frameCounter = 0;
    const int SKELETON_DRAW_INTERVAL = 15;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        frameCounter++;

        Mat displayFrame = frame.clone();

        string fpsInfo = "Frame: " + to_string(frameCounter) + " | Update every " +
            to_string(SKELETON_DRAW_INTERVAL) + " frames";
        putText(displayFrame, fpsInfo, Point(10, 60),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        imshow("Camera", displayFrame);

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}
/// <summary>
/// ии версия трекинга скелета
/// </summary>
void body() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "CUDA enabled: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "CUDA device found!" << std::endl;
        cv::cuda::printCudaDeviceInfo(0);
    }
    else {
        std::cout << "CUDA NOT available - need to rebuild OpenCV with CUDA" << std::endl;
    }
    // ОПТИМИЗАЦИЯ 1: Уменьшенное разрешение для входа (быстрее)
    int inWidth = 256;   // Было 368
    int inHeight = 256;  // Было 368
    float thresh = 0.1f;

    // ОПТИМИЗАЦИЯ 2: Увеличен интервал обработки скелета
    const int SKELETON_DRAW_INTERVAL = 10;  // Увеличил нагрузку? Нет, наоборот

    // ОПТИМИЗАЦИЯ 3: Используем более легкий бэкенд для нейросети
    string prototxt = "pose_deploy_linevec_faster_4_stages.prototxt";
    string caffemodel = "pose_iter_160000.caffemodel";

    Net net = readNetFromCaffe(prototxt, caffemodel);
    if (net.empty()) {
        cerr << "Cannot load model!" << endl;
        return;
    }

    // ОПТИМИЗАЦИЯ 4: Используем CPU с максимальной производительностью
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // ОПТИМИЗАЦИЯ 5: Уменьшаем разрешение входного видео
    int frameWidth = 320;   // Было 640
    int frameHeight = 240;  // Было 480

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return;
    }

    // Настройки для максимальной скорости
    cap.set(CAP_PROP_BUFFERSIZE, 1);
    cap.set(CAP_PROP_FRAME_WIDTH, frameWidth);
    cap.set(CAP_PROP_FRAME_HEIGHT, frameHeight);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(CAP_PROP_FPS, 60);  // Пытаемся получить максимальный FPS от камеры

    cout << "Body tracker started. Press ESC to exit." << endl;
    cout << "Optimized mode: " << frameWidth << "x" << frameHeight << endl;
    cout << "Skeleton updates every " << SKELETON_DRAW_INTERVAL << "th frame" << endl;

    Mat frame;
    int frameCounter = 0;
    Mat lastSkeletonFrame;
    bool hasSkeleton = false;

    // ОПТИМИЗАЦИЯ 6: Предварительное выделение памяти для точек
    vector<Point> points(16);

    // ОПТИМИЗАЦИЯ 7: Для измерения реального FPS
    int fpsCounter = 0;
    double fpsTimer = (double)getTickCount();
    double currentFPS = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        frameCounter++;
        fpsCounter++;

        // ОПТИМИЗАЦИЯ 8: Показываем FPS каждую секунду
        double now = (double)getTickCount();
        if (now - fpsTimer >= getTickFrequency()) {
            currentFPS = fpsCounter * getTickFrequency() / (now - fpsTimer);
            fpsCounter = 0;
            fpsTimer = now;
            cout << "Current FPS: " << (int)currentFPS << endl;
        }

        Mat displayFrame;

        // Обработка скелета на каждом N-м кадре
        if (frameCounter % SKELETON_DRAW_INTERVAL == 0) {
            displayFrame = frame.clone();

            // ОПТИМИЗАЦИЯ 9: Используем более быстрое масштабирование
            Mat inputBlob = blobFromImage(frame, 1.0 / 255.0,
                Size(inWidth, inHeight),
                Scalar(0, 0, 0), true, false);  // swapRB=true для правильных цветов

            net.setInput(inputBlob);
            Mat output = net.forward();

            int H = output.size[2];
            int W = output.size[3];

            // Сброс точек
            for (int i = 0; i < 16; i++) {
                points[i] = Point(-1, -1);
            }

            // ОПТИМИЗАЦИЯ 10: Объединенный цикл обработки точек
            for (int i = 0; i < 16; i++) {
                Mat heatMap(H, W, CV_32F, output.ptr<float>(0, i));
                double conf;
                Point maxLoc;
                minMaxLoc(heatMap, nullptr, &conf, nullptr, &maxLoc);

                if (conf > thresh) {
                    points[i] = Point(
                        (int)(maxLoc.x * frame.cols / W),
                        (int)(maxLoc.y * frame.rows / H)
                    );
                }
            }

            // Рисуем соединения
            for (int i = 0; i < 17; i++) {
                int from = POSE_PAIRS_MPI[i][0];
                int to = POSE_PAIRS_MPI[i][1];

                if (from < 16 && to < 16 &&
                    points[from].x > 0 && points[to].x > 0) {
                    line(displayFrame, points[from], points[to], Scalar(0, 255, 0), 2);
                }
            }

            // Рисуем точки
            for (int i = 0; i < 16; i++) {
                if (points[i].x > 0) {
                    circle(displayFrame, points[i], 3, Scalar(0, 0, 255), -1);
                }
            }

            lastSkeletonFrame = displayFrame.clone();
            hasSkeleton = true;
        }
        else {
            // Быстрое копирование без лишних операций
            if (hasSkeleton) {
                displayFrame = lastSkeletonFrame.clone();
            }
            else {
                displayFrame = frame.clone();
            }
        }

        // Информация на экране (минимально)
        string info = "FPS: " + to_string((int)currentFPS) +
            " | Frame: " + to_string(frameCounter) +
            " | Update: " + to_string(SKELETON_DRAW_INTERVAL);
        putText(displayFrame, info, Point(5, 20),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);

        imshow("Body Skeleton", displayFrame);

        int key = waitKey(1);
        if (key == 27) break;
        if (key == 'r') {
            frameCounter = 0;
            cout << "Frame counter reset!" << endl;
        }
    }

    cap.release();
    destroyAllWindows();
}