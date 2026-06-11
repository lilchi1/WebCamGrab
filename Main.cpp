#define _CRT_SECURE_NO_WARNINGS

#include <cstdlib>
#include "headers.h"
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <ctime>

using namespace std;

const int DISP_W = 960;
const int DISP_H = 540;

static CascadeClassifier g_faceCascade;
static bool g_faceLoaded = false;
static atomic<bool> g_running(true);

// --- Логирование в файл ---
static mutex g_logMtx;
static ofstream g_logFile;

static void logWrite(const string& level, const string& ip, const string& msg) {
    lock_guard<mutex> lock(g_logMtx);
    if (!g_logFile.is_open()) return;
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&t));
    g_logFile << "[" << level << "][" << buf << "][" << ip << "] " << msg << endl;
}

// --- Статусы камер для вывода в консоль ---
struct CamInfo {
    string ip;
    int mode = 1;
    double fps = 0;
    double scale = 1.0;
    bool connected = false;
};
static vector<CamInfo> g_cams;
static mutex g_camMtx;

const char* modeName(int m) {
    switch (m) {
        case 1: return "Simple";
        case 2: return "Motion Detection";
        case 3: return "Edge Detection";
        case 4: return "Face Detection";
        case 5: return "Color Tracking";
        case 6: return "Color Tracking (Multi)";
        case 7: return "Face Blur";
        default: return "Unknown";
    }
}

static void printAllStatusLocked() {
    system("cls");
    cout << "===========================================" << endl;
    cout << "            CAMERA STATUSES" << endl;
    cout << "===========================================" << endl;
    for (auto& c : g_cams) {
        cout << "  Camera:  " << c.ip << "  [" << (c.connected ? "CONNECTED" : "DISCONNECTED") << "]" << endl;
        cout << "  Mode:    " << modeName(c.mode) << " (" << c.mode << ")" << endl;
        cout << "  FPS:     " << (int)c.fps << endl;
        cout << "  Scale:   " << c.scale << endl;
        cout << "-------------------------------------------" << endl;
    }
    cout << "Controls: 1-7 switch mode, ESC exit all" << endl;
    cout << "===========================================" << endl;
}

static void printAllStatus() {
    lock_guard<mutex> lock(g_camMtx);
    printAllStatusLocked();
}

struct AdaptiveQuality {
    double scale = 1.0;
    int frameCount = 0;
    double lastTime = 0;
    double fps = 0;
    void update() {
        frameCount++;
        double now = (double)getTickCount();
        double elapsed = (now - lastTime) / getTickFrequency();
        if (elapsed >= 1.0) {
            fps = frameCount / elapsed;
            frameCount = 0;
            lastTime = now;
            if (fps < 15) scale = max(0.5, scale - 0.1);
            else if (fps > 25) scale = min(1.0, scale + 0.1);
        }
    }
};

struct ColorRange {
    Scalar lower, upper;
    Scalar color;
    string name;
};

static vector<ColorRange> g_colors = {
    {Scalar(35, 100, 100), Scalar(85, 255, 255), Scalar(0, 255, 0), "GREEN"},
    {Scalar(0, 100, 100), Scalar(10, 255, 255), Scalar(0, 0, 255), "RED"},
    {Scalar(100, 100, 100), Scalar(140, 255, 255), Scalar(255, 0, 0), "BLUE"},
};
static Scalar g_redLower2(160, 100, 100), g_redUpper2(179, 255, 255);

void processFrame(Mat& frame, int mode, AdaptiveQuality& aq, Mat& prevGray) {
    int ow = frame.cols, oh = frame.rows;
    Mat proc = frame;
    int pw = ow, ph = oh;

    if (aq.scale < 1.0) {
        pw = max(1, (int)(ow * aq.scale));
        ph = max(1, (int)(oh * aq.scale));
        resize(frame, proc, Size(pw, ph));
    }

    aq.update();
    double sf = 1.0 / aq.scale;

    if (mode == 2) {
        Mat gray, diff;
        cvtColor(proc, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(21, 21), 0);
        if (!prevGray.empty() && prevGray.size() == gray.size()) {
            absdiff(prevGray, gray, diff);
            threshold(diff, diff, 25, 255, THRESH_BINARY);
            int motion = countNonZero(diff);
            if (pw != ow) resize(diff, diff, Size(ow, oh));
            cvtColor(diff, frame, COLOR_GRAY2BGR);
            putText(frame, "Motion: " + to_string(motion) + "px", Point(10, 60),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }
        prevGray = gray.clone();
    }
    else if (mode == 3) {
        Mat edges;
        cvtColor(proc, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(5, 5), 1.5);
        Canny(edges, edges, 50, 150);
        if (pw != ow) resize(edges, edges, Size(ow, oh));
        cvtColor(edges, frame, COLOR_GRAY2BGR);
    }
    else if (mode == 4) {
        if (g_faceLoaded) {
            Mat gray;
            cvtColor(proc, gray, COLOR_BGR2GRAY);
            equalizeHist(gray, gray);
            vector<Rect> faces;
            g_faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
            for (auto& f : faces) {
                f.x = (int)(f.x * sf); f.y = (int)(f.y * sf);
                f.width = (int)(f.width * sf); f.height = (int)(f.height * sf);
                rectangle(frame, f, Scalar(0, 255, 0), 2);
            }
            putText(frame, "Faces: " + to_string(faces.size()), Point(10, 60),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }
    }
    else if (mode == 5) {
        Mat hsv, mask;
        cvtColor(proc, hsv, COLOR_BGR2HSV);
        inRange(hsv, g_colors[0].lower, g_colors[0].upper, mask);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (auto& c : contours) {
            if (contourArea(c) < 500) continue;
            Rect r = boundingRect(c);
            r.x = (int)(r.x * sf); r.y = (int)(r.y * sf);
            r.width = (int)(r.width * sf); r.height = (int)(r.height * sf);
            rectangle(frame, r, g_colors[0].color, 2);
            circle(frame, Point(r.x + r.width / 2, r.y + r.height / 2), 5, g_colors[0].color, -1);
        }
        putText(frame, "Tracking: " + g_colors[0].name, Point(10, 60),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
    }
    else if (mode == 6) {
        Mat hsv, mask;
        cvtColor(proc, hsv, COLOR_BGR2HSV);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        for (size_t ci = 0; ci < g_colors.size(); ci++) {
            if (ci == 1) {
                Mat m1, m2;
                inRange(hsv, g_colors[ci].lower, g_colors[ci].upper, m1);
                inRange(hsv, g_redLower2, g_redUpper2, m2);
                mask = m1 | m2;
            } else {
                inRange(hsv, g_colors[ci].lower, g_colors[ci].upper, mask);
            }
            morphologyEx(mask, mask, MORPH_OPEN, kernel);
            morphologyEx(mask, mask, MORPH_CLOSE, kernel);
            vector<vector<Point>> contours;
            findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            for (auto& c : contours) {
                if (contourArea(c) < 500) continue;
                Rect r = boundingRect(c);
                r.x = (int)(r.x * sf); r.y = (int)(r.y * sf);
                r.width = (int)(r.width * sf); r.height = (int)(r.height * sf);
                rectangle(frame, r, g_colors[ci].color, 2);
                Point center(r.x + r.width / 2, r.y + r.height / 2);
                circle(frame, center, 5, g_colors[ci].color, -1);
                putText(frame, g_colors[ci].name, Point(r.x, r.y - 5),
                        FONT_HERSHEY_SIMPLEX, 0.4, g_colors[ci].color, 1);
            }
        }
        putText(frame, "Multi-Color Tracking", Point(10, 60),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    }
    else if (mode == 7) {
        if (!g_faceLoaded) return;
        Mat gray;
        cvtColor(proc, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        vector<Rect> faces;
        g_faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

        if (!faces.empty()) {
            Mat blurred;
            GaussianBlur(proc, blurred, Size(51, 51), 0);

            for (auto& f : faces) {
                int pad = (int)(min(f.width, f.height) * 0.2);
                Rect r = f;
                r.x = max(0, r.x - pad);
                r.y = max(0, r.y - pad);
                r.width = min(proc.cols - r.x, r.width + 2 * pad);
                r.height = min(proc.rows - r.y, r.height + 2 * pad);
                proc(r).copyTo(blurred(r));
            }

            if (pw != ow) resize(blurred, blurred, Size(ow, oh));
            blurred.copyTo(frame);

            for (auto& f : faces) {
                Rect rf(
                    (int)(f.x * sf), (int)(f.y * sf),
                    (int)(f.width * sf), (int)(f.height * sf));
                rectangle(frame, rf, Scalar(0, 255, 0), 2);
            }
        }
        putText(frame, "Face Blur", Point(10, 60),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
    }
}

static void setCamConnected(int camIdx, bool connected, const string& ip) {
    lock_guard<mutex> lock(g_camMtx);
    if (g_cams[camIdx].connected == connected) return;
    g_cams[camIdx].connected = connected;
    logWrite(connected ? "INFO" : "WARN", ip, connected ? "Camera connected" : "Connection lost");
    printAllStatusLocked();
}

void cameraThread(string ip, int camIdx) {
    string url = "http://" + ip + "/video";
    string winName = "IP Camera: " + ip;
    namedWindow(winName, WINDOW_NORMAL);

    string ffpath = "ffmpeg";
    ifstream test("ffmpeg.exe");
    if (test.good()) ffpath = ".\\ffmpeg.exe";
    test.close();

    while (g_running) {
        string cmd = ffpath + " -i \"" + url +
                     "\" -f image2pipe -vcodec mjpeg -q 2 -an - 2>NUL";

        FILE* pipe = _popen(cmd.c_str(), "rb");
        if (!pipe) {
            logWrite("ERROR", ip, "Failed to start ffmpeg. Retry in 5s...");
            this_thread::sleep_for(chrono::seconds(5));
            continue;
        }

        int mode = 1;
        AdaptiveQuality aq;
        aq.lastTime = (double)getTickCount();
        Mat prevGray;
        vector<char> buf(768 * 1024);
        size_t pos = 0;
        bool hadFrame = false;

        while (g_running) {
            int n = fread(buf.data() + pos, 1, buf.size() - pos - 1, pipe);
            if (n <= 0) {
                if (hadFrame) setCamConnected(camIdx, false, ip);
                break;
            }
            pos += n;

            size_t start = SIZE_MAX;
            for (size_t i = 0; i + 1 < pos; i++) {
                if ((uchar)buf[i] == 0xFF && (uchar)buf[i + 1] == 0xD8) start = i;
                if (start != SIZE_MAX && (uchar)buf[i] == 0xFF && (uchar)buf[i + 1] == 0xD9) {
                    size_t end = i + 2;
                    Mat frame = imdecode(
                        Mat(1, (int)(end - start), CV_8U, buf.data() + start),
                        IMREAD_COLOR);
                    size_t remaining = pos - end;
                    memmove(buf.data(), buf.data() + end, remaining);
                    pos = remaining;

                    if (!frame.empty()) {
                        if (!hadFrame) {
                            hadFrame = true;
                            setCamConnected(camIdx, true, ip);
                        }
                        processFrame(frame, mode, aq, prevGray);
                        Mat display;
                        resize(frame, display, Size(DISP_W, DISP_H));
                        putText(display, ip + " | " + modeName(mode), Point(10, 30),
                                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                        putText(display, to_string((int)aq.fps) + " FPS",
                                Point(10, DISP_H - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
                        imshow(winName, display);
                    }
                    break;
                }
            }

            if (pos > buf.size() / 2) pos = 0;

            int key = waitKey(30);
            if (key == 27) { g_running = false; break; }
            if (key >= '1' && key <= '7') {
                mode = key - '0';
                prevGray = Mat();
                logWrite("INFO", ip, string("Mode: ") + modeName(mode));
                {
                    lock_guard<mutex> lock(g_camMtx);
                    g_cams[camIdx].mode = mode;
                    g_cams[camIdx].fps = aq.fps;
                    g_cams[camIdx].scale = aq.scale;
                }
                printAllStatus();
            }
        }

        _pclose(pipe);
        if (!g_running) break;

        logWrite("INFO", ip, "Reconnecting in 5s...");
        this_thread::sleep_for(chrono::seconds(5));
    }

    destroyWindow(winName);
}

int main() {
    _putenv_s("OPENCV_LOG_LEVEL", "ERROR");
    _putenv_s("OPENCV_FFMPEG_LOGLEVEL", "-8");

    g_logFile.open("camera_log.txt", ios::trunc);

    // Load face cascade
    vector<string> cascadePaths = {
        "openCV/build/etc/haarcascades/haarcascade_frontalface_default.xml",
        "cascades/haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_default.xml",
    };
    for (auto& p : cascadePaths) {
        if (g_faceCascade.load(p)) {
            g_faceLoaded = true;
            break;
        }
    }

    cout << "Enter camera IPs (comma-separated, e.g. 192.168.1.244:8080,192.168.1.245:8080): ";
    string input;
    getline(cin, input);

    vector<string> ips;
    stringstream ss(input);
    string item;
    while (getline(ss, item, ',')) {
        item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) ips.push_back(item);
    }

    if (ips.empty()) {
        cerr << "No IPs entered." << endl;
        return 1;
    }

    g_cams.resize(ips.size());
    for (size_t i = 0; i < ips.size(); i++) {
        g_cams[i].ip = ips[i];
    }
    printAllStatus();

    vector<thread> threads;
    for (size_t i = 0; i < ips.size(); i++) {
        threads.emplace_back(cameraThread, ips[i], (int)i);
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    return 0;
}
