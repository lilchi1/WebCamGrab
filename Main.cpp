#define _CRT_SECURE_NO_WARNINGS

#include "headers.h"

using namespace std;
int main() {
    // переподключить бибдиотеку, была в ../WebCamGrab/openCV ---> /qwe/openCV
    CAM1();
    return 0;
    system("cls");
    std::cout << "1 or 2" << endl;
    int chouse = 0;
    cin >> chouse;
    if (chouse == 1) _cam();
    else if (chouse == 2) body();
    else system("exit");
    return 0;
}