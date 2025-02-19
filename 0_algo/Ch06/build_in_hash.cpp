#include <iostream>
#include <string>
#include <functional>

using namespace std;

void test()
{
    int num = 10;
    size_t hashNum = hash<int>()(num);
    cout << hashNum << endl;

    double num1 = 12.12;
    size_t hashNum1 = hash<double>()(num1);
    cout << hashNum1 << endl;
}

int main()
{

}