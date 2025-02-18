#include <iostream>
#include <string>

using namespace std;

int addHash(string key)
{
    long long hash = 0;
    const int MOD = 1000000007;
    for (unsigned char c : key)
    {
        hash = (hash + int(c)) % MOD;
    }

    return int(hash);
}

int mulHash(string key)
{
    long long hash = 0;
    const int MOD = 1000000007;

    for (unsigned char c : key)
    {
        hash = (31 * hash + int(c)) % MOD;
    }

    return int(hash);
}

int xorHash(string key)
{
    long long hash = 0;
    const int MOD = 1000000007;
    for (unsigned char c : key)
    {
        hash ^= int(c);
    }

    return int(hash);
}

int rotHash(string key)
{
    long long hash = 0;
    const int MOD = 1000000007;
    for (unsigned char c : key)
    {
        hash = ((hash << 4) ^ (hash >> 28) ^ int(c)) % MOD;
    }

    return int(hash);
}

int main()
{
    string key = "rem";
    cout << addHash(key) << endl;
    cout << addHash("mer") << endl;
    cout << mulHash(key)  << endl;
    cout << xorHash(key) << endl;
    cout << rotHash(key) << endl;
    return 0;
}