#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class MinHeap
{
public:
    vector<int> data{};
    MinHeap() = default;
    MinHeap(vector<int> vec)
    {
        for (const int v : vec)
        {
            data.push_back(v);
        }

        for (int i = parent(data.size() - 1); i >= 0; i--)
        {
            siftDown(i);
        }
    }

    int left(int i)
    {
        return 2 * i + 1;
    }

    int right(int i)
    {
        return 2 * i + 2;
    }

    int parent(int i)
    {
        return (i - 1) / 2;
    }

    bool empty()
    {
        return data.size() > 0 ? false : true;
    }

    void siftUp(int i)
    {
        while (true)
        {
            int p = parent(i);
            if (p < 0 || data[i] >= data[p])
            {
                break;
            }
            swap(data[i], data[p]);
            i = p;
        }
    }

    void push_back(int val)
    {
        data.push_back(val);
        siftUp(data.size() - 1);
    }

    int peek()
    {
        return data.front();
    }

    void siftDown(int i)
    {
        while (true)
        {
            int l = left(i), r = right(i), curr = i;
            if (l < data.size() && data[l] < data[curr])
            {
                curr = l;
            }

            if (r < data.size() && data[r] < data[curr])
            {
                curr = r;
            }

            if (curr == i)
            {
                break;
            }

            swap(data[i], data[curr]);
            i = curr;
        }
    }

    void pop()
    {
        if (data.size() == 0)
        {
            throw("out of range.");
        }

        swap(data[0], data[data.size() - 1]);
        data.pop_back();
        siftDown(0);
    }
};

void test()
{
    vector<int> data = {2, 5, 3, 1, 2, 4};
    MinHeap h(data);

    vector<int> res;
    while (!h.empty())
    {
        res.push_back(h.peek());
        h.pop();
    }

    for (const int &v : res)
    {
        cout << v << " ";
    }

    cout << endl;
}

vector<int> topk(vector<int> vec, int k)
{
    vector<int> res;
    if (k > vec.size())
    {
        MinHeap h(vec);
        while (!h.empty())
        {
            res.push_back(h.peek());
            h.pop();
        }

        reverse(res.begin(), res.end());
        return res;
    }

    vector<int> tmp(vec.begin(), vec.begin() + k);
    MinHeap h(tmp);
    for (int i = k + 1; i < vec.size(); ++i)
    {
        if (vec[i] > h.peek())
        {
            h.pop();
            h.push_back(vec[i]);
        }
    }

    while (!h.empty())
    {
        res.push_back(h.peek());
        h.pop();
    }

    reverse(res.begin(), res.end());
    return res;
}

void test1()
{
    vector<int> vec = {1, 2, 3, 4, 2, 32, 43, 24, 35, 4, 324, 25, 23};
    vector<int> res = topk(vec, 3);

    for (const int &v : res)
    {
        cout << v << " ";
    }

    cout << endl;
}

int main()
{
    test1();
    return 0;
}