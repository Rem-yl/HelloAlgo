#include <iostream>
#include <deque>
#include <vector>

using namespace std;

class Heap
{
public:
    deque<int> data{};
    Heap() = default;
    Heap(vector<int> vec)
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

    int peek()
    {
        return data.front();
    }

    bool empty()
    {
        return data.size() > 0 ? false : true;
    }

    // 从节点i开始进行从底到顶的堆化
    void siftUp(int i)
    {
        while (true)
        {
            int p = parent(i);
            if (p < 0 || data[i] <= data[p])
            {
                break;
            }
            swap(data[i], data[p]);
            i = p;
        }
    }

    void siftDown(int i)
    {
        while (true)
        {
            int left_i = left(i);
            int right_i = right(i);
            int curr_i = i;

            if (left_i < data.size() && data[left_i] > data[curr_i])
            {
                curr_i = left_i;
            }

            if (right_i < data.size() && data[right_i] > data[curr_i])
            {
                curr_i = right_i;
            }

            if (curr_i == i)
            {
                break;
            }

            swap(data[i], data[curr_i]);
            i = curr_i;
        }
    }

    void push_back(int val)
    {
        data.push_back(val);

        // 由底至顶进行堆化
        siftUp(data.size() - 1);
    }

    void pop()
    {
        swap(data[0], data[data.size() - 1]);
        data.pop_back();
        siftDown(0);
    }
};

void test()
{
    Heap h;
    vector<int> data = {2, 4, 1, 3, 2, 6};
    for (const int v : data)
    {
        h.push_back(v);
    }

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

    Heap h2(data);
    res.clear();
    while (!h2.empty())
    {
        res.push_back(h2.peek());
        h2.pop();
    }

    for (const int &v : res)
    {
        cout << v << " ";
    }

    cout << endl;
}

int main()
{
    test();
    return 0;
}
