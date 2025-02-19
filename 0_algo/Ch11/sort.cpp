#include <iostream>
#include <vector>

using namespace std;

void selectSort(vector<int> &vec)
{
    if (vec.size() <= 1)
    {
        return;
    }

    for (int i = 1; i < vec.size(); ++i)
    {
        int base = vec[i], j = i - 1;
        while (j >= 0 && vec[j] > base)
        {
            vec[j + 1] = vec[j];
            j--;
        }

        vec[j + 1] = base;
    }
}

int partition(vector<int> &vec, int left, int right)
{
    int i = left, j = right;
    while (i < j)
    {
        while (i < j && vec[j] >= vec[left]) // 先从右往左找
        {
            j--;
        }

        while (i < j && vec[i] <= vec[left])
        {
            i++;
        }

        swap(vec[i], vec[j]);
    }

    swap(vec[i], vec[left]);

    return i;
}

void quickSort(vector<int> &vec, int left, int right)
{
    if (left >= right)
    {
        return;
    }

    int i = partition(vec, left, right);
    quickSort(vec, left, i - 1);
    quickSort(vec, i + 1, right);
}

void countSort(vector<int> &vec)
{
    int max = vec[0];
    for (int i = 1; i < vec.size(); ++i)
    {
        if (vec[i] > max)
        {
            max = vec[i];
        }
    }

    vector<int> bucket(max + 1, 0);
    for (int i = 0; i < vec.size(); ++i)
    {
        bucket[vec[i]]++; // 对数字计数
    }

    for (int i = 0; i < bucket.size() - 1; ++i)
    {
        bucket[i + 1] += bucket[i]; // 计算前缀和
    }

    vector<int> res(vec.size(), 0);
    for (int i = vec.size() - 1; i >= 0; --i)
    {
        int tmp = bucket[vec[i]] - 1;
        bucket[vec[i]]--;
        res[tmp] = vec[i];
    }

    vec = res;
}

void test()
{
    vector<int> vec = {2, 4, 1, 0, 3, 5};
    // quickSort(vec, 0, vec.size() - 1);
    countSort(vec);

    for (const int &v : vec)
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