#include <iostream>

using namespace std;

void print_arr(int arr[], int size)
{
    for (int i = 0; i < size; ++i)
    {
        cout << *arr++ << " ";
    }
    cout << endl;
}

// 随机访问
int random_access(int arr[], int size)
{
    int i = rand() % size;

    return arr[i];
}

// 插入元素
void insert(int arr[], int size, int idx, int num)
{
    for (int i = size - 1; i > idx; --i)
    {
        arr[i] = arr[i-1];
    }

    arr[idx] = num;
}

int main()
{
    int arr[5];
    int nums[5] = {1, 2, 3, 4, 5};

    int num = random_access(nums, 4);
    cout << num << endl;
    print_arr(nums, 5);

    insert(nums, 5, 2, 100);
    print_arr(nums, 5);
}