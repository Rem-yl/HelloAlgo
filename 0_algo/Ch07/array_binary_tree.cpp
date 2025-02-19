#include <iostream>
#include <vector>
#include <queue>

using namespace std;

class ArrayBinaryTree
{
public:
    ArrayBinaryTree(vector<int> arr) : tree(arr) {};
    int size()
    {
        return tree.size();
    }

    int val(int i)
    {
        if (i < 0 || i >= size())
        {
            return INT_MAX;
        }

        return tree[i];
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

    vector<int> levelOrder()
    {
        vector<int> res;
        for (int i = 0; i < size(); i++)
        {
            if (val(i) != INT_MAX)
            {
                res.push_back(val(i));
            }
        }

        return res;
    }

    vector<int> preOrder()
    {
        vector<int> res;
        dfs(0, "pre", res);

        return res;
    }

    vector<int> inOrder()
    {
        vector<int> res;
        dfs(0, "in", res);

        return res;
    }

    vector<int> postOrder()
    {
        vector<int> res;
        dfs(0, "post", res);

        return res;
    }

private:
    vector<int> tree;

    void dfs(int i, string order, vector<int> &res)
    {
        if (val(i) == INT_MAX)
        {
            return;
        }
        if (order == "pre")
        {
            res.push_back(val(i));
        }
        dfs(left(i), order, res);

        if (order == "in")
        {
            res.push_back(val(i));
        }
        dfs(right(i), order, res);

        if (order == "post")
        {
            res.push_back(val(i));
        }
    }
};