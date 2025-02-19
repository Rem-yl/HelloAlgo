#include <iostream>
#include <queue>
#include <vector>

using namespace std;

class Node
{
public:
    int val;
    Node *left;
    Node *right;

    Node(int val) : val(val), left(nullptr), right(nullptr) {};
};

class BinarySearchTree
{
public:
    Node *root;
    BinarySearchTree() : root(nullptr) {};
    BinarySearchTree(Node *root) : root(root) {};

    vector<int> inOrder()
    {
        vector<int> res;
        dfs(root, "in", res);
        return res;
    }

    Node *find(int val)
    {
        Node *curr = root;

        while (curr != nullptr)
        {
            if (val > curr->val)
            {
                curr = curr->right;
            }
            else if (val < curr->val)
            {
                curr = curr->left;
            }
            else
            {
                break;
            }
        }

        return curr;
    }

    void del(int val)
    {
        if (root == nullptr)
        {
            return;
        }

        Node *curr = root, *pre = nullptr;
        while (curr != nullptr)
        {
            if (curr->val == val)
            {
                break;
            }
            pre = curr;
            if (val > curr->val)
            {
                curr = curr->right;
            }
            else
            {
                curr = curr->left;
            }
        }

        if (curr == nullptr) // 没有查找到待删除节点
        {
            return;
        }

        if (curr->left == nullptr || curr->right == nullptr) // 子节点数量为0或者1
        {
            Node *child = curr->left != nullptr ? curr->left : curr->right;
            if (curr != root)
            {
                if (pre->left == curr)
                {
                    pre->left = child;
                }
                else
                {
                    pre->right = child;
                }
            }
            else
            {
                root = child;
            }

            delete curr;
        }
        else
        {
            Node *tmp = curr->right;
            while (tmp->left != nullptr)    //遍历到最后, tmp将没有left子树
            {
                tmp = tmp->left;
            }
            int tmpVal = tmp->val;
            del(tmp->val);      // tmp最多只会有一个右子树, 节点数为0或者1
            curr->val = tmpVal;
        }
    }

private:
    void dfs(Node *root, string order, vector<int> &res)
    {
        if (root == nullptr)
        {
            return;
        }

        if (order == "pre")
        {
            res.push_back(root->val);
        }
        dfs(root->left, order, res);

        if (order == "in")
        {
            res.push_back(root->val);
        }
        dfs(root->right, order, res);

        if (order == "post")
        {
            res.push_back(root->val);
        }
    }
};

BinarySearchTree *BuildBinarySearchTree(const vector<int> &vec)
{
    if (vec.empty())
    {
        return new BinarySearchTree();
    }

    Node *head = new Node(vec[0]);
    BinarySearchTree *t = new BinarySearchTree(head);

    if (vec.size() == 1)
    {
        return t;
    }

    for (int i = 1; i < vec.size(); ++i)
    {
        Node *newNode = new Node(vec[i]);
        Node *curr = head;
        Node *prev = head;

        while (curr != nullptr)
        {
            if (vec[i] > curr->val)
            {
                prev = curr;
                curr = curr->right;
            }
            else if (vec[i] < curr->val)
            {
                prev = curr;
                curr = curr->left;
            }
            else
            {
                break;
            }
        }

        if (vec[i] > prev->val)
        {
            prev->right = newNode;
        }
        else if (vec[i] < prev->val)
        {
            prev->left = newNode;
        }
        else
        {
            continue;
        }
    }

    return t;
}

void test()
{
    vector<int> vec = {2, 3, 1, 5, 5, 4, 7, 6, 7, 8, 9, 4, 5, 6, 10, 23, 34};
    BinarySearchTree *tree = BuildBinarySearchTree(vec);
    tree->del(2);
    vector<int> res = tree->inOrder();

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