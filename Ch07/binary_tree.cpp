#include <iostream>
#include <string>
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

class Tree
{
public:
    Node *root;
    Tree(Node *root) : root(root) {};
};

Tree *BuildTree(queue<int> vec)
{
    if (vec.empty())
    {
        return nullptr;
    }

    Node *newNode = new Node(vec.front());
    vec.pop();

    Tree *head = new Tree(newNode);
    Node *curr = head->root;

    while (!vec.empty())
    {
        newNode = new Node(vec.front());
        vec.pop();
        if (curr->left == nullptr)
        {
            curr->left = newNode;
        }
        else if (curr->right == nullptr)
        {
            curr->right = newNode;
            curr = curr->left;
        }
        else
        {
            curr = curr->left;
            curr->left = newNode;
        }
    }

    return head;
}

vector<int> GetTreeArray(Tree *t)
{
    vector<int> res;
    if (t->root == nullptr)
    {
        return res;
    }

    queue<Node *> node;
    node.push(t->root);
    res.push_back(t->root->val);

    while (!node.empty())
    {
        Node *tmp = node.front();
        node.pop();
        if (tmp->left != nullptr)
        {
            node.push(tmp->left);
            res.push_back(tmp->left->val);
            if (tmp->right != nullptr)
            {
                node.push(tmp->right);
                res.push_back(tmp->right->val);
            }
        }
    }

    return res;
}

void test()
{
    queue<int> vec;
    vec.push(1);
    vec.push(2);
    vec.push(3);
    vec.push(4);
    vec.push(5);
    vec.push(6);

    Tree *t = BuildTree(vec);
    vector<int> res = GetTreeArray(t);

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