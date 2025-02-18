#include <iostream>
#include <vector>
#include <string>

using namespace std;

class TreeNode
{
public:
    int val{};
    int height = 0;
    TreeNode *left{};
    TreeNode *right{};
    TreeNode() = default;
    explicit TreeNode(int x) : val(x) {};
};

int height(TreeNode *node)
{
    return node == nullptr ? -1 : node->height;
}

void updateHeight(TreeNode *node)
{
    node->height = max(height(node->left), height(node->right)) + 1;
}

int balanceFactor(TreeNode *node)
{
    if (node == nullptr)
    {
        return 0;
    }

    // 节点平衡因子 = 左子树高度 - 右子树高度
    return height(node->left) - height(node->right);
}

TreeNode *rightRotate(TreeNode *node)
{
    TreeNode *child = node->left;
    TreeNode *grand_child = child->right;
    child->right = node;
    node->left = grand_child;
    updateHeight(node);
    updateHeight(child);
    return child;
}

TreeNode *leftRotate(TreeNode *node)
{
    TreeNode *child = node->right;
    TreeNode *grand_child = child->left;
    child->left = node;
    node->right = grand_child;
    updateHeight(node);
    updateHeight(child);
    return child;
}

TreeNode *rotate(TreeNode *node)
{
    int _balanceFactor = balanceFactor(node);
    if (_balanceFactor > 1) // 左偏树
    {
        if (balanceFactor(node->left) >= 0)
        {
            return rightRotate(node);
        }
        else
        {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }
    }
    else if (_balanceFactor < -1) // 右偏数
    {
        if (balanceFactor(node->right) <= 0)
        {
            return leftRotate(node);
        }
        else
        {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }
    }
    else
    {
        return node;
    }
}

void insert(int val)
{
    TreeNode *root = insertHelper(root, val);
}

TreeNode *insertHelper(TreeNode *node, int val)
{
    if (node == nullptr)
    {
        return new TreeNode(val);
    }
    if (val > node->val)
    {
        node->left = insertHelper(node->left, val);
    }
    else if (val < node->val)
    {
        node->right = insertHelper(node->right, val);
    }
    else
    {
        return node;
    }
    updateHeight(node);
    node = rotate(node);
    return node;
}

void remove(int val)
{
    TreeNode *root = removeHelper(root, val);
}

TreeNode *removeHelper(TreeNode *node, int val)
{
    if (node == nullptr)
    {
        return nullptr;
    }
    if (val > node->val)
    {
        node->right = removeHelper(node->right, val);
    }
    else if (val < node->val)
    {
        node->left = removeHelper(node->left, val);
    }
    else
    {
        if (node->left == nullptr || node->right == nullptr) // 叶子节点数量为0或1
        {
            TreeNode *child = node->left != nullptr ? node->left : node->right;
            if (child == nullptr)
            {
                delete node;
                return nullptr;
            }
            else
            {
                delete node;
                node = child;
                return node;
            }
        }
        else    // 叶子节点数量为2
        {
            // 将node的右子树的最小节点删除, 并替代当前的node
            TreeNode *tmp = node->right;
            while (tmp->left != nullptr)
            {
                tmp = tmp->left;
            }

            int tmpVal = tmp->val;
            node->right = removeHelper(node->right, tmpVal);
            node->val = tmpVal;     // 将当前的node值替换为tmpVal
        }
    }

    updateHeight(node);
    rotate(node);

    return node;
}