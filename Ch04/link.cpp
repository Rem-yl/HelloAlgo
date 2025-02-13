#include <iostream>
#include <vector>

using namespace std;

void print_vec(vector<int> vec)
{
    for (const auto &v : vec)
    {
        cout << v << " ";
    }

    cout << endl;
}

class ListNode
{
public:
    int val;
    ListNode *next;
    ListNode(int val) : val(val), next(nullptr) {};
    vector<int> data() const;
};

vector<int> ListNode::data() const
{
    vector<int> vec;
    const ListNode *curr = this;
    while (curr)
    {
        vec.push_back(this->val);
        curr = curr->next;
    }

    return vec;
}

ListNode *NewLinkList(const vector<int> &vals)
{
    if (vals.empty())
        return nullptr;
    shared_ptr<ListNode> head = make_shared<ListNode>(ListNode(vals[0]));
    shared_ptr<ListNode> curr = head;

    for (int i = 1; i < vals.size(); ++i)
    {
        shared_ptr<ListNode> newNode = make_shared<ListNode>(ListNode(vals[i]));
        curr->next = newNode.get();
        curr = newNode;
    }

    return head.get();
}

int main()
{
    vector<int> vec = {1, 2, 3, 4, 5};
    ListNode *list = NewLinkList(vec);

    print_vec(list->data());
}
