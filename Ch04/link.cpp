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
    size_t size() const;
    void insert(int idx, int num);
    void del(int idx);
    ListNode* get(int idx);
    int find(int val);
};

vector<int> ListNode::data() const
{
    vector<int> vec;
    const ListNode *curr = this;
    while (curr)
    {
        vec.push_back(curr->val);
        curr = curr->next;
    }

    return vec;
}

size_t ListNode::size() const
{
    size_t size = 0;

    const ListNode *curr = this;
    while (curr)
    {
        size++;
        curr = curr->next;
    }

    return size;
}

void ListNode::insert(int idx, int num)
{
    size_t size = this->size();
    if (idx >= size)
    {
        cerr << "索引越界" << endl;
        return;
    }

    ListNode *newNode = new ListNode(num);
    ListNode *curr = this;
    size_t cnt = 0;

    while (cnt < idx)
    {
        curr = curr->next;
        cnt++;
    }

    if (idx == size - 1)
    {
        curr->next = newNode;
    }
    else
    {
        ListNode *tmp = curr->next;
        curr->next = newNode;
        newNode->next = tmp;
    }
}

void ListNode::del(int idx)
{
    size_t size = this->size();
    if(idx >= size)
    {
        cerr << "索引越界" << endl;
        return;
    }

    size_t cnt = 0;
    ListNode *curr = this;
    while(cnt < idx -1)
    {
        curr = curr->next;
        cnt++;
    }

    if(idx == size-1)
    {
        curr->next = nullptr;
    }
    else
    {
        ListNode *tmp = curr->next;
        curr->next = tmp->next;
    }
}

ListNode* ListNode::get(int idx)
{
    size_t size = this->size();

    if(idx >= size)
    {
        cerr << "索引越界" << endl;
        return nullptr;
    }

    ListNode *curr = this;

    for(size_t i = 0;i<idx;++i)
    {
        curr = curr->next;
    }

    return curr;
}

int ListNode::find(int val)
{
    ListNode *curr = this;
    if(curr == nullptr)
    {
        return -1;
    }

    int cnt = 0;
    while(curr != nullptr)
    {
        if(curr->val == val)
        {
            return cnt;
        }
        curr = curr->next;
        cnt++;
    }

    return -1;
}

ListNode *NewLinkList(const vector<int> &vals)
{
    if (vals.empty())
        return nullptr;

    ListNode *head = new ListNode(vals[0]);
    ListNode *curr = head;

    for (int i = 1; i < vals.size(); ++i)
    {
        ListNode *newNode = new ListNode(vals[i]);
        curr->next = newNode;
        curr = curr->next;
    }

    return head;
}

int main()
{
    vector<int> vec = {1, 2, 3, 4, 5};
    ListNode *list = NewLinkList(vec);
    print_vec(list->data());

    cout << list->size() << endl;
    list->insert(2, 100);
    print_vec(list->data());
    list->del(3);
    print_vec(list->data());
    ListNode *res = list->get(2);
    cout << res->val << endl;
    cout << list->find(10) << endl;
}
