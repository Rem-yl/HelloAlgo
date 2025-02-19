#include <iostream>
#include <vector>

using namespace std;

class GraphAdjMat
{
public:
    vector<int> vec; // 顶点列表
    vector<vector<int>> adjMat;

    GraphAdjMat(const vector<int> &vec, const vector<vector<int>> &edges)
    {
        for (int val : vec)
        {
            addVertex(val);
        }

        for (const vector<int> &edge : edges)
        {
            addEdge(edge[0], edge[1]);
        }
    }

    int size()
    {
        return vec.size();
    }

    void addVertex(int val)
    {
        int n = size();
        vec.push_back(val);
        adjMat.emplace_back(vector<int>(n, 0)); // 向邻接矩阵中添加一行
        for (vector<int> &row : adjMat)
        {
            row.push_back(0);
        }
    }

    void removeVertex(int index)
    {
        if (index >= size())
        {
            throw out_of_range("顶点不存在");
        }

        vec.erase(vec.begin() + index);
        adjMat.erase(adjMat.begin() + index); // 删除行

        for (vector<int> &row : adjMat)
        {
            row.erase(row.begin() + index); // 删除index列
        }
    }

    void addEdge(int i, int j)
    {
        if (i < 0 || j < 0 || i >= size() || j >= size())
        {
            throw("out of range");
        }
        adjMat[i][j] = 1;
        adjMat[j][i] = 1;
    }

    void removeEdge(int i, int j)
    {
        if (i < 0 || j < 0 || i >= size() || j >= size())
        {
            throw("out of range");
        }

        adjMat[i][j] = 0;
        adjMat[j][i] = 1;
    }
};