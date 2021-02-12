# Warehouse
# Declaration
```c++
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct TrieNode { // 前缀树
    char c;
    unordered_map<char, TrieNode*> children;
    bool isWord;

    TrieNode() {}
    TrieNode(char c) {
        this->c = c;
        this->isWord = false;
    }
};
```
---
# Array
```c++
class Array {
public:
    vector<int> builtVector(int n) {
        vector<int> input(n);
        for (int i = 0; i < n; ++i) {
            cin >> input[i];
        }
        return input;
    }
    
    vector<vector<int>> builtVector(int row, int column) {
        vector<vector<int>> input(row, vector<int>(column));
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < column; ++j) {
                cin >> input[i][j];
            }
        }
        return input;
    }
    
    void print(vector<int>& nums) {
        for (auto& num : nums) {
            cout << num << " ";
        }
        cout << endl;
    }
    
    void print(vector<vector<int>>& nums) {
        for (auto& row : nums) {
            for (auto& column : row) {
                cout << column << " ";
            }
            cout << endl;
        }
    }
};
```
---
# HexConversion
```c++
class HexConversion {
private:
    char hex[16] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
public:
    //  Decimal to binary
    string DecimalToBinary(int n) {
        string ans;
        while (n) {
            if (n & 1) {
                ans = '1' + ans;
            } else {
                ans = '0' + ans;
            }
            n >>= 1;
        }
        return ans + "b";
    }
    
    //Decimal to hexadecimal
    string DecimalToHexadecimal(int n) {
        if (n == 0) {
            return "0x0";
        }
        string ans;
        while (n > 0) {
            ans = hex[n % 16] + ans;
            n /= 16;
        }
        return "0x" + ans;
    }
    
    // Integer to binary string
    string integerToBinary(int num) {
        // or using s = to_string(num)
        string s;
        while (num) {
            s = (num & 1 ? '1' : '0') + s;
            num >>= 1;
        }
        s += 'b';
        return s;
    }
};
```
---
# List
```c++
class List {
public:
    ListNode* createNewList(int n) {
        if (n < 1) {
            return nullptr;
        }
        int x;  cin >> x;
        ListNode* head = new ListNode(x);
        ListNode* tail = head;
        while (--n) {
            cin >> x;
            tail->next = new ListNode(x);
            tail = tail->next;
        }
        return head;
    }
    
    void printList(ListNode* head) {
        while (head) {
            cout << head->val << " ";
            head = head->next;
        }
        cout << endl;
    }
    
    //找到中间节点
    ListNode* findMidPoint(ListNode* head) {
        if (!head) {
            return nullptr;
        }
        if (!head->next) {
            return head;
        }
        ListNode* p = head, *q = head->next;
        while (q && q->next) {
            p = p->next;
            q = q->next->next;
        }
        return p;
    }
    
    // 反转链表
    ListNode* reverse(ListNode* head) { // 会改变原来链表
        ListNode* new_head = nullptr, *temp = nullptr;
        while (head) {
            temp = head->next;
            head->next = new_head;
            new_head = head;
            head = temp;
        }
        return new_head;
    }
    
    // 反转指定位置的链表段
//    输入: 1->2->3->4->5->NULL, m = 2, n = 4
//    输出: 1->4->3->2->5->NULL
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        if (!head) return nullptr;
        
        // Move the two pointers until they reach the proper starting point
        // in the list.
        ListNode* cur = head, *prev = nullptr;
        while (m > 1) {
            prev = cur;
            cur = cur->next;
            m--;
            n--;
        }
        
        // The two pointers that will fix the final connections.
        ListNode* con = prev, *tail = cur;

        // Iteratively reverse the nodes until n becomes 0.
        ListNode* third = nullptr;
        while (n > 0) {
            third = cur->next;
            cur->next = prev;
            prev = cur;
            cur = third;
            n--;
        }

        // Adjust the final connections as explained in the algorithm
        if (con != nullptr) {
            con->next = prev;
        } else {
            head = prev;
        }

        tail->next = cur;
        return head;
    }
    
    // 求两条链表的相交节点
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }
        ListNode *pA = headA, *pB = headB;
        while (pA != pB) {
            pA = pA == nullptr ? headB : pA->next;
            pB = pB == nullptr ? headA : pB->next;
        }
        return pA;
    }
    
    // 判断是否有环 (注意：相交的那个节点不一定是入环点)
    bool hasCycle(ListNode *head) {
        ListNode* fast = head, *slow = head;
        while (fast != nullptr) {
            slow = slow->next;
            if (!fast->next) {
                return false;
            }
            fast = fast->next->next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }
    
    // 循环链表中寻找入环节点
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast != nullptr) {
            slow = slow->next;
            if (fast->next == nullptr) {
                return nullptr;
            }
            fast = fast->next->next;
            if (fast == slow) {
                ListNode *ptr = head;
                while (ptr != slow) {
                    ptr = ptr->next;
                    slow = slow->next;
                }
                return ptr;
            }
        }
        return nullptr;
    }
};
```
---
# Tree
```c++
class Tree {
public:
    TreeNode* builtTrees(TreeNode* root, vector<int>& in, int i) {
        if (i >= in.size() || in[i] == TreeNull) {
            return nullptr;
        }
        root = new TreeNode(in[i]);
        root->left = builtTrees(root->left, in, 2 * i + 1);
        root->right = builtTrees(root->right, in, 2 * i + 2);
        return root;
    }
    
    void Preorder(TreeNode* root, vector<int>& ans) {
        if (!root) return;
        ans.push_back(root->val);
        Preorder(root->left, ans);
        Preorder(root->right, ans);
    }
    
    vector<int> preorderNotRecursion(TreeNode* root) {
        vector<int> res;
        if (root == nullptr) {
            return res;
        }
        stack<TreeNode*> stk;
        TreeNode* node = root;
        while (!stk.empty() || node != nullptr) {
            while (node != nullptr) {
                res.emplace_back(node->val);
                stk.emplace(node);
                node = node->left;
            }
            node = stk.top();
            stk.pop();
            node = node->right;
        }
        return res;
    }
    
    void Inorder(TreeNode* root, vector<int>& ans) {
        if (!root) return;
        Inorder(root->left, ans);
        ans.push_back(root->val);
        Inorder(root->right, ans);
    }
    
    vector<int> inorderNotRecursion(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> stk;
        while (root != nullptr || !stk.empty()) {
            while (root != nullptr) {
                stk.push(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            res.push_back(root->val);
            root = root->right;
        }
        return res;
    }
    
    void Postorder(TreeNode* root, vector<int>& ans) {
        if (!root) return;
        Postorder(root->left, ans);
        Postorder(root->right, ans);
        ans.push_back(root->val);
    }
    
    vector<int> postorderNotRecursion(TreeNode *root) {
        vector<int> res;
        if (root == nullptr) {
            return res;
        }

        stack<TreeNode *> stk;
        TreeNode *prev = nullptr;
        while (root != nullptr || !stk.empty()) {
            while (root != nullptr) {
                stk.emplace(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            if (root->right == nullptr || root->right == prev) {
                res.emplace_back(root->val);
                prev = root;
                root = nullptr;
            } else {
                stk.emplace(root);
                root = root->right;
            }
        }
        return res;
    }
    
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        queue<TreeNode*> q;
        q.push(root);
        vector<int> layerNodes;
        q.push(NULL);
        while (!q.empty()) {
            TreeNode* temp = q.front();
            if (temp == NULL) {
                break;
            }
            q.pop();
            layerNodes.push_back(temp->val);
            if (temp->left) {
                q.push(temp->left);
            }
            if (temp->right) {
                q.push(temp->right);
            }
            if (q.front() == NULL) {
                ans.push_back(layerNodes);
                layerNodes.clear();
                q.pop();
                q.push(NULL);
            }
        }
        return ans;
    }
    
    //求树的最大高度
    int Height(TreeNode* root) {
        if (!root) return 0;
        int left = Height(root->left);
        int right = Height(root->right);
        return max(left, right) + 1;
    }
    
    //判断是否二叉搜索树
    int findMax(TreeNode* root) {
        int maxn = root->val;
        if (root->left) {
            int maxleft = findMax(root->left);
            maxn = maxn > maxleft ? maxn : maxleft;
        }
        if (root->right) {
            int maxright = findMax(root->right);
            maxn = maxn > maxright ? maxn : maxright;
        }
        return maxn;
    }
    int findMin(TreeNode* root) {
        int minn = root->val;
        if (root->left) {
            int minleft = findMin(root->left);
            minn = minn < minleft ? minn : minleft;
        }
        if (root->right) {
            int minright = findMin(root->right);
            minn = minn < minright ? minn : minright;
        }
        return minn;
    }
    bool isValidBST(TreeNode* root) {
        if (!root) return true;
        if ((root->left && findMax(root->left) >= root->val) || (root->right && findMin(root->right) <= root->val)) {
            return false;
        }
        return isValidBST(root->left) && isValidBST(root->right);
    }
};
```
---
# KMP
```c++
class KMP {
public:
    vector<int> getNext(string p) {
        int m = (int)p.size(), i = 0;
        vector<int> next(m, -1);
        int j = -1; // 模式串指针
        while (i < m - 1) {
            if (j == -1 || p[i] == p[j]) {
                ++i;  ++j;
                next[i] = j;
            } else {
                j = next[j];
            }
        }
        return next;
    }
    int kmpMatch(string s, string p) {
        if (s == p) return 0;
        auto next = getNext(p);
        int m = (int)s.length(), i = 0; // 文本串指针
        int n = (int)p.length(), j = 0; // 模式串指针
        while (i < m && j < n) {
            if (j < 0 || s[i] == p[j]) {
                ++i; ++j;
            } else {
                j = next[j];
            }
        }
        if (j == n) {
            return i - j;
        }
        return -1;
    }
};
```
---
# UnionFind
```c++
class UnionFind {
public:
    vector<int> parent;
    vector<int> size;
    int n;
    // 当前连通分量数目
    int setCount;
    
public:
    UnionFind(int _n): n(_n), setCount(_n), parent(_n), size(_n, 1) {
        iota(parent.begin(), parent.end(), 0);
    }
    
    int findset(int x) {
        return parent[x] == x ? x : parent[x] = findset(parent[x]);
    }
    
    bool unite(int x, int y) {
        x = findset(x);
        y = findset(y);
        if (x == y) {
            return false;
        }
        if (size[x] < size[y]) {
            swap(x, y);
        }
        parent[y] = x;
        size[x] += size[y];
        --setCount;
        return true;
    }
    
    bool connected(int x, int y) {
        x = findset(x);
        y = findset(y);
        return x == y;
    }
};
```
---
# GraphUsingMatrix
```c++
class GraphUsingMatrix {
private:
    vector<vector<int>> graph;
    vector<bool> vis;
    int n;
public:
    void print() {
        Array().print(graph);
    }
    GraphUsingMatrix(int n) {
        graph.resize(n, vector<int>(n));
        vis.resize(n);
        this->n = n;
    }
    
    void buildGraph() {
        for (int i = 0; i < this->n; ++i) {
            for (int j = 0; j < this->n; ++j) {
                cin >> graph[i][j];
            }
        }
    }
    
    void dfs(vector<int>& nums, int u) {
        vis[u] = true;
        nums.emplace_back(u);
        for (int i = 0; i < n; ++i) {
            if (!vis[i] and graph[u][i]) {
                dfs(nums, i);
            }
        }
    }
    
    void bfs(vector<int>& nums, int u) {
        vis[u] = true;
        queue<int> qu;
        qu.push(u);
        nums.emplace_back(u);
        while (!qu.empty()) {
            int f = qu.front();
            qu.pop();
            for (int i = 0; i < this->n; ++i) {
                if (!vis[i] and graph[f][i]) {
                    vis[i] = true;
                    qu.push(i);
                    nums.emplace_back(i);
                }
            }
        }
    }
    
    bool checkLoop(int u) {
        vector<bool> vis(this->n, false);
        queue<int> qu;
        qu.push(u);
        vis[u] = true;
        while (!qu.empty()) {
            int father = qu.front();
            qu.pop();
            for (int i = 0; i < this->n; ++i) {
                if (!vis[i] and graph[father][i]) {
                    qu.push(i);
                    vis[i] = true;
                    for (int j = 0; j < this->n; ++j) {
                        if (graph[i][j] and vis[j] and j != father) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
};
```
---
# GraphUsingAdjacencyTable
```c++
class GraphUsingAdjacencyTable {
private:
    vector<vector<int>> graph;
    vector<bool> vis;
    int n;
public:
    GraphUsingAdjacencyTable(int n) {
        graph.resize(n);
        vis.resize(n);
        this->n = n;
    }
    
    void print() {
        Array().print(graph);
    }
    
    void addEdgeUndirected(int u, int v) {
        graph[u].emplace_back(v);
        graph[v].emplace_back(u);
    }
    
    void addEdgeDirected(int u, int v) {
        graph[u].emplace_back(v);
    }
    
    void dfs(vector<int>& nums, int u) {
        vis[u] = true;
        nums.emplace_back(u);
        for (int& v : graph[u]) {
            if (!vis[v]) {
                dfs(nums, v);
            }
        }
    }
    
    void bfs(vector<int>& nums, int u) {
        vis[u] = true;
        queue<int> qu;
        qu.push(u);
        nums.emplace_back(u);
        while (!qu.empty()) {
            int f = qu.front();
            qu.pop();
            for (int& v : graph[f]) {
                if (!vis[v]) {
                    vis[v] = true;
                    qu.push(v);
                    nums.emplace_back(v);
                }
            }
        }
    }
    
    // 找到通过 s 的所有回路
    void findCycle(int u, int s, vector<int>& path, vector<bool>& vis, vector<vector<int>>& result) {
        vis[u] = true;
        path.push_back(u);
        for (int v : graph[u]) {
            if (v == s and path.size() > 2) {
                path.push_back(s);
                result.push_back(path);
                path.pop_back();
            }
            if (!vis[v]) {
                findCycle(v, s, path, vis, result);
            }
        }
        path.pop_back();
        vis[u] = false;
    }
};
```
---
# TrieTree
```c++
class TrieTree { // 前缀树
private:
    TrieNode* root;
public:
    /** Initialize your data structure here. */
    TrieTree() {
        this->root = new TrieNode();
    }

    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode* p = this->root;
        for (int i = 0; i < word.length(); ++i) {
            TrieNode* next;
            char c = word[i];
            if (p->children.count(c) > 0) { // 若有前缀
                next = p->children[c];
            } else { // 没有前缀则新创建一个节点
                next = new TrieNode(c);
//                p->children.insert(make_pair(c, next));
                p->children[c] = next;
            }

            // 遍历下去
            p = next;

            // 最后置为一个单词
            if (i == word.length() - 1) {
                next->isWord = true;
            }
        }
    }

    TrieNode* searchNode(string str) {
        TrieNode* p = root;
        TrieNode* cur = nullptr;
        for (int i = 0; i < str.length(); ++i) {
            char c = str[i];
            if (p->children.count(c) > 0) {
                cur = p->children[c];
                p = cur;
            } else {
                return nullptr;
            }
        }
        return cur;
    }

    /** Returns if the word is in the trie. 查找单词是否存在*/
    bool search(string word) {
        TrieNode* t = searchNode(word);

        if (t != nullptr and t->isWord) {
            return true;
        }

        return false;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. 查找前缀是否存在*/
    bool searchPrefix(string prefix) {
        if (searchNode(prefix) == nullptr) {
            return false;
        }
        return true;
    }
};
```
