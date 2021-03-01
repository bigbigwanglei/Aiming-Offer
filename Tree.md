## Preface

> 树的解决我们一般使用递归方法，递归概述如下：
>
> - 将问题转化为规模更小的子问题，直至边界情况
> - 递归方程 + 边界条件
> - 借助于计算机的程序栈，利用函数自身调用来实现

## [124. 二叉树中的最大路径和「Hard」](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 `root` ，返回其 最大路径和 。

**示例 1：**

![picture](https://assets.leetcode.com/uploads/2020/10/13/exx1.jpg)

```tex
输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
```

**示例 2：**

![picture](https://assets.leetcode.com/uploads/2020/10/13/exx2.jpg)

```tex
输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```

> 算法分析：
>
> 首先，考虑实现一个简化的函数 `maxGain(node)`，该函数计算二叉树中的一个节点的最大贡献值，具体而言，就是在以该节点为根节点的子树中寻找以该节点为起点的一条路径，使得该路径上的节点值之和最大。
>
> 具体而言，该函数的计算如下：
>
> - 空节点的最大贡献值等于 $0$.
> - 非空节点的最大贡献值等于节点值与其子节点中的最大贡献值之和 ( 对于叶节点而言，最大贡献值等于节点值 )
>
> 例如，考虑如下二叉树。
>
> ```c++
>    -10
>    / \
>   9  20
>     /  \
>    15   7
> ```
>
> 叶节点 `9、15、7` 的最大贡献值分别为 `9、15、7`。
>
> 得到叶节点的最大贡献值之后，再计算非叶节点的最大贡献值。
>
> 节点 20 的最大贡献值等于 `20 + max(15, 7) =  35`.
>
> 节点 -10 的最大贡献值等于 `-10 + max(35, 9) = 25`.
>
> 上述计算过程是递归的过程，因此，对根节点调用函数 `maxGain`，即可得到每个节点的最大贡献值。
>
> 根据函数 `maxGain` 得到每个节点的最大贡献值之后，如何得到二叉树的最大路径和？==对于二叉树中的一个节点，该节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值，如果子节点的最大贡献值为正，则计入该节点的最大路径和，否则不计入该节点的最大路径和==。维护一个全局变量 `maxSum` 存储最大路径和，在递归过程中更新 `maxSum` 的值，最后得到的 `maxSum` 的值即为二叉树中的最大路径和。

```c++
class Solution {
private:
    int maxSum = INT_MIN;
public:
    int maxGain(TreeNode* root) {
        if (!root) return 0;
        
        // 递归计算左右子节点的最大贡献值
        // 只有在最大贡献值大于 0 时，才会选取对应子节点
        int leftGain = max(maxGain(root->left), 0);
        int rightGain = max(maxGain(root->right), 0);
        
        // 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
        int priceNewpath = root->val + leftGain + rightGain;
        
        // 更新答案
        maxSum = max(maxSum, priceNewpath);
        
        // 返回节点的最大贡献值
        return root->val + max(leftGain, rightGain);
    }
    
    int maxPathSum(TreeNode* root) {
        maxGain(root);
        return maxSum;
    }
};
```
