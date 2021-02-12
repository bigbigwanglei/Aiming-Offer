## [84. 柱状图中最大的矩形「Hard」](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

给定 `n` 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 `1` 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

![picture](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram.png)

以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 `[2,1,5,6,2,3]`。

![pictture](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram_area.png)

图中阴影部分为所能勾勒出的最大矩形面积，其面积为 `10` 个单位。

**示例:**

```tex
输入: [2,1,5,6,2,3]
输出: 10
```

> 算法分析：
>
> 枚举「宽」的暴力法这里就不再过多赘述了。
>
> ==算法引入==：
>
> 我们对枚举「高」来做一些分析以及优化。
>
> 对枚举「高」的暴力法就是对每一条「高」进行中心扩展：
>
> - 首先我们枚举某一根柱子 $i$ 作为高 $h=height[i]$.
> - 随后我们需要进行向左右两边扩展，使得扩展到的柱子的高度均不小于 $h$，也就是说，我们需要找到**左右两侧最近的高度小于 $h$ 的柱子**，这两根柱子之间的所有柱子的高度均不小于 $h$，这就是 $i$ 能扩展的最远范围.
>
> 对于上述分析我们可知，我们可以使用**单调栈**来维护左边和右边的高度依赖关系。由分析可知，我们需要查找的是距离 $i$ 最近比 $h$ 低的柱子，所以我们可以维护一个单调递增的栈，每次只需要查找对比栈顶元素与当前的柱子高度，进行 $pop$ 操作和 $push$ 操作即可。
>
> ==单调栈的算法如下==：
>
> - 当我们枚举到第 $i$ 根柱子时，我们从栈顶不断移除 $height[j]>height[i]$ 的 $j$ 值。在移除完毕后，栈顶元素的 $j$ 值就一定满足 $height[j]<height[i]$，此时 $j$ 就是 $i$ 左侧且最近的小于其高度的柱子。
> - 这里会有一种特殊情况。如果我们移除了栈中所有的 $j$ 值，说明 $i$ 左侧所有柱子的高度都大于 $height[i]$，那么我们可以认为 $i$ 左侧且最近的小于其高度的柱子在位置 $j=-1$，它是一根「虚拟」的、高度无限低的柱子。这样的定义不会对我们的答案产生任何的影响，我们也称这根「虚拟」的柱子为「哨兵」。
> - 最后我们将 $i$ 这根柱子入栈。
>
> Details：
>
> 我们在栈中存放的是柱子的下标而非柱子高度。
>
> 最后求面积为 `maxArea = max(maxArea, (right[i] - left[i] - 1) * heights[i])`，具体下标的长度需要注意。
>
> 本题还能进行常数优化 $\Rightarrow$ 一次遍历，这里就不提了 ( 本人也没学 )

```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int maxArea = 0;
        int n = (int)heights.size();
        stack<int> min_stack;
        vector<int> left(n), right(n);
        
        for (int i = 0; i < n; ++i) {
            while (!min_stack.empty() and heights[i] <= heights[min_stack.top()]) {
                min_stack.pop();
            }
            left[i] = min_stack.empty() ? -1 : min_stack.top();
            min_stack.push(i);
        }
        
        min_stack = stack<int>();
        for (int i = n - 1; i >= 0; --i) {
            while (!min_stack.empty() and heights[i] <= heights[min_stack.top()]) {
                min_stack.pop();
            }
            right[i] = min_stack.empty() ? n : min_stack.top();
            min_stack.push(i);
        }
        
        for (int i = 0; i < n; ++i) {
            maxArea = max(maxArea, (right[i] - left[i] - 1) * heights[i]);
        }
        return maxArea;
    }
};
```
