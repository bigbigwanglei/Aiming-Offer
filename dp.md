# Chapter 2 动态规划

## [72. 编辑距离「Hard」](https://leetcode-cn.com/problems/edit-distance/)

给你两个单词 `word1` 和 `word2`，请你计算出将 `word1` 转换成 `word2` 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

**示例 1：**

```tex
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2：**

```tex
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

> 算法分析：
>
> 1. 状态：`dp[i][j]` 表示 `word1` 的前 `i` 个字母和 `word2` 的前 `j` 个字母之间的编辑距离。
>
> 2. 状态转移方程：
>
>    - 当 ==`word1[i - 1] == word2[j - 1]`== 时，不需要转换，编辑距离为 ==`dp[i] [j] = dp[i - 1] [j - 1]`==.
>    - 当 ==`word1[i - 1] != word2[j - 1]`== 时，分三种情况讨论：
>      - 插入一个字符，==`dp[i] [j - 1]`==：为 `A` 的前 `i` 个字符和 `B` 的前 `j - 1` 个字符编辑距离的子问题。即对于 `B` 的第 `j` 个字符，我们在 `A` 的末尾添加了一个相同的字符
>      - 删除一个字符，==`dp[i - 1] [j]`==：为 `A` 的前 `i - 1` 个字符和 `B` 的前 `j` 个字符编辑距离的子问题。即对于 `A` 的第 `i` 个字符，我们在 `B` 的末尾添加了一个相同的字符
>      - 替换一个字符，==`dp[i - 1] [j - 1]`==：为 `A` 前 `i - 1` 个字符和 `B` 的前 `j - 1` 个字符编辑距离的子问题。即对于 `B` 的第 `j` 个字符，我们修改 `A` 的第 `i` 个字符使它们相同
>    - 对于这三种情况，我们去其中编辑距离的最小值再加一，即 ==`dp[i][j] = min(dp[i - 1] [j - 1], dp[i - 1] [j], dp[i] [j - 1]) + 1`==.
>
> 3. 边界条件：
>
>    <img src="/Users/wanglei/Library/Application Support/typora-user-images/image-20210206201135253.png" alt="image-20210206201135253" style="zoom:50%;" />

```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        // dp[i][j] 表示 word1 的前 i 个字母和 word2 的前 j 个字母之间的编辑距离。
        int len1 = (int)word1.length(), len2 = (int)word2.length();
        vector<vector<int>> dp(len1 + 1, vector<int>(len2 + 1));
        for (int i = 0; i < len1 + 1; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j < len2 + 1; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i < len1 + 1; i++) {
            for (int j = 1; j < len2 + 1; j++) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {    //dp[i-1][j-1] 表示替换操作，dp[i-1][j] 表示删除操作，dp[i][j-1] 表示插入操作
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }
        return dp[len1][len2];
    }
};
```
