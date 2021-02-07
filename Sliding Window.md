## [424. 替换后的最长重复字符「Medium」](https://leetcode-cn.com/problems/longest-repeating-character-replacement/)

给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 *k* 次。在执行上述操作后，找到包含重复字母的最长子串的长度。

**注意：**字符串长度 和 *k* 不会超过 $10^4$。

**示例 1：**

```tex
输入：s = "ABAB", k = 2
输出：4
解释：用两个'A'替换为两个'B',反之亦然。
```

**示例 2：**

```tex
输入：s = "AABABBA", k = 1
输出：4
解释：
将中间的一个'A'替换为'B',字符串变为 "AABBBBA"。
子串 "BBBB" 有最长重复字母, 答案为 4。
```

> 算法分析：
>
> ​	我们可以枚举字符串中的每一个位置作为右端点，然后找到其最远的左端点的位置，满足该区间内除了出现次数最多的那一类字符之外，剩余的字符（即非最长重复字符）数量不超过 $k$ 个。
>
> ​	这样我们可以想到使用双指针维护这些区间，每次右指针右移，如果区间仍然满足条件，那么左指针不移动，否则左指针至多右移一格，保证区间长度不减小。
>
> ​	虽然这样的操作会导致部分区间不符合条件，即该区间内非最长重复字符超过了 $k$ 个。但是这样的区间也同样不可能对答案产生贡献。当我们右指针移动到尽头，左右指针对应的区间的长度必然对应一个长度最大的符合条件的区间。
>
> ​	实际代码中，由于字符串中仅包含大写字母，我们可以使用一个长度为 $26$ 的数组维护每一个字符的出现次数。每次区间右移，我们更新右移位置的字符出现的次数，然后尝试用它==更新重复字符出现次数的历史最大值==，最后我们==使用该最大值计算出区间内非最长重复字符的数量==，以此判断左指针是否需要右移即可。

变式「Medium 简单版」：[1004. 最大连续1的个数 III](https://leetcode-cn.com/problems/max-consecutive-ones-iii/)

```c++
class Solution {
public:
    int characterReplacement(string s, int k) {
        vector<int> num(26);
        int n = s.length();
        int maxn = 0;
        int left = 0, right = 0;
        while (right < n) {
            num[s[right] - 'A']++;
            maxn = max(maxn, num[s[right] - 'A']);
            if (right - left + 1 - maxn > k) {
                num[s[left] - 'A']--;
                left++;
            }
            right++;
        }
        return right - left;
    }
};
```

## [1423. 可获得的最大点数「Medium」](https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/)

几张卡牌 排成一行，每张卡牌都有一个对应的点数。点数由整数数组 `cardPoints` 给出。

每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 `k` 张卡牌。

你的点数就是你拿到手中的所有卡牌的点数之和。

给你一个整数数组 `cardPoints` 和整数 `k`，请你返回可以获得的最大点数。

**示例 1：**

```tex
输入：cardPoints = [1,2,3,4,5,6,1], k = 3
输出：12
解释：第一次行动，不管拿哪张牌，你的点数总是 1 。但是，先拿最右边的卡牌将会最大化你的可获得点数。最优策略是拿右边的三张牌，最终点数为 1 + 6 + 5 = 12 。
```

**示例 2：**

```tex
输入：cardPoints = [2,2,2], k = 2
输出：4
解释：无论你拿起哪两张卡牌，可获得的点数总是 4 。
```

**示例 3：**

```tex
输入：cardPoints = [9,7,7,9,7,7,9], k = 7
输出：55
解释：你必须拿起所有卡牌，可以获得的点数为所有卡牌的点数之和。
```

**示例 4：**

```tex
输入：cardPoints = [1,1000,1], k = 1
输出：1
解释：你无法拿到中间那张卡牌，所以可以获得的最大点数为 1 。 
```

> 算法分析：
>
> 我们考虑到每次拿牌都是从首尾拿一张，总共拿 `k` 张，与其去想拿首或者尾的一张，那么不妨逆向思维一下，拿走 `k` 张之后还剩下 `n - k` 张，我们只需要保证剩下的 `n - k` 张牌的点数之和**最小**，这样就可以保证拿走的牌的点数之和最大，这样，我们就可以维护一个滑动窗口来求剩下点数的最小值。

```go
func maxScore(cardPoints []int, k int) int {
    n := len(cardPoints)
    sum := 0
    windowSize := n - k
    for _, val := range cardPoints[:windowSize] {
        sum += val
    }
    minSum := sum
    for i := windowSize; i < n; i++ {
        sum += cardPoints[i] - cardPoints[i - windowSize]
        minSum = min(minSum, sum)
    }
    total := 0
    for _, v := range cardPoints {
        total += v
    }
    return total - minSum
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

## [42. 接雨水「Hard」](https://leetcode-cn.com/problems/trapping-rain-water/)

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

**示例 1：**

![picture](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)

```c++
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```

> 算法分析：
>
> 1. 我们维护两个指针分别指向首尾两端 $\Rightarrow$ $left=1,~right=n-1$ 以及两个变量维护左边和右边的最大高度 $\Rightarrow$ $left\_most$ 和 $right\_most$.
> 2. 此时我们依次遍历，可以分两种情况讨论：
>    - 当 $left\_most<right\_most$ 时，当前位置能存储的水的最大高度取决于 $left\_most$，无论中间的柱子情况如何，我们此时一定可以存储 $left\_most-nums[i]$ 高度的水，直至在左边遇到高度大于 $left\_most$ 的柱子，然后更新 $left\_most$ 的值。
>    - 当 $left\_most>=right\_most$ 时，当前位置能存储的水的最大高度取决于 $right\_most$，无论中间的柱子情况如何，我们此时一定可以存储 $right\_most-nums[i]$ 高度的水，直至在右边遇到高度大于  $right\_most$ 的柱子，然后更新 $right\_most$ 的值。
> 3. 运行至 $left=right$ 结束。

```c++
class Solution {
public:
    int trap(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int left_most = 0, right_most = 0;
        int count = 0;
        while (left < right) {
            if (height[left] < height[right]) {
                height[left] >= left_most ? left_most = height[left] : count += left_most - height[left];
                ++left;
            } else {
                height[right] >= right_most ? right_most = height[right] : count += right_most - height[right];
                --right;
            }
        }
        return count;
    }
};
```

## [76. 最小覆盖子串「Hard」](https://leetcode-cn.com/problems/minimum-window-substring/)

给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 "" 。

注意：如果 `s` 中存在这样的子串，我们保证它是唯一的答案。

**示例 1：**

```tex
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
```

> 算法分析：
>
> 本问题要求我们返回字符串 `s` 中包含字符串 `t` 的全部字符的**最小**窗口。我们称包含 `t` 的全部字母的窗口为「可行」窗口。
>
> 我们考虑使用滑动窗口来解决此问题，维护两个指针 $left$ 和 $right$，其中 ==$right$ 用来「延展」窗口，$left$ 用来 「收缩」窗口==。
>
> 我们的遍历在字符串 `s` 中进行，会出现以下几种情况：
>
> - 首先，不断右移 $right$ 指针，直至目前的子串完全包含 `t` 中的所有字符.
> - 其次，我们开始收缩 $left$ 指针，直至 $[left,~right]$ 区间内**不完全**包含 `t` 中所有字符，其中我们使用变量 `begin` 保存答案字符串的开头位置，`len` 表示符合条件字符串的长度.
> - 当 $right$ 指针遍历至 `s` 的末尾，遍历结束.
>
> Details：
>
> 我们使用两个哈希表 `SFreq` 和 `TFreq` 来存储各个出现字符的次数，`check()` 函数用于判断这两个哈希表之间是否具有**包含关系**.
>
> **算法缺陷**：
>
> 每次左指针 $left$ 移动我们都需要判断两个哈希表之间的差异，造成了时间上的浪费，因此我们可以做一些优化.
>
> 
>
> ==优化算法==：
>
> 我们使用一个变量 `distant` 来维护目前滑动窗口中出现了与 `t` 字符串中的匹配数目，当 `distant == tlen` 的时候，我们此时就可以移动 $left$ 指针了，移动的过程中更新答案字符串的开头和长度，直至 $[left,~right]$ 区间内**不完全**包含 `t` 中所有字符.
>
> Details：
>
> 要清楚答案字符串的长度是 $right - left+1$ 还是 $right-left$，这个长度取决于 $right$ 的自加过程是在「收缩」阶段的前面或者后面，具体细节自己体会.

```c++
class Solution {
private:
    unordered_map<char, int> SFreq, TFreq;
public:
    bool check() {
        for (const auto &p: TFreq) {
            if (SFreq[p.first] < p.second) {
                return false;
            }
        }
        return true;
    }

    string minWindow(string s, string t) {
        for (auto& c : t) {
            ++TFreq[c];
        }
        int begin = -1, slen = (int)s.size();
        int len = INT_MAX;
        int left = 0, right = -1;
        while (right < slen) {
           if (TFreq.count(s[++right])) {
                ++SFreq[s[right]];
            }

            while (check() and left <= right) {
                if (right - left + 1 < len) {
                    len = right - left + 1;
                    begin = left;
                }
               if (TFreq.count(s[left])) {
                    --SFreq[s[left]];
                }
                ++left;
            }
        }
        return begin == -1 ? string() : s.substr(begin, len);
    }
};
```

优化代码：

```c++
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> SFreq, TFreq;
        int slen = (int)s.size(), tlen = (int)t.size();
        if (slen == 0 || tlen == 0 || slen < tlen) {
            return "";
        }
        
        for (const auto& c : t) {
            ++TFreq[c];
        }
        
        int distant = 0;
        int len = slen + 1;
        int left = 0, right = 0;
        int begin = 0;
        
        // [left, right)
        while (right < slen) {
            if (TFreq[s[right]] == 0) {
                ++right;
                continue;
            }
            
            if (SFreq[s[right]] < TFreq[s[right]]) {
                distant++;
            }
            
            SFreq[s[right]]++;
            right++;
            
            while (distant == tlen) {
                if (right - left < len) {
                    len = right - left;
                    begin = left;
                }
                
                if (TFreq[s[left]] == 0) {
                    ++left;
                    continue;
                }
                
                if (SFreq[s[left]] == TFreq[s[left]]) {
                    distant--;
                }
                
                SFreq[s[left]]--;
                left++;
            }
        }
        
        return len == slen + 1 ? "" : s.substr(begin, len);
    }
};
```
