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
## [992. K 个不同整数的子数组「Hard」](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/)

给定一个正整数数组 `A`，如果 `A` 的某个子数组中不同整数的个数恰好为 `K`，则称 `A` 的这个连续、不一定独立的子数组为*好子数组*。

( 例如，`[1,2,3,1,2]` 中有 `3` 个不同的整数：`1`，`2`，以及 `3`。)

返回 `A` 中*好子数组*的数目。

**示例 1：**

```tex
输入：A = [1,2,1,2,3], K = 2
输出：7
解释：恰好由 2 个不同整数组成的子数组：[1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2].
```

**示例 2：**

```tex
输入：A = [1,2,1,3,4], K = 3
输出：3
解释：恰好由 3 个不同整数组成的子数组：[1,2,1,3], [2,1,3], [1,3,4].
```

**提示：**

- `1 <= A.length <= 20000`
- `1 <= A[i] <= A.length`
- `1 <= K <= A.length`

> 算法分析：
>
> 滑动窗口的==思维定势==：
>
> 我们一般考虑滑动窗口的时候都是每轮循环使 $right$ 向右移动一位，然后固定 $right$，然后「收缩」$left$，但是考虑本题，$right$ 指针其实并不固定：
>
> 对于一个固定的左边界来说，满足「恰好存在 `K` 个不同整数的子区间」的右边界 **不唯一**，且形成区间。
>
> 示例：左边界固定的时候，恰好存在 $2$ 个不同整数的子区间为 `[1, 2], [1, 2, 1], [1, 2, 1, 2]`，总数为 $3$。
>
> <img src="https://pic.leetcode-cn.com/1612775858-VWbhYR-image.png" alt="picture" style="zoom:67%;" />
>
> 但是本题是==「恰好存在 `K` 个不同整数的子区间」==，所以我们需要找到左边界固定的情况下，满足「恰好存在 `K` 个不同整数的子区间」最小右边界和最大右边界。
>
> 对比以前我们做过的，使用「滑动窗口」解决的问题的问法基本都会出现「最小」、「最大」这样的字眼。那么这题如何解决呢？对此，我们可以进行一定的转换：
>
> 把「**恰好**」改成「**最多**」就可以使用双指针一前一后交替向右的方法完成，这是因为 **对于每一个确定的左边界，最多包含** $K$ **种不同整数的右边界是唯一确定的**，并且在左边界向右移动的过程中，右边界或者在原来的地方，或者在原来地方的右边。
>
> 而「最多存在 $K$ 个不同整数的子区间的个数」与「恰好存在 `K` 个不同整数的子区间的个数」的差恰好等于「最多存在 $K-1$ 个不同整数的子区间的个数」。
>
> ![picture](https://pic.leetcode-cn.com/1612776085-sZFGqE-image.png)
>
> 因此原问题就可以转换为求解「最多存在 $K$ 个不同整数的子区间的个数」和 「最多存在 $K-1$ 个不同整数的子区间的个数」。

```go
// 主求解函数
func subarraysWithKDistinct(A []int, K int) int {
    return atMostKDistinct(A, K) - atMostKDistinct(A, K - 1)
}

func atMostKDistinct(A []int, K int) int {
    n := len(A)
    // count 代表 [left, right) 里不同整数的个数
    res, count, left, right := 0, 0, 0, 0
    freq := make([]int, n + 1)
    
    // [left, right) 包含不同整数的个数小于等于 K
    for right < n {
        if freq[A[right]] == 0 {
            count++
        }
        
        freq[A[right]]++
        right++
        
        for count > K {
            freq[A[left]]--
            if freq[A[left]] == 0 {
                count--
            }
            left++
        }
        
        // [left, right) 区间的长度就是对结果的贡献
        res += right - left
    }
    
    return res
}
```
---
## [995. K 连续位的最小翻转次数「Hard」](https://leetcode-cn.com/problems/minimum-number-of-k-consecutive-bit-flips/)

在仅包含 `0` 和 `1` 的数组 `A` 中，一次 `K` 位翻转包括选择一个长度为 `K` 的（连续）子数组，同时将子数组中的每个 `0` 更改为 `1`，而每个 `1` 更改为 `0`。

返回所需的 `K` 位翻转的最小次数，以便数组没有值为 `0` 的元素。如果不可能，返回 `-1`。

**示例 1：**

```tex
输入：A = [0,1,0], K = 1
输出：2
解释：先翻转 A[0]，然后翻转 A[2]。
```

**示例 2：**

```tex
输入：A = [0,0,0,1,0,1,1,0], K = 3
输出：3
解释：
翻转 A[0],A[1],A[2]: A变成 [1,1,1,1,0,1,1,0]
翻转 A[4],A[5],A[6]: A变成 [1,1,1,1,1,0,0,0]
翻转 A[5],A[6],A[7]: A变成 [1,1,1,1,1,1,1,1]
```

**提示：**

1. `1 <= A.length <= 30000`
2. `1 <= K <= A.length`

> 算法分析：
>
> ==**方法一：差分数组**==
>
> 由于对同一个子数组执行两次翻转操作不会改变该子数组，所以对每个长度为 $K$ 的子数组，应至多执行一次翻转操作。
>
> 对于若干个 $K$ 位翻转操作，改变先后顺序并不影响最终翻转的结果。不妨从 $A[0]$ 开始考虑，若 $A[0]=0$，则必定要翻转从位置 $0$ 开始的子数组；若 $A[0]=1$，则不翻转从位置 $0$ 开始的子数组。
>
> 按照这一策略，我们从左到右地执行这些翻转操作。由于翻转操作是唯一的，若最终数组元素均为 $1$，则执行的翻转次数就是最小的。
>
> 若直接模拟上述过程，复杂度将会是 $O(NK)$ 的。考虑优化问题：
>
> 考虑不去翻转数字，而是统计每个数字需要翻转的次数。对于一次翻转操作，相当于把子数组中所有数字的翻转次数加 $1$.
>
> 这启发我们用**差分数组**的思想来计算当前数字需要翻转的次数。我们可以维护一个差分数组 $diff$，其中 $diff[i]$ 表示两个相邻元素 $A[i-1]$ 和 $A[i]$ 的翻转次数的差，对于区间 $[l,r]$，将其元素全部加 $1$，只会影响到 $l$ 和 $r+1$ 处的差分值，故 `diff[l]++ && diff[r + 1]--`.
>
> 通过累加差分数组可以得到当前位置需要翻转的次数，我们用变量 $revCnt$ 来表示这一累加值。
>
> 遍历到 $A[i]$ 时，==若 $A[i]+revCnt$ 是偶数，则说明当前元素的实际值为 $0$，需要翻转区间 $[i,i+K-1]$== ，我们可以直接将 $revCnt$ 增加 $1$，$diff[i+K]$ 减少 $1$.
>
> 注意到若 $i+K>n$ 则无法执行翻转操作，此时应返回 $-1$.
>
> 
>
> ==**方法二：滑动窗口**==
>
> 我们考虑能否将空间复杂度简化为 $O(1)$ ?
>
> 回顾方法一的代码，当遍历到位置 $i$ 时，若能知道位置 $i-K$ 上发生了翻转操作，便可以直接修改 $revCnt$ 从而去掉 $diff$ 数组。
>
> 注意到 $0≤A[i]≤1$，我们可以==用 $A[i]$ 范围**之外**的数来表达「是否翻转过」的含义==。
>
> 具体来说，若要翻转从位置 $i$ 开始的子数组，可以将 $A[i]$ 加 $2$，这样当遍历到位置 $i'$ 时，若有 $A[i'-K]>1$，则说明在位置 $i'-K$ 上发生了翻转操作。

方法一：

```go
func minKBitFlips(A []int, K int) (ans int) {
    n := len(A)
    diff := make([]int, n + 1)
    revCnt := 0
    for i, v := range A {
        revCnt += diff[i]
        if (v + revCnt) % 2 == 0 {
            if i + K > n {
                return -1
            }
            ans++
            revCnt++
            diff[i + K]--
        }
    }
    return
}
```

由于模 $2$ 意义下的加减法与异或等价，我们也可以用异或改写上面的代码。

```go
func minKBitFlips(A []int, K int) (ans int) {
    n := len(A)
    diff := make([]int, n + 1)
    revCnt := 0
    for i, v := range A {
        revCnt ^= diff[i]
        if v == revCnt { // v ^ revCnt == 0
            if i + K > n {
                return -1
            }
            ans++
            revCnt ^= 1
            diff[i + K] ^= 1
        }
    }
    return
}
```

方法二：

```go
func minKBitFlips(A []int, K int) (ans int) {
    n := len(A)
    revCnt := 0
    for i, v := range A {
        if i >= K && A[i - K] > 1 {
            revCnt ^= 1
            A[i - K] -= 2 // 复原数组元素，若允许修改数组 A，则可以省略
        }
        if v == revCnt {
            if i + K > n {
                return -1
            }
            ans++
            revCnt ^= 1
            A[i] += 2
        }
    }
    return
}
```
