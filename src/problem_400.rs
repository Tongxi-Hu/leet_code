use std::usize;

/// p303
struct NumArray {
    sum: Vec<i32>,
}

impl NumArray {
    fn new(nums: Vec<i32>) -> Self {
        let sum = nums
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let mut sum = 0;
                for i in 0..=index {
                    sum = sum + nums[i]
                }
                return sum;
            })
            .collect::<Vec<i32>>();
        NumArray { sum }
    }

    fn sum_range(&self, left: i32, right: i32) -> i32 {
        if left == 0 {
            return self.sum[right as usize];
        } else {
            return self.sum[right as usize] - self.sum[(left - 1) as usize];
        }
    }
}

/// p304
struct NumMatrix {
    sum: Vec<Vec<i32>>,
}

impl NumMatrix {
    fn new(matrix: Vec<Vec<i32>>) -> Self {
        let m = matrix.len();
        let n = matrix[0].len();
        let mut sum = vec![vec![0; n + 1]; m + 1];
        for (i, row) in matrix.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + x;
            }
        }
        Self { sum }
    }

    fn sum_region(&self, row1: i32, col1: i32, row2: i32, col2: i32) -> i32 {
        let r1 = row1 as usize;
        let c1 = col1 as usize;
        let r2 = row2 as usize + 1;
        let c2 = col2 as usize + 1;
        self.sum[r2][c2] - self.sum[r2][c1] - self.sum[r1][c2] + self.sum[r1][c1]
    }
}

/// p306
pub fn is_additive_number(num: String) -> bool {
    let mut first = 0;
    let num_arr: Vec<char> = num.chars().collect();
    for i in 0..num.len() {
        if i > 0 && num_arr[0] == '0' {
            return false;
        }
        first = first * 10 + (num_arr[i] as u8 - '0' as u8) as i64;
        let mut second = 0;
        for j in i + 1..num.len() {
            second = second * 10 + (num_arr[j] as u8 - '0' as u8) as i64;
            if j > i + 1 && num_arr[i + 1] == '0' {
                break;
            }
            if j + 1 < num.len() && is_can_added(first, second, num.as_str(), j + 1) {
                return true;
            }
        }
    }
    false
}

fn is_can_added(first: i64, second: i64, num: &str, sum_idx: usize) -> bool {
    if sum_idx == num.len() {
        return true;
    }

    let sum_str = i64::to_string(&(first + second));
    if sum_idx + sum_str.len() > num.len() {
        return false;
    }

    let actual_sum = &num[sum_idx..sum_idx + sum_str.len()];
    actual_sum == sum_str && is_can_added(second, first + second, num, sum_idx + sum_str.len())
}

/// p307
struct NumArray_2 {
    nums: Vec<i32>,
    tree: Vec<i32>,
}

impl NumArray_2 {
    fn new(nums: Vec<i32>) -> Self {
        let mut na = Self {
            nums: vec![0; nums.len()],
            tree: vec![0; nums.len() + 1],
        };
        for (i, &x) in nums.iter().enumerate() {
            na.update(i as i32, x);
        }
        na
    }

    fn update(&mut self, index: i32, val: i32) {
        let index = index as usize;
        let delta = val - self.nums[index];
        self.nums[index] = val;
        let mut i = index + 1;
        while i < self.tree.len() {
            self.tree[i] += delta;
            i += (i as i32 & -(i as i32)) as usize;
        }
    }

    fn prefix_sum(&self, i: i32) -> i32 {
        let mut s = 0;
        let mut i = i as usize;
        while i > 0 {
            s += self.tree[i];
            i &= i - 1; // i -= i & -i 的另一种写法
        }
        s
    }

    fn sum_range(&self, left: i32, right: i32) -> i32 {
        self.prefix_sum(right + 1) - self.prefix_sum(left)
    }
}

/// p309
pub fn max_profit(prices: Vec<i32>) -> i32 {
    let length = prices.len();
    let mut dp = vec![vec![0, 0, 0]; length];
    dp[0] = vec![-prices[0], 0, 0];
    for i in 1..length {
        dp[i][0] = dp[i - 1][0].max(dp[i - 1][2] - prices[i]);
        dp[i][1] = dp[i - 1][0] + prices[i];
        dp[i][2] = dp[i - 1][1].max(dp[i - 1][2]);
    }
    return dp[length - 1][1].max(dp[length - 1][2]);
}

/// p310
pub fn find_min_height_trees(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
    if n == 1 {
        return vec![0];
    }
    let mut edge = vec![0; n as usize];
    let mut g = vec![vec![]; n as usize];
    for e in edges.iter() {
        let (x, y) = (e[0] as usize, e[1] as usize);
        edge[x] += 1;
        edge[y] += 1;
        g[x].push(y);
        g[y].push(x);
    }

    let mut q = std::collections::VecDeque::new();
    for (i, &n) in edge.iter().enumerate() {
        if n == 1 {
            q.push_back(i);
        }
    }

    let mut remains = n;
    while remains > 2 {
        remains -= q.len() as i32;
        for _ in 0..q.len() {
            let top = q.pop_front().unwrap();
            for &x in &g[top] {
                edge[x] -= 1;
                if edge[x] == 1 {
                    q.push_back(x);
                }
            }
        }
    }
    q.iter().map(|&x| x as i32).collect()
}

///p312
pub fn max_coins(mut nums: Vec<i32>) -> i32 {
    let n = nums.len();
    nums.insert(0, 1);
    nums.push(1);
    let mut dp = vec![vec![0; n + 2]; n + 2];
    for i in (0..n).rev() {
        for j in i + 2..n + 2 {
            for k in i + 1..j {
                let mut sum = nums[i] * nums[j] * nums[k];
                sum += dp[i][k] + dp[k][j];
                dp[i][j] = dp[i][j].max(sum);
            }
        }
    }
    dp[0][n + 1]
}

/// p313
pub fn nth_super_ugly_number(n: i32, primes: Vec<i32>) -> i32 {
    let n = n as usize;
    let mut idxs = vec![0_usize; primes.len()];
    let mut uglys = vec![1; n];

    for i in 1..n {
        let min_ugly = idxs
            .iter()
            .enumerate()
            .map(|(j, &idx)| i32::saturating_mul(primes[j], uglys[idx]))
            .min()
            .unwrap();
        idxs.iter_mut()
            .enumerate()
            .filter_map(|(j, idx)| {
                if primes[j] * uglys[*idx] == min_ugly {
                    Some(idx)
                } else {
                    None
                }
            })
            .for_each(|idx| *idx += 1);
        uglys[i] = min_ugly;
    }

    uglys[n - 1]
}

/// p315
pub fn count_smaller(nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len();
    let mut right = vec![nums[n - 1]];
    let mut res = vec![0];
    for i in (1..n).rev() {
        let j = nums[i - 1];
        let p = right.partition_point(|&x| x < j);
        right.insert(p, j);
        res.push(p as i32);
    }
    res.reverse();
    res
}

/// p316
pub fn remove_duplicate_letters(s: String) -> String {
    let mut last_appear_index = [0; 26];
    let mut if_in_stack = [false; 26];

    for (n, i) in s.bytes().enumerate() {
        last_appear_index[(i - b'a') as usize] = n as i16;
    }

    let mut stack = vec![b'a'];
    for (n, i) in s.bytes().enumerate() {
        if if_in_stack[(i - b'a') as usize] {
            continue;
        }

        while let Some(s) = stack.pop() {
            if s > i && last_appear_index[(s - b'a') as usize] > n as i16 {
                if_in_stack[(s - b'a') as usize] = false;
            } else {
                stack.push(s);
                break;
            }
        }
        stack.push(i);
        if_in_stack[(i - b'a') as usize] = true;
    }
    stack.drain(1..).map(|x| x as char).collect()
}

/// p318
pub fn max_product(words: Vec<String>) -> i32 {
    let mask: Vec<i32> = words
        .iter()
        .map(|word| {
            word.chars()
                .fold(0, |acc, c| acc | 1 << (c as u8 - 'a' as u8))
        })
        .collect();
    let mut ans = 0;
    for i in 0..mask.len() {
        for j in i + 1..mask.len() {
            if mask[i] & mask[j] == 0 {
                ans = ans.max(words[i].len() * words[j].len());
            }
        }
    }
    ans as i32
}

/// p319
pub fn bulb_switch(n: i32) -> i32 {
    (n as f64 + 0.5).sqrt() as i32
}

/// p321
pub fn max_number(nums1: Vec<i32>, nums2: Vec<i32>, k: i32) -> Vec<i32> {
    let (m, n) = (nums1.len(), nums2.len());
    ((k - n as i32).max(0)..k.min(m as i32) + 1)
        .map(|x| merge(select_max(&nums1, x), select_max(&nums2, k - x)))
        .max()
        .unwrap()
}

fn merge(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    let mut ans = vec![];
    let (mut i, mut j) = (0, 0);

    while i < nums1.len() && j < nums2.len() {
        if nums1[i..] > nums2[j..] {
            ans.push(nums1[i]);
            i += 1;
        } else {
            ans.push(nums2[j]);
            j += 1;
        }
    }
    ans.extend(&nums1[i..]);
    ans.extend(&nums2[j..]);
    ans
}

fn select_max(nums: &[i32], k: i32) -> Vec<i32> {
    let mut to_drop = nums.len() - k as usize;
    let mut stk = vec![i32::MAX];

    for num in nums {
        while to_drop > 0 && stk.last() < Some(num) {
            stk.pop();
            to_drop -= 1;
        }
        stk.push(*num);
    }

    stk[1..stk.len() - to_drop].to_owned()
}

/// p322
pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 {
    let amount = amount as usize;
    let mut coin_nums = vec![amount + 1; amount + 1];
    coin_nums[0] = 0;
    for i in 1..=amount {
        for val in coins.iter() {
            let val = *val as usize;
            if val > i {
                continue;
            } else {
                coin_nums[i] = coin_nums[i].min(coin_nums[i - val] + 1);
            }
        }
    }
    if coin_nums[amount] == amount + 1 {
        -1
    } else {
        coin_nums[amount] as i32
    }
}

/// p324
pub fn wiggle_sort(nums: &mut Vec<i32>) {
    let mut nums_copy = nums.clone();
    nums_copy.sort();
    let n = nums.len();
    let (mut p, mut q) = ((n - 1) / 2, n - 1);
    for i in 0..n {
        if (i & 1) == 1 {
            nums[i] = nums_copy[q];
            q -= 1;
        } else {
            nums[i] = nums_copy[p];
            p -= 1;
        };
    }
}

/// p326
pub fn is_power_of_three(n: i32) -> bool {
    if n <= 0 {
        return false;
    }
    let mut n = n;
    while n % 3 == 0 {
        n = n / 3;
    }
    if n == 1 {
        true
    } else {
        false
    }
}
