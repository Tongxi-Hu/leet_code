use crate::common::{ListNode, TreeNode};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};
use std::rc::Rc;

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

/// p327
pub fn count_range_sum(mut nums: Vec<i32>, lower: i32, upper: i32) -> i32 {
    let mut nums = nums.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    let mut count = 0;
    for i in 1..nums.len() {
        nums[i] += nums[i - 1];
    }
    nums.insert(0, 0);
    sort(&mut nums, lower as i64, upper as i64, &mut count);
    count
}

fn sort(nums: &mut [i64], lower: i64, upper: i64, count: &mut i32) {
    if nums.len() <= 1 {
        return;
    }
    let mid = nums.len() / 2;
    sort(&mut nums[..mid], lower, upper, count);
    sort(&mut nums[mid..], lower, upper, count);
    merge_2(nums, mid, lower, upper, count);
}

fn merge_2(nums: &mut [i64], mid: usize, lower: i64, upper: i64, count: &mut i32) {
    // 滑动窗口
    let (mut start, mut end) = (mid, mid);
    for i in 0..mid {
        while start < nums.len() && nums[start] - nums[i] < lower {
            start += 1;
        }
        while end < nums.len() && nums[end] - nums[i] <= upper {
            end += 1;
        }
        *count += (end - start) as i32;
    }

    let mut tmp = Vec::with_capacity(nums.len());
    let (mut i, mut j) = (0, mid);
    while i < mid && j < nums.len() {
        if nums[i] <= nums[j] {
            tmp.push(nums[i]);
            i += 1;
        } else {
            tmp.push(nums[j]);
            j += 1;
        }
    }
    nums.iter().take(mid).skip(i).for_each(|x| {
        tmp.push(*x);
    });
    nums.iter().skip(j).for_each(|x| {
        tmp.push(*x);
    });
    nums.copy_from_slice(&tmp);
}

/// p328
pub fn odd_even_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    if head.is_none() {
        return None;
    }
    let mut head = head.unwrap();
    let mut p1 = head.as_mut();
    let head2 = p1.next.take();
    if let Some(mut head2) = head2 {
        let mut p2 = head2.as_mut();
        while p2.next.is_some() {
            p1.next = p2.next.take();
            p1 = p1.next.as_mut().unwrap();
            p2.next = p1.next.take();
            if p2.next.is_some() {
                p2 = p2.next.as_mut().unwrap();
            }
        }
        p1.next = Some(head2);
    }
    Some(head)
}

/// p329
pub fn longest_increasing_path(matrix: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = (matrix.len(), matrix[0].len());
    if rows == 0 || cols == 0 {
        return 0;
    }
    let mut longest = vec![vec![0; cols]; rows];
    fn dfs(
        r: usize,
        c: usize,
        longest: &mut Vec<Vec<i32>>,
        matrix: &Vec<Vec<i32>>,
        rows: usize,
        cols: usize,
    ) -> i32 {
        if longest[r][c] != 0 {
            return longest[r][c];
        } else {
            longest[r][c] = 1;
            if r > 0 && matrix[r - 1][c] > matrix[r][c] {
                longest[r][c] = longest[r][c].max(dfs(r - 1, c, longest, matrix, rows, cols) + 1);
            }
            if r < rows - 1 && matrix[r + 1][c] > matrix[r][c] {
                longest[r][c] = longest[r][c].max(dfs(r + 1, c, longest, matrix, rows, cols) + 1);
            }
            if c > 0 && matrix[r][c - 1] > matrix[r][c] {
                longest[r][c] = longest[r][c].max(dfs(r, c - 1, longest, matrix, rows, cols) + 1);
            }
            if c < cols - 1 && matrix[r][c + 1] > matrix[r][c] {
                longest[r][c] = longest[r][c].max(dfs(r, c + 1, longest, matrix, rows, cols) + 1);
            }
        }
        longest[r][c]
    }
    for r in 0..rows {
        for c in 0..cols {
            dfs(r, c, &mut longest, &matrix, rows, cols);
        }
    }
    *longest.iter().flatten().max().unwrap_or(&(0 as i32))
}

/// p330
pub fn min_patches(nums: Vec<i32>, n: i32) -> i32 {
    let mut idx = 0;
    let mut x = 1;
    let mut patches = 0;

    while x <= n {
        let v = match nums.get(idx) {
            Some(&v) if v <= x => {
                idx += 1;
                v
            }
            _ => {
                patches += 1;
                x
            }
        };
        x = match x.checked_add(v) {
            Some(v) => v,
            None => break,
        }
    }

    patches
}

/// p331
pub fn is_valid_serialization(preorder: String) -> bool {
    if preorder.len() == 0 {
        return true;
    }
    let contents = preorder.split(',').collect::<Vec<&str>>();
    let mut slots: Vec<bool> = vec![true];
    for content in contents {
        match content {
            "#" => {
                if slots.len() == 0 {
                    return false;
                } else {
                    slots.pop();
                }
            }
            _ => {
                if slots.len() == 0 {
                    return false;
                } else {
                    slots.push(true);
                }
            }
        }
    }
    slots.len() == 0
}

/// p332
pub fn find_itinerary(tickets: Vec<Vec<String>>) -> Vec<String> {
    let mut graph: HashMap<String, BinaryHeap<Reverse<String>>> = HashMap::new();
    for v in tickets.into_iter() {
        graph
            .entry(v[0].to_owned())
            .or_insert(BinaryHeap::new())
            .push(Reverse(v[1].to_owned()));
    }

    let mut result = vec![];
    dfs(&mut graph, "JFK", &mut result);

    result.reverse();
    result
}

fn dfs(
    graph: &mut HashMap<String, BinaryHeap<Reverse<String>>>,
    from: &str,
    result: &mut Vec<String>,
) {
    while let Some(tos) = graph.get_mut(from) {
        match tos.pop() {
            Some(Reverse(to)) => dfs(graph, &to, result),
            None => break,
        }
    }
    result.push(from.to_string());
}

/// p334
pub fn increasing_triplet(nums: Vec<i32>) -> bool {
    if nums.len() < 3 {
        return false;
    };
    let (mut first, mut second) = (nums[0], i32::MAX);
    for num in nums {
        if num > second {
            return true;
        } else if num > first {
            second = num;
        } else {
            first = num;
        }
    }
    return false;
}

/// p335
pub fn is_self_crossing(distance: Vec<i32>) -> bool {
    distance.windows(4).any(|v| v[0] >= v[2] && v[1] <= v[3])
        || distance
            .windows(5)
            .any(|v| v[0] + v[4] >= v[2] && v[1] == v[3])
        || distance
            .windows(6)
            .any(|v| v[0] + v[4] >= v[2] && v[1] + v[5] >= v[3] && v[1] <= v[3] && v[2] >= v[4])
}

/// p336
pub fn palindrome_pairs(words: Vec<String>) -> Vec<Vec<i32>> {
    let mp = words
        .iter()
        .enumerate()
        .map(|(i, word)| (word.clone(), i))
        .collect::<HashMap<_, _>>();

    let mut ans = Vec::new();

    for (i, word) in words.iter().enumerate() {
        for k in 0..word.len() {
            if (0..=k / 2).all(|l| word[l..=l] == word[k - l..=k - l]) {
                if let Some(&j) = mp.get(&(word[k + 1..].chars().rev().collect::<String>())) {
                    ans.push(vec![j as i32, i as i32]);
                }
            }
        }

        let rev = word.chars().rev().collect::<String>();
        if let Some(&j) = mp.get(&rev) {
            if i != j {
                ans.push(vec![i as i32, j as i32]);
            }
        }

        for k in 0..rev.len() {
            if (0..=k / 2).all(|l| rev[l..=l] == rev[k - l..=k - l]) {
                if let Some(&j) = mp.get(&rev[k + 1..]) {
                    ans.push(vec![i as i32, j as i32]);
                }
            }
        }
    }

    ans
}

/// p337
pub fn rob(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dp(node: &Option<Rc<RefCell<TreeNode>>>) -> [i32; 2] {
        if let Some(n) = node {
            let left_gain = dp(&n.borrow().left);
            let right_gain = dp(&n.borrow().right);
            [
                left_gain[1] + right_gain[1],
                (left_gain[0] + right_gain[0] + n.borrow().val).max(left_gain[1] + right_gain[1]),
            ]
        } else {
            [0, 0]
        }
    }
    dp(&root)[1]
}

/// p338
pub fn count_bits(n: i32) -> Vec<i32> {
    let mut ones: Vec<i32> = vec![];
    for i in 0..=n {
        ones.push(i.count_ones() as i32);
    }
    ones
}

/// p341
#[derive(Debug, PartialEq, Eq)]
pub enum NestedInteger {
    Int(i32),
    List(Vec<NestedInteger>),
}

struct NestedIterator(Vec<i32>);
impl NestedIterator {
    fn new(nestedList: Vec<NestedInteger>) -> Self {
        let mut v = collect(nestedList);
        v.reverse();
        Self(v)
    }
    fn next(&mut self) -> i32 {
        self.0.pop().unwrap()
    }
    fn has_next(&self) -> bool {
        self.0.len() != 0
    }
}
fn collect(nestedList: Vec<NestedInteger>) -> Vec<i32> {
    nestedList
        .into_iter()
        .map(|x| match x {
            NestedInteger::Int(x) => vec![x],
            NestedInteger::List(x) => collect(x),
        })
        .flatten()
        .collect()
}

/// p342
pub fn is_power_of_four(n: i32) -> bool {
    if n <= 0 {
        return false;
    }
    let mut n = n;
    while n % 4 == 0 {
        n = n / 4;
    }
    if n == 1 {
        true
    } else {
        false
    }
}

/// p343
pub fn integer_break(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let n = n as usize;
    let mut dp: Vec<usize> = vec![0; n + 1];
    for i in 2..n + 1 {
        for j in 1..i {
            dp[i] = dp[i].max((j * (i - j)).max(j * dp[i - j]))
        }
    }
    dp[n] as i32
}

/// p344
pub fn reverse_string(s: &mut Vec<char>) {
    s.reverse();
}

/// p345
pub fn reverse_vowels(s: String) -> String {
    let mut chars = s.chars().collect::<Vec<char>>();
    let (mut l, mut r) = (0, chars.len() - 1);
    let vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'];
    while l < r {
        if vowels.iter().find(|&&v| v == chars[l]) == None {
            l = l + 1;
            continue;
        }
        if vowels.iter().find(|&&v| v == chars[r]) == None {
            r = r - 1;
            continue;
        }
        let old = chars[l];
        chars[l] = chars[r];
        chars[r] = old;
        l = l + 1;
        r = r - 1;
    }
    chars.into_iter().collect::<String>()
}

/// p347
pub fn top_k_frequent(nums: Vec<i32>, k: i32) -> Vec<i32> {
    let mut appearance: HashMap<i32, i32> = HashMap::new();
    nums.into_iter().for_each(|item| {
        appearance
            .entry(item)
            .and_modify(|count| *count = *count - 1)
            .or_insert(-1);
    });
    let mut heap: BinaryHeap<(i32, i32)> = BinaryHeap::with_capacity(k as usize);
    appearance.into_iter().for_each(|item| {
        heap.push((item.1, item.0));
        if heap.len() > k as usize {
            heap.pop();
        }
    });
    heap.into_iter().map(|item| item.1).collect::<Vec<i32>>()
}

// p349
pub fn intersection(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    let mut counter: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    nums1.into_iter().for_each(|num| {
        counter.entry(num).or_insert(1);
    });
    nums2.into_iter().for_each(|num| {
        counter.entry(num).and_modify(|count| {
            if *count == 1 {
                *count = 2
            }
        });
    });
    counter
        .into_iter()
        .filter(|item| item.1 == 2)
        .map(|item| item.0)
        .collect::<Vec<i32>>()
}

/// p350
pub fn intersect(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    let (mut nums1, mut nums2) = (nums1, nums2);
    nums1.sort();
    nums2.sort();
    let mut common: Vec<i32> = vec![];
    let (mut i, mut j) = (0, 0);
    while i < nums1.len() && j < nums2.len() {
        if nums1[i] < nums2[j] {
            i = i + 1;
        } else if nums2[j] < nums1[i] {
            j = j + 1
        } else {
            common.push(nums1[i]);
            i = i + 1;
            j = j + 1;
        }
    }
    common
}

/// p352
struct SummaryRanges {
    intervals: BTreeSet<(i32, i32)>,
}

impl SummaryRanges {
    fn new() -> Self {
        Self {
            intervals: BTreeSet::new(),
        }
    }

    fn add_num(&mut self, value: i32) {
        let mut new_edge1 = value;
        let mut new_edge2 = value;

        if let Some(&prev_range) = self.intervals.range(..(value, value)).next_back() {
            if prev_range.1 >= value {
                return;
            }
            if prev_range.1 + 1 == value {
                self.intervals.remove(&prev_range);
                new_edge1 = prev_range.0;
            }
        }

        if let Some(&next_range) = self.intervals.range((value, value)..).next() {
            if next_range.0 <= value {
                return;
            }
            if value + 1 == next_range.0 {
                self.intervals.remove(&next_range);
                new_edge2 = next_range.1;
            }
        }

        self.intervals.insert((new_edge1, new_edge2));
    }

    fn get_intervals(&self) -> Vec<Vec<i32>> {
        self.intervals
            .iter()
            .map(|&(edge1, edge2)| vec![edge1, edge2])
            .collect::<Vec<_>>()
    }
}

/// p354
pub fn max_envelopes(envelopes: Vec<Vec<i32>>) -> i32 {
    let mut envelopes = envelopes;
    envelopes.sort_unstable_by(|a, b| a[0].cmp(&b[0]).then(b[1].cmp(&a[1])));
    let mut sub = vec![];
    for envelope in envelopes {
        let (_, h) = (envelope[0], envelope[1]);
        let i = sub.binary_search(&h);
        let i = match i {
            Ok(i) => i,
            Err(i) => i,
        };
        if i == sub.len() {
            sub.push(h);
        } else {
            sub[i] = h;
        }
    }
    sub.len() as i32
}

/// p355
struct Twitter {
    follows: HashMap<i32, HashSet<i32>>,
    tweets: HashMap<i32, Vec<(usize, i32)>>,
    id: usize,
}

impl Twitter {
    fn new() -> Self {
        Self {
            follows: HashMap::new(),
            tweets: HashMap::new(),
            id: 0,
        }
    }

    fn post_tweet(&mut self, user_id: i32, tweet_id: i32) {
        let cur_id = self.id;
        self.tweets
            .entry(user_id)
            .or_insert(Vec::new())
            .push((cur_id, tweet_id));
        self.id += 1;
    }

    fn get_news_feed(&self, user_id: i32) -> Vec<i32> {
        // 1.
        let mut all_tweets = vec![];
        let mut idxs = vec![];

        if let Some(tweets) = self.tweets.get(&user_id) {
            idxs.push(tweets.len());
            all_tweets.push(tweets);
        }
        if let Some(followee_ids) = self.follows.get(&user_id) {
            for id in followee_ids.iter() {
                if let Some(tweets) = self.tweets.get(id) {
                    idxs.push(tweets.len());
                    all_tweets.push(tweets);
                }
            }
        }

        // 2.
        Self::get_news_tweet(&all_tweets, &mut idxs)
    }

    fn get_news_tweet(all_tweets: &Vec<&Vec<(usize, i32)>>, idxs: &mut Vec<usize>) -> Vec<i32> {
        let mut max_heap = BinaryHeap::new();
        for (i, tweets) in all_tweets.iter().enumerate() {
            let idx = idxs.get_mut(i).unwrap();
            if *idx > 0 {
                let (time, id) = tweets[*idx - 1];
                max_heap.push((time, id, i));
                *idx -= 1;
            }
        }

        let mut result = vec![];
        while let Some((_, id, idx)) = max_heap.pop() {
            result.push(id);
            if result.len() == 10 {
                break;
            }

            let cur_idx = idxs.get_mut(idx).unwrap();
            if *cur_idx > 0 {
                let (time, id) = all_tweets[idx][*cur_idx - 1];
                max_heap.push((time, id, idx));
                *cur_idx -= 1;
            }
        }

        result
    }

    fn follow(&mut self, follower_id: i32, followee_id: i32) {
        if follower_id == followee_id {
            return;
        }
        self.follows
            .entry(follower_id)
            .or_insert(HashSet::new())
            .insert(followee_id);
    }

    fn unfollow(&mut self, follower_id: i32, followee_id: i32) {
        if follower_id == followee_id {
            return;
        }
        self.follows
            .entry(follower_id)
            .or_insert(HashSet::new())
            .remove(&followee_id);
    }
}

/// p357
pub fn count_numbers_with_unique_digits(n: i32) -> i32 {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 10;
    }
    let (mut res, mut cur) = (10, 9);
    for i in 0..n - 1 {
        cur *= 9 - i;
        res += cur;
    }
    return res;
}

/// p363
pub fn max_sum_submatrix(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut result = i32::MIN;

    fn max_sum(nums: &Vec<i32>, k: i32) -> i32 {
        let mut sum = 0;
        let mut set = BTreeSet::new();
        set.insert(0);
        let mut result = i32::MIN;

        for &num in nums.iter() {
            sum += num;
            if let Some(&val) = set.range((sum - k)..).next() {
                result = i32::max(result, sum - val);
                if result == k {
                    return k;
                }
            }
            set.insert(sum);
        }

        result
    }

    for i in 0..m {
        let mut sums = vec![0; n];
        for j in i..m {
            for k in 0..n {
                sums[k] += matrix[j][k];
            }

            let max_val = max_sum(&sums, k);
            result = i32::max(result, max_val);
            if result == k {
                return k;
            }
        }
    }

    result
}

/// p365
pub fn can_measure_water(x: i32, y: i32, target: i32) -> bool {
    if target > x + y {
        return false;
    }
    if x == 0 || y == 0 {
        return target == 0 || x + y == target;
    }
    let (mut x, mut y) = (x, y);
    while y != 0 {
        if y < x {
            let temp = x;
            x = y;
            y = temp;
        }
        y = y % x;
    }
    return target % x == 0;
}

/// p367
pub fn is_perfect_square(num: i32) -> bool {
    let temp = f64::from(num).sqrt().floor() as i32;
    return temp * temp == num || (temp + 1) * (temp + 1) == num;
}

/// p368
pub fn largest_divisible_subset(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    nums.sort();
    let length = nums.len();
    let mut max_size = 1;
    let mut max_value = nums[0];
    let mut dp = vec![1; length];
    for i in 1..length {
        for j in 0..i {
            if nums[i] % nums[j] == 0 {
                dp[i] = dp[i].max(dp[j] + 1);
            }
            if dp[i] > max_size {
                max_size = dp[i];
                max_value = nums[i];
            }
        }
    }
    if max_value == nums[0] {
        return vec![nums[0]];
    }
    let mut res = Vec::<i32>::new();
    for (index, num) in nums.iter().enumerate().rev() {
        if max_size > 0 && dp[index] == max_size && max_value % num == 0 {
            res.push(*num);
            max_size = max_size - 1;
            max_value = *num;
        }
    }
    res.sort();
    res
}

/// p371
pub fn get_sum(a: i32, b: i32) -> i32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let carry = (a & b) << 1;
        a = a ^ b;
        b = carry;
    }
    return a;
}

/// p372
pub fn super_pow(a: i32, b: Vec<i32>) -> i32 {
    fn pow(x: i64, n: i64) -> i64 {
        let mut ans = 1;
        let (mut x, mut n) = (x, n);
        while n > 0 {
            if n & 1 == 1 {
                ans = ans * x % 1337;
            }
            x = x * x % 1337;

            n >>= 1;
        }
        ans
    }
    let mut ans: i64 = 1;
    for i in b {
        ans = pow(ans, 10) * pow(a as i64, i as i64) % 1337;
    }
    ans as i32
}

/// p373
pub fn k_smallest_pairs(nums1: Vec<i32>, nums2: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
    let mut heap: BinaryHeap<Reverse<(i32, usize, usize)>> = nums1
        .iter()
        .enumerate()
        .map(|(i, &n1)| Reverse((n1 + nums2[0], i, 0)))
        .collect();
    let mut ans = vec![];
    let mut k = k;
    while k > 0 {
        if let Some(Reverse((_, i, j))) = heap.pop() {
            if j + 1 < nums2.len() {
                heap.push(Reverse((nums1[i] + nums2[j + 1], i, j + 1)));
            }
            k -= 1;
            ans.push(vec![nums1[i], nums2[j]]);
        } else {
            break;
        }
    }
    ans
}

// p375
pub fn get_money_amount(n: i32) -> i32 {
    let n = n as usize;
    let mut dp = vec![vec![0; n + 1]; n + 1];

    for i in 1..n {
        for j in 1..n - i + 1 {
            dp[j][i + j] = (i / 2 + j..i + j)
                .map(|v| v as i32 + dp[j][v - 1].max(dp[v + 1][i + j]))
                .min()
                .unwrap_or(0);
        }
    }

    dp[1][n]
}
