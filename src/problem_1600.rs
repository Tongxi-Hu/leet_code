use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    i32,
    rc::Rc,
    str::FromStr,
    usize,
};

use crate::common::TreeNode;

/// 02
pub fn can_make_arithmetic_progression(mut arr: Vec<i32>) -> bool {
    arr.sort();
    arr.windows(2)
        .fold((i32::MAX, true), |acc, cur| {
            if acc.1 == false {
                (acc.0, false)
            } else if acc.0 == i32::MAX {
                (cur[1] - cur[0], true)
            } else if cur[1] - cur[0] != acc.0 {
                (acc.0, false)
            } else {
                (acc.0, true)
            }
        })
        .1
}

/// 03
pub fn get_last_moment(n: i32, mut left: Vec<i32>, mut right: Vec<i32>) -> i32 {
    left.sort();
    right.sort();
    let mut t = 0;
    if right.len() > 0 {
        t = t.max(n - right[0]);
    }
    if left.len() > 0 {
        t = t.max(left[left.len() - 1]);
    }
    t
}

/// 04
pub fn num_submat(mat: Vec<Vec<i32>>) -> i32 {
    let (height, width) = (mat.len(), mat[0].len());
    let mut row = vec![vec![0; width]; height];
    let mut ans = 0;
    for i in 0..height {
        for j in 0..width {
            if mat[i][j] == 0 {
                row[i][j] = 0;
            } else {
                if j == 0 {
                    row[i][j] = 1;
                } else {
                    row[i][j] = 1 + row[i][j - 1];
                }
            }
            let mut cur = row[i][j];
            for k in (0..=i).rev() {
                cur = cur.min(row[k][j]);
                if cur == 0 {
                    break;
                }
                ans = ans + cur;
            }
        }
    }
    ans
}

/// 05
pub fn min_integer(num: String, mut k: i32) -> String {
    let n = num.len();
    let p = num.into_bytes();
    let mut position = vec![vec![]; 10];
    for i in (0..n).rev() {
        position[(p[i] - b'0') as usize].push(i + 1);
    }

    let tree = RefCell::new(vec![0; n + 1]);
    let update = |mut x: i32| {
        loop {
            if x > n as i32 {
                break;
            }
            tree.borrow_mut()[x as usize] += 1;
            x += x & -x;
        }
    };
    let query = |mut x: i32| -> i32 {
        let mut a = 0;
        loop {
            if x == 0 {
                break;
            }
            a += tree.borrow()[x as usize];
            x -= x & -x;
        }
        a
    };
    let sum_range = |l: i32, r: i32| -> i32 { query(r as i32) - query((l - 1) as i32) };
    let mut ans = vec![0_u8; n];
    let mut l = 0;
    for i in 1..=n {
        for j in 0..10 {
            if position[j].len() > 0 {
                let c = *position[j].last().unwrap();
                let b = sum_range(c as i32, n as i32);
                let dist = c as i32 + b - i as i32;
                if dist <= k {
                    update(c as i32);
                    position[j].pop();
                    ans[l] = (j + 48) as u8;
                    l += 1;
                    k -= dist;
                    break;
                }
            }
        }
    }
    unsafe { String::from_utf8_unchecked(ans) }
}

/// 07
pub fn reformat_date(date: String) -> String {
    let vs: Vec<&str> = date.split(' ').collect();
    let ms = vec![
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut i = 0;
    let mut md = 1;
    let mut dd = 0;
    let mut res = String::new();
    for w in vs {
        if i == 0 {
            let l = w.len();
            let t = &w[0..(l - 2)];
            dd = FromStr::from_str(t).unwrap();
        }
        if i == 1 {
            for m in ms.clone() {
                if m == w {
                    break;
                }
                md += 1;
            }
        }
        if i == 2 {
            res.push_str(w);
            res.push('-');
        }
        i += 1;
    }
    if md < 10 {
        res.push('0');
        res.push_str(&md.to_string());
    } else {
        res.push_str(&md.to_string());
    }
    res.push('-');
    if dd < 10 {
        res.push('0');
        res.push_str(&dd.to_string());
    } else {
        res.push_str(&dd.to_string());
    }
    res
}

/// 08
pub fn range_sum(nums: Vec<i32>, _: i32, left: i32, right: i32) -> i32 {
    let mut dp = vec![];
    for i in 0..nums.len() {
        dp.push(vec![nums[i] as i64]);
        if i != 0 {
            dp[i - 1].clone().iter().for_each(|s| {
                dp[i].push(*s as i64 + nums[i] as i64);
            });
        }
    }
    let mut total = dp.into_iter().flatten().collect::<Vec<i64>>();
    total.sort();
    (total[(left - 1) as usize..right as usize]
        .iter()
        .sum::<i64>()
        % 1000000007) as i32
}

/// 09
pub fn min_difference(nums: Vec<i32>) -> i32 {
    let mut max = vec![i32::MIN; 4];
    let mut min = vec![i32::MAX; 4];
    if nums.len() <= 4 {
        return 0;
    }
    for x in nums.into_iter() {
        for i in 0..4 {
            if x >= max[i] {
                for j in (i + 1..4).rev() {
                    max[j] = max[j - 1];
                }
                max[i] = x;
                break;
            }
        }
        for i in 0..4 {
            if x <= min[i] {
                for j in (i + 1..4).rev() {
                    min[j] = min[j - 1];
                }
                min[i] = x;
                break;
            }
        }
    }
    max.into_iter()
        .zip(min.into_iter().rev())
        .fold(i32::MAX, |ans, (x, y)| ans.min(x - y))
}

/// 10
pub fn winner_square_game(n: i32) -> bool {
    let n = n as usize;
    let mut dp = vec![false; n + 1];
    for i in 1..=n {
        for j in 1..=i {
            if j * j <= i && dp[i - j * j] == false {
                dp[i] = true;
                break;
            }
        }
    }
    dp[n]
}

/// 12
pub fn num_identical_pairs(nums: Vec<i32>) -> i32 {
    let mut cnt = HashMap::new();
    nums.iter().for_each(|n| {
        *cnt.entry(n).or_insert(0) += 1;
    });
    cnt.iter().fold(0, |acc, c| {
        if *c.1 > 1 {
            acc + c.1 * (c.1 - 1) / 2
        } else {
            acc
        }
    })
}

/// 13
pub fn num_sub(s: String) -> i32 {
    let chars = s.chars().collect::<Vec<char>>();
    let (mut dp, mut ans) = (vec![0; chars.len()], 0);
    for i in 0..chars.len() {
        if chars[i] == '0' {
            dp[i] = 0 as i64;
        } else {
            if i == 0 {
                dp[i] = 1;
            } else {
                dp[i] = dp[i - 1] + 1;
            }
            ans = ans + dp[i]
        }
    }
    (ans % 1000000007) as i32
}

/// 14
#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct MyF64(f64);

impl Eq for MyF64 {}
impl Ord for MyF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub fn max_probability(
    n: i32,
    edges: Vec<Vec<i32>>,
    succ_prob: Vec<f64>,
    start_node: i32,
    end_node: i32,
) -> f64 {
    let n = n as usize;

    // 1.
    let mut graph = vec![vec![]; n];
    for (i, edge) in edges.iter().enumerate() {
        let (node1, node2) = (edge[0] as usize, edge[1] as usize);
        graph[node1].push((node2, succ_prob[i]));
        graph[node2].push((node1, succ_prob[i]));
    }
    let mut records = vec![0.0; n];
    let mut max_heap = BinaryHeap::new();
    records[start_node as usize] = 1.0;
    max_heap.push((MyF64(1.0), start_node as usize));
    while let Some((MyF64(cur_prob), cur_node)) = max_heap.pop() {
        if cur_node == end_node as usize {
            return cur_prob;
        }
        if cur_prob > records[cur_node] {
            continue;
        }
        for &(next_node, prob) in graph[cur_node].iter() {
            let next_prob = cur_prob * prob;
            if next_prob > records[next_node] {
                records[next_node] = next_prob;
                max_heap.push((MyF64(next_prob), next_node));
            }
        }
    }
    records[end_node as usize]
}

/// 18
pub fn num_water_bottles(mut num_bottles: i32, num_exchange: i32) -> i32 {
    let mut total = num_bottles;
    while num_bottles >= num_exchange {
        let new_bottle = num_bottles / num_exchange;
        total = total + new_bottle;
        let old_bottle = num_bottles % num_exchange;
        num_bottles = new_bottle + old_bottle;
    }
    total
}

/// 20
pub fn max_num_of_substrings(s: String) -> Vec<String> {
    let mut res: Vec<String> = Vec::new();
    let mut left: Vec<i32> = vec![2147483647; 26];
    let mut right: Vec<i32> = vec![-2147483648; 26];

    let s_b = s.as_bytes();

    for i in 0..s_b.len() {
        left[(s_b[i] - 'a' as u8) as usize] =
            (i as i32).min(left[(s_b[i] as u8 - 'a' as u8) as usize]);
        right[(s_b[i] - 'a' as u8) as usize] =
            (i as i32).max(right[(s_b[i] as u8 - 'a' as u8) as usize]);
    }

    let extention = |i: i32| -> i32 {
        let mut p = right[(s_b[i as usize] - 'a' as u8) as usize];
        let mut cur = i;
        while cur < p {
            if left[(s_b[cur as usize] - 'a' as u8) as usize] < i {
                return -1;
            }

            if right[(s_b[cur as usize] - 'a' as u8) as usize] > p {
                p = right[(s_b[cur as usize] - 'a' as u8) as usize]
            }

            cur += 1;
        }

        return p;
    };

    let mut last: i32 = -1;
    for i in 0..s_b.len() {
        if i != left[(s_b[i] as u8 - 'a' as u8) as usize] as usize {
            continue;
        }

        let p = extention(i as i32);
        if p == -1 {
            continue;
        }

        if i as i32 > last {
            res.push(String::from(&s[i..(p as usize + 1)]));
        } else {
            res.pop();
            res.push(String::from(&s[i..(p as usize + 1)]));
        }

        last = p
    }

    res
}

/// 21
pub fn closest_to_target(arr: Vec<i32>, target: i32) -> i32 {
    const M: usize = 25;

    let n = arr.len();
    let mut sum = vec![[0_usize; M]; n];

    for j in 0..M {
        if arr[0] & (1 << j) > 0 {
            sum[0][j] = 1;
        }
    }

    for i in 1..n {
        for j in 0..M {
            if arr[i] & (1 << j) > 0 {
                sum[i][j] = sum[i - 1][j] + 1;
            } else {
                sum[i][j] = sum[i - 1][j];
            }
        }
    }

    let get = |l: usize, r: usize| -> i32 {
        let mut x = 0;
        for j in 0..M {
            let cnt = match l {
                0 => sum[r][j],
                l => sum[r][j] - sum[l - 1][j],
            };
            if cnt == r - l + 1 {
                x |= 1 << j;
            }
        }
        x
    };

    let mut ans = i32::MAX;
    let mut j = 0;
    for i in 0..n {
        j = j.max(i);
        let mut acc = get(i, j);
        ans = ans.min((acc - target).abs());
        while j < n - 1 && acc >= target {
            j += 1;
            acc &= arr[j];
            ans = ans.min((acc - target).abs());
        }
    }
    ans
}

/// 23
pub fn count_odds(low: i32, high: i32) -> i32 {
    (low..=high)
        .into_iter()
        .fold(0, |acc, cur| if cur % 2 == 0 { acc } else { acc + 1 })
}

/// 24
pub fn num_of_subarrays(arr: Vec<i32>) -> i32 {
    let (mut even, mut odd, mut total, mut prefix_sum) = (1, 0, 0, 0);
    for i in 0..arr.len() {
        prefix_sum = prefix_sum + arr[i];
        if prefix_sum % 2 == 0 {
            total = (total + odd) % 1000000007;
            even = even + 1;
        } else {
            total = (total + even) % 1000000007;
            odd = odd + 1;
        }
    }
    total
}

/// 25
pub fn num_splits(s: String) -> i32 {
    let chars = s.chars().collect::<Vec<char>>();
    let mut right_cnt = chars.iter().fold(HashMap::new(), |mut acc, cur| {
        *acc.entry(cur).or_insert(0) += 1;
        acc
    });
    let (mut cnt, mut left_cnt) = (0, HashMap::new());
    for i in 0..chars.len() {
        if let Some(r) = right_cnt.get_mut(&chars[i]) {
            *r -= 1;
            if *r == 0 {
                right_cnt.remove(&chars[i]);
            }
        }
        *left_cnt.entry(chars[i]).or_insert(0) += 1;
        if left_cnt.len() == right_cnt.len() {
            cnt += 1;
        }
    }
    cnt
}

/// 26
pub fn min_number_operations(target: Vec<i32>) -> i32 {
    let (size, mut ans) = (target.len(), target[0]);
    for i in 1..size {
        ans += (target[i] - target[i - 1]).max(0);
    }
    ans
}

/// 28
pub fn restore_string(s: String, indices: Vec<i32>) -> String {
    let (chars, mut ans) = (s.chars().collect::<Vec<char>>(), vec!['a'; indices.len()]);
    for i in 0..indices.len() {
        ans[indices[i] as usize] = chars[i];
    }
    ans.iter().collect::<String>()
}

/// 29
pub fn min_flips(target: String) -> i32 {
    let (chars, mut cur, mut total) = (target.chars().collect::<Vec<char>>(), '0', 0);
    for i in 0..chars.len() {
        if chars[i] != cur {
            total = total + 1;
            cur = chars[i];
        }
    }
    total
}

/// 30
pub fn count_pairs(root: Option<Rc<RefCell<TreeNode>>>, distance: i32) -> i32 {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, total: &mut i32, distance: i32) -> Vec<i32> {
        if let Some(node) = root.as_ref() {
            let (left, right) = (node.borrow().left.clone(), node.borrow().right.clone());
            if left.is_none() && right.is_none() {
                vec![0]
            } else {
                let mut left_depth = dfs(left, total, distance);
                let mut right_depth = dfs(right, total, distance);
                for i in left_depth.iter() {
                    for j in right_depth.iter() {
                        if i + j + 2 <= distance {
                            *total += 1;
                        }
                    }
                }
                left_depth.append(&mut right_depth);
                left_depth.iter().map(|d| d + 1).collect()
            }
        } else {
            vec![]
        }
    }
    let mut total = 0;
    dfs(root.clone(), &mut total, distance);
    total
}

/// 31
pub fn get_length_of_optimal_compression(s: String, k: i32) -> i32 {
    const CALC: [u8; 101] = [
        0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4,
    ];
    let s = s.into_bytes();
    let n = s.len();
    let mut f = vec![vec![127; k as usize + 1]; n + 1];
    f[0][0] = 0;
    for i in 1..=n {
        for j in 0..i.min(k as usize) + 1 {
            if j > 0 {
                f[i][j] = f[i - 1][j - 1]
            }
            let (mut same, mut diff) = (0, 0);
            for i0 in (1..=i).rev() {
                if s[i0 - 1] == s[i - 1] {
                    same += 1;
                    f[i][j] = f[i][j].min(f[i0 - 1][j - diff] + CALC[same])
                } else {
                    diff += 1;
                    if diff > j {
                        break;
                    }
                }
            }
        }
    }
    f[n][k as usize] as i32
}

/// 34
pub fn count_good_triplets(arr: Vec<i32>, a: i32, b: i32, c: i32) -> i32 {
    let (size, mut cnt) = (arr.len(), 0);
    for i in 0..size - 2 {
        for j in i + 1..size - 1 {
            for k in j + 1..size {
                if (arr[i] - arr[j]).abs() <= a
                    && (arr[j] - arr[k]).abs() <= b
                    && (arr[i] - arr[k]).abs() <= c
                {
                    cnt = cnt + 1;
                }
            }
        }
    }
    cnt
}

/// 35
pub fn get_winner(arr: Vec<i32>, k: i32) -> i32 {
    let mut prev = arr[0].max(arr[1]);
    if k == 1 {
        return prev;
    }
    let mut consecutive = 1;
    let mut max_num = prev;
    for &num in &arr[2..] {
        let curr = num;
        if prev > curr {
            consecutive += 1;
            if consecutive == k {
                return prev;
            }
        } else {
            prev = curr;
            consecutive = 1;
        }
        max_num = max_num.max(curr);
    }
    max_num
}

/// 36
pub fn min_swaps(grid: Vec<Vec<i32>>) -> i32 {
    let n = grid.len();
    let mut pos = vec![-1; n];

    for i in 0..n {
        for j in (0..n).rev() {
            if grid[i][j] == 1 {
                pos[i] = j as i32;
                break;
            }
        }
    }

    let mut ans = 0;
    let mut pos = pos;
    for i in 0..n {
        let mut k = -1;
        for j in i..n {
            if pos[j] <= i as i32 {
                ans += j - i;
                k = j as i32;
                break;
            }
        }

        if k != -1 {
            let k = k as usize;
            for j in (i + 1..=k).rev() {
                pos.swap(j, j - 1);
            }
        } else {
            return -1;
        }
    }
    ans as i32
}

/// 37
pub fn max_sum(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let (mut p_1, mut p_2, size_1, size_2, mut score_1, mut score_2) =
        (0, 0, nums1.len(), nums2.len(), 0, 0);
    while p_1 < size_1 || p_2 < size_2 {
        if p_1 < size_1 && p_2 < size_2 {
            if nums1[p_1] < nums2[p_2] {
                score_1 += nums1[p_1] as i64;
                p_1 += 1;
            } else if nums1[p_1] > nums2[p_2] {
                score_2 += nums2[p_2] as i64;
                p_2 += 1;
            } else {
                let max = score_1.max(score_2) + nums1[p_1] as i64;
                score_1 = max;
                score_2 = max;
                p_1 += 1;
                p_2 += 1;
            }
        } else {
            if p_1 < size_1 {
                score_1 += nums1[p_1] as i64;
                p_1 += 1;
            } else if p_2 < size_2 {
                score_2 += nums2[p_2] as i64;
                p_2 += 1;
            }
        }
    }
    (score_1.max(score_2) % 1000000007) as i32
}

/// 39
pub fn find_kth_positive(arr: Vec<i32>, mut k: i32) -> i32 {
    arr.iter().for_each(|n| {
        if *n <= k {
            k += 1;
        }
    });
    k
}

/// 40
pub fn can_convert_string(s: String, t: String, k: i32) -> bool {
    if s.len() != t.len() {
        return false;
    }

    s.bytes()
        .zip(t.bytes())
        .enumerate()
        .fold(std::collections::HashMap::new(), |mut acc, (_, (u, v))| {
            if u != v {
                *acc.entry((26 + v - u) % 26).or_insert(0) += 1;
            }

            acc
        })
        .into_iter()
        .fold(0, |prev, (k, v)| prev.max((v - 1) * 26 + k as i32))
        <= k
}

/// 41
pub fn min_insertions(s: String) -> i32 {
    let chars = s.chars().collect::<Vec<char>>();
    let (mut left, mut insertion, mut pointer) = (0, 0, 0);
    while pointer < chars.len() {
        if chars[pointer] == '(' {
            left += 1;
            pointer += 1;
        } else {
            if left > 0 {
                left -= 1;
            } else {
                insertion += 1;
            }
            if pointer < chars.len() - 1 && chars[pointer + 1] == ')' {
                pointer += 2;
            } else {
                insertion += 1;
                pointer += 1;
            }
        }
    }
    insertion + left * 2
}

/// 42
pub fn longest_awesome(s: String) -> i32 {
    let mut prefix: HashMap<i32, i32> = HashMap::new();
    prefix.insert(0, -1);
    let mut ans = 0;
    let mut sequence = 0;
    for (j, ch) in s.chars().enumerate() {
        let digit = ch.to_digit(10).unwrap() as i32;
        sequence ^= 1 << digit;
        if let Some(&prev_index) = prefix.get(&sequence) {
            ans = ans.max(j as i32 - prev_index);
        } else {
            prefix.insert(sequence, j as i32);
        }
        for k in 0..10 {
            if let Some(&prev_index) = prefix.get(&(sequence ^ (1 << k))) {
                ans = ans.max(j as i32 - prev_index);
            }
        }
    }
    ans
}

/// 44
pub fn make_good(s: String) -> String {
    let (mut stack, chars) = (VecDeque::<char>::new(), s.chars().collect::<Vec<char>>());
    chars.iter().for_each(|&c| {
        if let Some(&pre) = stack.back() {
            if pre != c && pre.to_ascii_lowercase() == c.to_ascii_lowercase() {
                stack.pop_back();
            } else {
                stack.push_back(c);
            }
        } else {
            stack.push_back(c)
        }
    });
    stack.iter().collect::<String>()
}

/// 45
pub fn find_kth_bit(n: i32, k: i32) -> char {
    fn invert(bit: char) -> char {
        if bit == '0' { '1' } else { '0' }
    }
    fn find_kth_bit_recursive(n: i32, k: i32) -> char {
        if k == 1 {
            return '0';
        }
        let mid = 1 << (n - 1);
        if k == mid {
            return '1';
        } else if k < mid {
            return find_kth_bit_recursive(n - 1, k);
        } else {
            let new_k = mid * 2 - k;
            return invert(find_kth_bit_recursive(n - 1, new_k));
        }
    }
    find_kth_bit_recursive(n, k)
}

/// 46
pub fn max_non_overlapping(nums: Vec<i32>, target: i32) -> i32 {
    let mut prefix_sum = vec![0; nums.len() + 1];
    nums.iter().enumerate().for_each(|(i, &n)| {
        if i == 0 {
            prefix_sum[i + 1] = n;
        } else {
            prefix_sum[i + 1] = prefix_sum[i] + n;
        }
    });
    let (mut last, mut cnt) = (0, 0);
    'a: for right in 1..prefix_sum.len() {
        for left in last..right {
            if prefix_sum[right] - prefix_sum[left] == target {
                cnt = cnt + 1;
                last = right;
                continue 'a;
            }
        }
    }
    cnt
}

/// 47
pub fn min_cost(n: i32, cuts: Vec<i32>) -> i32 {
    let mut cuts = cuts;
    cuts.push(0);
    cuts.push(n);
    cuts.sort();
    let m = cuts.len();
    let mut dp = vec![vec![0; m]; m];
    for i in (0..m).rev() {
        for j in i + 2..m {
            dp[i][j] = i32::MAX;
            for k in i + 1..j {
                dp[i][j] = dp[i][j].min(dp[i][k] + dp[k][j]);
            }
            dp[i][j] += cuts[j] - cuts[i];
        }
    }
    dp[0][m - 1]
}

/// 50
pub fn three_consecutive_odds(arr: Vec<i32>) -> bool {
    arr.windows(3)
        .any(|v| v[0] % 2 == 1 && v[1] % 2 == 1 && v[2] % 2 == 1)
}

/// 51
pub fn min_operations(n: i32) -> i32 {
    (1..=((n - 1) / 2) * 2 + 1)
        .step_by(2)
        .into_iter()
        .fold(0, |acc, cur| acc + n - cur)
}

/// 52
pub fn max_distance(position: Vec<i32>, m: i32) -> i32 {
    let mut position = position;
    position.sort();
    let mut left = 1;
    let mut right = position[position.len() - 1] - position[0];
    let mut ans = -1;

    fn check(x: i32, position: &Vec<i32>, m: i32) -> bool {
        let mut pre = position[0];
        let mut cnt = 1;
        for &pos in &position[1..] {
            if pos - pre >= x {
                pre = pos;
                cnt += 1;
            }
        }
        cnt >= m
    }
    while left <= right {
        let mid = (left + right) / 2;
        if check(mid, &position, m) {
            ans = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    ans
}

/// 53
pub fn min_days(n: i32) -> i32 {
    let mut cache = HashMap::new();
    fn get_min(n: i32, cache: &mut HashMap<i32, i32>) -> i32 {
        if n <= 1 {
            return 1;
        }
        if let Some(&v) = cache.get(&n) {
            v
        } else {
            let two = get_min(n / 2, cache) + 1 + n % 2;
            let three = get_min(n / 3, cache) + 1 + n % 3;
            let min = two.min(three);
            cache.insert(n, min);
            min
        }
    }
    get_min(n, &mut cache)
}

/// 56
fn thousand_separator(mut n: i32) -> String {
    if n == 0 {
        return "0".to_string();
    }

    let mut ans = vec![];
    let mut i = 0;

    while n > 0 {
        if i % 3 == 0 && i > 0 {
            ans.push('.');
        }

        ans.push(((n % 10) as u8 + 48) as char);
        n /= 10;
        i += 1;
    }

    ans.into_iter().rev().collect()
}

/// 57
pub fn find_smallest_set_of_vertices(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
    let mut ans = (0..n).collect::<HashSet<i32>>();
    edges.iter().for_each(|e| {
        ans.remove(&e[1]);
    });
    ans.into_iter().collect::<Vec<i32>>()
}

/// 58
pub fn min_operations_2(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    let len = nums.len();
    let target = vec![0; len];
    let mut res = 0;

    while nums != target {
        nums = nums
            .iter()
            .map(|x| {
                if x % 2 == 1 {
                    res += 1;
                    (x - 1) / 2
                } else {
                    x / 2
                }
            })
            .collect();
        if nums == target {
            break;
        }
        res += 1;
    }
    res
}

/// 59
pub fn contains_cycle(grid: Vec<Vec<char>>) -> bool {
    const DIFF_X: [i32; 4] = [1, 0, -1, 0];
    const DIFF_Y: [i32; 4] = [0, 1, 0, -1];

    fn dfs(
        grid: &Vec<Vec<char>>,
        visited: &mut Vec<Vec<bool>>,
        cur_pos: (usize, usize),
        route: &mut HashMap<(usize, usize), i32>,
        depth: i32,
    ) -> bool {
        let mark = grid[cur_pos.0][cur_pos.1];
        let mut ret = false;

        for i in 0..4 {
            let next_x = cur_pos.0 as i32 + DIFF_X[i];
            let next_y = cur_pos.1 as i32 + DIFF_Y[i];
            if next_x >= 0
                && next_y >= 0
                && next_x < grid.len() as i32
                && next_y < grid[0].len() as i32
            {
                if grid[next_x as usize][next_y as usize] == mark {
                    if !visited[next_x as usize][next_y as usize] {
                        visited[next_x as usize][next_y as usize] = true;
                        route.insert((next_x as usize, next_y as usize), depth + 1);
                        ret |= dfs(
                            grid,
                            visited,
                            (next_x as usize, next_y as usize),
                            route,
                            depth + 1,
                        );
                        route.remove(&(next_x as usize, next_y as usize));
                    } else {
                        let start_depth = *route.get(&(next_x as usize, next_y as usize)).unwrap();
                        if depth - start_depth + 1 >= 4 {
                            ret = true;
                        }
                    }
                }
            }
            if ret {
                break;
            }
        }

        return ret;
    }

    let mut route = HashMap::new();
    let mut visited = vec![vec![false; grid[0].len()]; grid.len()];

    for i in 0..grid.len() {
        for j in 0..grid[i].len() {
            if !visited[i][j] {
                visited[i][j] = true;
                route.insert((i, j), 0);
                if dfs(&grid, &mut visited, (i, j), &mut route, 0) {
                    return true;
                }
                route.remove(&(i, j));
            }
        }
    }

    return false;
}

/// 60
pub fn most_visited(n: i32, rounds: Vec<i32>) -> Vec<i32> {
    let m = rounds.len();
    let start = rounds[0];
    let end = rounds[m - 1];

    if start == end {
        return vec![start];
    }

    if start < end {
        return (start..=end).collect();
    }

    let mut ans = (start..=end + n)
        .map(|x| ((x - 1) % n) + 1)
        .collect::<Vec<i32>>();
    ans.sort();
    ans
}

/// 61
pub fn max_coins(mut piles: Vec<i32>) -> i32 {
    piles.sort();
    piles.reverse();
    let cnt = piles.len() / 3;
    (1..=cnt)
        .into_iter()
        .map(|i| i * 2 - 1)
        .fold(0, |acc, cur| acc + piles[cur])
}

/// 62
pub fn find_latest_step(arr: Vec<i32>, m: i32) -> i32 {
    let mut pos: Vec<Option<(usize, usize)>> = vec![None; arr.len() + 2];
    let (mut ans, mut cnt, m) = (-1, 0, m as usize);
    for (t, &i) in arr.iter().enumerate() {
        let i = i as usize;
        let left = if let Some((start, _)) = pos[i - 1] {
            let p = pos[start].unwrap();
            if p.1 - p.0 + 1 == m {
                cnt -= 1;
            }
            p.0
        } else {
            i
        };
        let right = if let Some((_, end)) = pos[i + 1] {
            let p = pos[end].unwrap();
            if p.1 - p.0 + 1 == m {
                cnt -= 1;
            }
            p.1
        } else {
            i
        };
        if right - left + 1 == m {
            cnt += 1;
        }
        if cnt > 0 {
            ans = t as i32 + 1;
        }
        pos[left] = Some((left, right));
        pos[right] = Some((left, right));
    }
    ans
}

/// 66
pub fn contains_pattern(arr: Vec<i32>, m: i32, k: i32) -> bool {
    arr.windows((m * k) as usize)
        .any(|w| w.chunks(m as usize).all(|p| *p == w[0..m as usize]))
}

/// 67
pub fn get_max_len(nums: Vec<i32>) -> i32 {
    let size = nums.len();
    let (mut dp, mut ans) = (vec![(0, 0); size], 0);
    nums.iter().enumerate().for_each(|(i, &n)| {
        if n > 0 {
            if i == 0 {
                dp[i] = (1, 0);
            } else {
                dp[i] = (
                    dp[i - 1].0 + 1,
                    if dp[i - 1].1 > 0 { dp[i - 1].1 + 1 } else { 0 },
                );
            }
        } else if n == 0 {
            dp[i] = (0, 0);
        } else if n < 0 {
            if i == 0 {
                dp[i] = (0, 1);
            } else {
                dp[i] = (
                    if dp[i - 1].1 > 0 { dp[i - 1].1 + 1 } else { 0 },
                    dp[i - 1].0 + 1,
                );
            }
        }
        ans = ans.max(dp[i].0);
    });
    ans
}

/// 72
pub fn diagonal_sum(mat: Vec<Vec<i32>>) -> i32 {
    let size = mat.len();
    let mut sum = 0;
    for i in 0..size {
        sum = sum + mat[i][i] + mat[i][size - i - 1];
    }
    if size % 2 != 0 {
        sum = sum - mat[size / 2][size / 2]
    }
    sum
}

/// 73
pub fn num_ways(s: String) -> i32 {
    let indices = s
        .bytes()
        .enumerate()
        .filter(|&(_, c)| c == b'1')
        .map(|(i, _)| i as i32)
        .collect::<Vec<_>>();

    if indices.len() % 3 != 0 {
        return 0;
    }

    if indices.is_empty() {
        return ((s.len() - 2) as i64 * (s.len() - 1) as i64 / 2 % 1000000007) as i32;
    }

    let third = indices.len() / 3;
    let split_12 = (indices[third] - indices[third - 1]) as i64;
    let split_23 = (indices[2 * third] - indices[2 * third - 1]) as i64;

    (split_12 * split_23 % 1000000007) as i32
}

/// 74
pub fn find_length_of_shortest_subarray(arr: Vec<i32>) -> i32 {
    let (mut l, mut r, n) = (0, arr.len() - 1, arr.len());
    while l < n - 1 && arr[l] <= arr[l + 1] {
        l += 1;
    }
    if l == n - 1 {
        return 0;
    }
    while r > l && arr[r - 1] <= arr[r] {
        r -= 1;
    }
    let (mut i, mut j, mut ret) = (0, r, (r as i32).min((n - l) as i32 - 1));
    while i <= l && j < n {
        if arr[i] <= arr[j] {
            ret = ret.min((j - i) as i32 - 1);
            i += 1;
        } else {
            j += 1;
        }
    }
    ret
}
