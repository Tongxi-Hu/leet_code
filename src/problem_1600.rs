use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    i32,
    str::FromStr,
};

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
