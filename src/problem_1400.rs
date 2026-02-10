use std::{
    cell::RefCell,
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    hash::{DefaultHasher, Hasher},
    i32,
    rc::Rc,
};

use crate::common::{ListNode, TreeNode};

/// 1301
pub fn paths_with_max_score(board: Vec<String>) -> Vec<i32> {
    let board = board
        .iter()
        .map(|s| s.chars().collect::<Vec<char>>())
        .collect::<Vec<Vec<char>>>();
    let mut f = board
        .iter()
        .map(|s| {
            s.iter()
                .map(|&c| match c {
                    'X' => (-1, -1),
                    'E' => (0, 1),
                    'S' => (0, 1),
                    _ => (c.to_digit(10).unwrap() as i32, 0),
                })
                .collect::<Vec<(i32, i32)>>()
        })
        .collect::<Vec<Vec<(i32, i32)>>>();

    for i in (0..board.len()).rev() {
        for j in (0..board[i].len()).rev() {
            if f[i][j].0 == -1 || (i == board.len() - 1 && j == board[i].len() - 1) {
                continue;
            } else if i == board.len() - 1 && j < board[i].len() - 1 {
                f[i][j] = (f[i][j].0 + f[i][j + 1].0, f[i][j + 1].1);
            } else if j == board[i].len() - 1 && i < board.len() - 1 {
                f[i][j] = (f[i][j].0 + f[i + 1][j].0, f[i + 1][j].1);
            } else {
                let v = Vec::<(i32, i32)>::from([f[i + 1][j + 1], f[i + 1][j], f[i][j + 1]]);
                let max = match v.iter().filter(|&&x| x.1 != -1).max() {
                    Some(x) => x,
                    None => &(-1, -1),
                };
                if max.0 == -1 {
                    f[i][j] = (-1, -1);
                } else {
                    let max_count = v.iter().filter(|&x| x.0 == max.0).map(|x| x.1).sum::<i32>();
                    f[i][j] = (max.0 + f[i][j].0, max_count % 1000000007);
                }
            }
        }
    }
    match f[0][0].0 {
        -1 => vec![0, 0],
        _ => vec![f[0][0].0, f[0][0].1],
    }
}

/// 1302
pub fn deepest_leaves_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn bfs(nodes: &mut Vec<Option<Rc<RefCell<TreeNode>>>>, sum: &mut i32) {
        let size = nodes.len();
        *sum = (0..size).into_iter().fold(0, |mut acc, _| {
            if let Some(node) = nodes.remove(0) {
                acc = acc + node.borrow().val;
                if node.borrow().left.is_some() {
                    nodes.push(node.borrow().left.clone());
                }
                if node.borrow().right.is_some() {
                    nodes.push(node.borrow().right.clone());
                }
            }
            acc
        });
    }
    let (mut nodes, mut sum) = (vec![root], 0);
    while nodes.len() != 0 {
        bfs(&mut nodes, &mut sum);
    }
    sum
}

/// 1304
pub fn sum_zero(n: i32) -> Vec<i32> {
    (-n / 2..=n / 2)
        .filter(|&i| if n % 2 == 0 { i != 0 } else { true })
        .collect()
}

/// 1305
pub fn get_all_elements(
    root1: Option<Rc<RefCell<TreeNode>>>,
    root2: Option<Rc<RefCell<TreeNode>>>,
) -> Vec<i32> {
    fn in_order(root: Option<Rc<RefCell<TreeNode>>>, vals: &mut VecDeque<i32>) {
        if let Some(node) = root {
            in_order(node.borrow().left.clone(), vals);
            vals.push_back(node.borrow().val);
            in_order(node.borrow().right.clone(), vals);
        }
    }
    let (mut v1, mut v2) = (VecDeque::new(), VecDeque::new());
    in_order(root1, &mut v1);
    in_order(root2, &mut v2);
    let mut ans = vec![];
    while !v1.is_empty() || !v2.is_empty() {
        let (num1, num2) = (
            if v1.is_empty() {
                i32::MAX
            } else {
                *v1.front().unwrap()
            },
            if v2.is_empty() {
                i32::MAX
            } else {
                *v2.front().unwrap()
            },
        );
        ans.push(if num1 < num2 {
            v1.pop_front().unwrap()
        } else {
            v2.pop_front().unwrap()
        })
    }
    ans
}

/// 1306
pub fn can_reach(arr: Vec<i32>, start: i32) -> bool {
    let mut next = VecDeque::new();
    next.push_back(start as usize);
    let (mut reach, mut visited) = (false, vec![false; arr.len()]);
    while next.len() != 0 {
        let index = next.pop_front().unwrap() as usize;
        if arr[index] == 0 {
            reach = true;
            break;
        }
        let dis = arr[index] as usize;
        visited[index] = true;
        if index + dis < arr.len() && visited[index + dis] != true {
            if arr[index + dis] == 0 {
                reach = true;
                break;
            }
            next.push_back(index + dis);
        }
        if index - dis < arr.len() && visited[index - dis] != true {
            if arr[index - dis] == 0 {
                reach = true;
                break;
            }
            next.push_back(index - dis);
        }
    }
    reach
}

/// 1309
pub fn freq_alphabets(s: String) -> String {
    let chars = s.chars().collect::<Vec<char>>();
    let mut s = "".to_string();
    let mut i = 0;
    while i < chars.len() {
        if i + 2 < chars.len() && chars[i + 2] == '#' {
            s.push(
                ((chars[i].to_digit(10).unwrap() - 0) as u8 * 10
                    + (chars[i + 1].to_digit(10).unwrap() - 1) as u8
                    + b'a') as char,
            );
            i = i + 3;
        } else {
            s.push(((chars[i].to_digit(10).unwrap() - 1) as u8 + b'a') as char);
            i = i + 1;
        }
    }
    s
}

/// 1310
pub fn xor_queries(arr: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    let n = arr.len();
    let mut prefix = vec![0; n + 1];

    for i in 0..n {
        prefix[i + 1] = prefix[i] ^ arr[i];
    }

    queries
        .iter()
        .map(|v| prefix[v[0] as usize] ^ prefix[v[1] as usize + 1])
        .collect::<Vec<_>>()
}

/// 1311
pub fn watched_videos_by_friends(
    watched_videos: Vec<Vec<String>>,
    friends: Vec<Vec<i32>>,
    id: i32,
    level: i32,
) -> Vec<String> {
    use std::collections::{HashMap, HashSet, VecDeque};

    let mut q = VecDeque::new();
    let mut state = HashSet::new();
    let mut map = HashMap::new();
    q.push_back((id as usize, 0));
    while !q.is_empty() {
        let (top_id, top_deep) = q.pop_front().unwrap();
        if state.contains(&top_id) {
            continue;
        }
        state.insert(top_id);
        if top_deep == level {
            let watched_vec = &watched_videos[top_id];
            for s in watched_vec {
                map.entry(s).and_modify(|cnt| *cnt += 1).or_insert(0);
            }
            continue;
        }
        let top_friends = &friends[top_id];
        for f_id in top_friends {
            if top_deep + 1 <= level {
                q.push_back((*f_id as usize, top_deep + 1));
            }
        }
    }
    let mut vec = Vec::with_capacity(map.len());
    for item in map {
        vec.push(item);
    }

    vec.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap()
            .then(a.0.partial_cmp(b.0).unwrap())
    });
    let mut ans = Vec::new();
    for (s, _) in vec {
        ans.push(String::from(s));
    }
    ans
}

/// 1312
pub fn min_insertions(s: String) -> i32 {
    let (n, s) = (s.len(), s.as_bytes());

    let mut dp = vec![0; n + 1];
    let (mut pre, mut tmp);
    for i in 1..=n {
        pre = dp[0];
        dp[0] = 0;
        for j in 1..=n {
            tmp = dp[j];
            dp[j] = if s[i - 1] == s[n - j] {
                pre + 1
            } else {
                dp[j].max(dp[j - 1])
            };
            pre = tmp;
        }
    }

    (n - dp[n]) as i32
}

/// 1313
pub fn decompress_rl_elist(nums: Vec<i32>) -> Vec<i32> {
    nums.chunks(2).into_iter().fold(vec![], |mut acc, cur| {
        acc.append(
            &mut (0..cur[0])
                .into_iter()
                .map(|_| cur[1])
                .collect::<Vec<i32>>(),
        );
        acc
    })
}

/// 1314
pub fn matrix_block_sum(mat: Vec<Vec<i32>>, k: i32) -> Vec<Vec<i32>> {
    let n = mat.len();
    let m = mat[0].len();
    let mut pre_sum = vec![vec![0; m]; n];

    for i in 0..n {
        for j in 0..m {
            pre_sum[i][j] += mat[i][j];

            if i > 0 {
                pre_sum[i][j] += pre_sum[i - 1][j];
            }

            if j > 0 {
                pre_sum[i][j] += pre_sum[i][j - 1];
            }

            if i > 0 && j > 0 {
                pre_sum[i][j] -= pre_sum[i - 1][j - 1];
            }
        }
    }

    let k = k as usize;
    let mut ans = vec![vec![0; m]; n];

    for i in 0..n {
        for j in 0..m {
            let left = if j > k { j - k } else { 0 };
            let right = if j + k < m { j + k } else { m - 1 };
            let top = if i > k { i - k } else { 0 };
            let bottom = if i + k < n { i + k } else { n - 1 };

            ans[i][j] = pre_sum[bottom][right];

            if left > 0 {
                ans[i][j] -= pre_sum[bottom][left - 1];
            }

            if top > 0 {
                ans[i][j] -= pre_sum[top - 1][right];
            }

            if left > 0 && top > 0 {
                ans[i][j] += pre_sum[top - 1][left - 1];
            }
        }
    }

    ans
}

/// 1315
pub fn sum_even_grandparent(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, father: bool, grandfather: bool, ans: &mut i32) {
        if let Some(node) = root.as_ref() {
            let is_even = node.borrow().val % 2 == 0;
            if grandfather {
                *ans = *ans + node.borrow().val
            }
            dfs(node.borrow().left.clone(), is_even, father, ans);
            dfs(node.borrow().right.clone(), is_even, father, ans);
        }
    }
    let mut ans = 0;
    dfs(root, false, false, &mut ans);
    ans
}

/// 1316
pub fn distinct_echo_substrings(text: String) -> i32 {
    let mut len = 2;
    let mut res = 0;
    let mut state = HashSet::new();
    while len <= text.len() {
        let mut l = 0;
        let mut r = l + len - 1;
        while r < text.len() {
            let mid = l + r >> 1;

            let mut dh = DefaultHasher::new();
            let l_s = &text[l..mid + 1];
            dh.write(l_s.as_ref());
            let l_hash = dh.finish();

            let mut dh = DefaultHasher::new();
            let r_s = &text[mid + 1..r + 1];
            dh.write(r_s.as_ref());
            let r_hash = dh.finish();

            if l_hash == r_hash {
                if !state.contains(&text[l..r + 1]) {
                    state.insert(&text[l..r + 1]);
                    res += 1;
                }
            }

            l += 1;
            r += 1;
        }
        len += 2;
    }

    res
}

/// 1317
pub fn get_no_zero_integers(n: i32) -> Vec<i32> {
    for i in 1..n / 2 + 1 {
        if !i.to_string().contains('0') && !(n - i).to_string().contains('0') {
            return vec![i, n - i];
        }
    }
    return vec![];
}

/// 1318
pub fn min_flips(a: i32, b: i32, c: i32) -> i32 {
    let mut ans = 0;
    for i in 0..31 {
        let bit_a = (a >> i) & 1;
        let bit_b = (b >> i) & 1;
        let bit_c = (c >> i) & 1;
        if bit_c == 0 {
            ans = ans + bit_a + bit_b
        } else {
            ans = ans + if bit_a + bit_b == 0 { 1 } else { 0 }
        }
    }
    ans
}

/// 1319
pub fn make_connected(mut n: i32, connections: Vec<Vec<i32>>) -> i32 {
    if connections.len() + 1 < n as usize {
        return -1;
    }

    let mut parent: Vec<_> = (0..100000).collect();

    fn union(parent: &mut Vec<usize>, idx1: usize, idx2: usize) {
        let idx1 = find(parent, idx1);
        let idx2 = find(parent, idx2);
        parent[idx1] = idx2;
    }
    fn find(parent: &mut Vec<usize>, mut idx: usize) -> usize {
        while parent[idx] != idx {
            parent[idx] = parent[parent[idx]];
            idx = parent[idx];
        }
        idx
    }
    for conn in connections {
        if find(&mut parent, conn[0] as usize) != find(&mut parent, conn[1] as usize) {
            union(&mut parent, conn[0] as usize, conn[1] as usize);
            n -= 1;
        }
    }
    n - 1
}

/// 1320
pub fn minimum_distance(word: String) -> i32 {
    fn distance(a: i32, b: i32) -> i32 {
        return i32::abs(a / 6 - b / 6) + i32::abs(a % 6 - b % 6);
    }
    let words: Vec<char> = word.chars().collect();
    let n = words.len();
    let mut dp = vec![vec![0x3f3f3f3f; 26]; n];
    dp[0].fill(0);
    for i in 1..n {
        let c = (words[i] as u8 - b'A') as i32;
        let p = (words[i - 1] as u8 - b'A') as i32;
        let d = distance(c, p);
        for j in 0..26 {
            dp[i][j] = dp[i][j].min(dp[i - 1][j] + d);
            dp[i][p as usize] = dp[i][p as usize].min(dp[i - 1][j] + distance(j as i32, c));
        }
    }
    return *dp.last().unwrap().iter().min().unwrap();
}

/// 1323
pub fn maximum69_number(num: i32) -> i32 {
    let chars = num.to_string().chars().collect::<Vec<char>>();
    let mut changed = false;
    let mut ans = 0;
    for i in 0..chars.len() {
        match chars[i] {
            '6' => {
                if !changed {
                    ans = ans * 10 + 9;
                    changed = true;
                } else {
                    ans = ans * 10 + 6;
                }
            }
            '9' => {
                ans = ans * 10 + 9;
            }
            _ => {}
        }
    }
    ans
}

/// 1324
pub fn print_vertically(s: String) -> Vec<String> {
    let vec: Vec<&str> = s.split(" ").collect();
    let max_len = vec.iter().map(|s| s.len()).max().unwrap();
    let mut ans = vec![String::with_capacity(vec.len()); max_len];
    let mut ids = vec![0; vec.len()];
    for i in 0..vec.len() {
        for j in 0..max_len {
            let id = ids[i];
            let s;
            if id >= vec[i].len() {
                s = " ";
            } else {
                s = &vec[i][id..id + 1];
            }
            ans[j].push_str(s);
            ids[i] += 1;
        }
    }
    let ans: Vec<String> = ans.iter().map(|s| String::from(s.trim_end())).collect();
    ans
}

/// 1325
pub fn remove_leaf_nodes(
    root: Option<Rc<RefCell<TreeNode>>>,
    target: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    if let Some(node) = root {
        let left = remove_leaf_nodes(node.borrow_mut().left.take(), target);
        let right = remove_leaf_nodes(node.borrow_mut().right.take(), target);
        if left.is_none() && right.is_none() && node.borrow().val == target {
            return None;
        } else {
            node.borrow_mut().left = left;
            node.borrow_mut().right = right;
            return Some(node);
        }
    }
    None
}

/// 1326
pub fn min_taps(n: i32, ranges: Vec<i32>) -> i32 {
    let mut right_most = vec![0; n as usize + 1];
    let mut ans = 0;

    for (i, r) in ranges.iter().enumerate() {
        let left = (i as i32 - r).max(0);
        right_most[left as usize] = right_most[left as usize].max(i as i32 + r);
    }

    let mut curr_right = 0;
    let mut next_right = 0;
    for i in 0..n {
        next_right = next_right.max(right_most[i as usize]);
        if next_right <= i {
            return -1;
        }
        if curr_right == i {
            curr_right = next_right;
            ans += 1;
        }
    }

    ans
}

/// 1327
pub fn break_palindrome(palindrome: String) -> String {
    let n = palindrome.len();
    if n == 1 {
        return "".to_string();
    }
    let mut data: Vec<char> = palindrome.chars().collect();
    for i in 0..(n / 2) {
        if data[i] != 'a' {
            data[i] = 'a';
            return data.iter().collect();
        }
    }
    data[n - 1] = 'b';
    data.iter().collect()
}

/// 1329
pub fn diagonal_sort(mut mat: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let n = mat.len();
    let m = mat[0].len();
    let mut diag = vec![vec![]; m + n];
    for i in 0..n {
        for j in 0..m {
            diag[i - j + m].push(mat[i][j]);
        }
    }
    for d in diag.iter_mut() {
        d.sort_by(|a, b| b.cmp(a));
    }
    for i in 0..n {
        for j in 0..m {
            mat[i][j] = diag[i - j + m].pop().unwrap();
        }
    }
    mat
}

/// 1330
pub fn max_value_after_reverse(nums: Vec<i32>) -> i32 {
    let mut value = 0;
    let n = nums.len();
    for i in 0..n - 1 {
        value += i32::abs(nums[i] - nums[i + 1]);
    }
    let mut mx1 = 0;
    for i in 1..n - 1 {
        mx1 = i32::max(
            mx1,
            i32::abs(nums[0] - nums[i + 1]) - i32::abs(nums[i] - nums[i + 1]),
        );
        mx1 = i32::max(
            mx1,
            i32::abs(nums[n - 1] - nums[i - 1]) - i32::abs(nums[i] - nums[i - 1]),
        );
    }
    let mut mx2 = -100_000;
    let mut mn2 = 1000_000;
    for i in 0..n - 1 {
        mx2 = i32::max(mx2, i32::min(nums[i], nums[i + 1]));
        mn2 = i32::min(mn2, i32::max(nums[i], nums[i + 1]));
    }

    value + i32::max(mx1, 2 * (mx2 - mn2))
}

/// 1331
pub fn array_rank_transform(arr: Vec<i32>) -> Vec<i32> {
    if arr.is_empty() {
        return vec![];
    }
    let mut rank = arr
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i))
        .collect::<Vec<_>>();
    rank.sort();
    let mut ret = vec![0; arr.len()];
    ret[rank[0].1] = 1;
    rank.windows(2)
        .for_each(|v| ret[v[1].1] = ret[v[0].1] + if v[0].0 == v[1].0 { 0 } else { 1 });
    ret
}

/// 1332
pub fn remove_palindrome_sub(s: String) -> i32 {
    let chars = s.chars().collect::<Vec<char>>();
    let (mut i, mut j) = (0, chars.len());
    while i < j {
        if chars[i] == chars[j] {
            i = i + 1;
            j = j - 1;
        } else {
            return 2;
        }
    }
    1
}

/// 1333
pub fn filter_restaurants(
    restaurants: Vec<Vec<i32>>,
    vegan_friendly: i32,
    max_price: i32,
    max_distance: i32,
) -> Vec<i32> {
    let mut filter_res: Vec<Vec<i32>> = restaurants
        .iter()
        .filter(|re| {
            if vegan_friendly == 1 && re[2] != 1 {
                return false;
            }
            if max_distance < re[4] || max_price < re[3] {
                return false;
            }
            true
        })
        .map(|re| re.clone())
        .collect();

    filter_res.sort_by(|a, b| {
        let cmp_result = b[1].cmp(&a[1]);
        if cmp_result == std::cmp::Ordering::Equal {
            b[0].cmp(&a[0])
        } else {
            cmp_result
        }
    });
    filter_res.into_iter().map(|a| a[0]).collect()
}

/// 1334
pub fn find_the_city(n: i32, edges: Vec<Vec<i32>>, distance_threshold: i32) -> i32 {
    let n = n as usize;
    let (mut w, mut f) = (
        vec![vec![i32::MAX / 2; n]; n],
        vec![vec![vec![0; n]; n]; n + 1],
    );
    edges.iter().for_each(|e| {
        let (x, y, wt) = (e[0] as usize, e[1] as usize, e[2]);
        w[x][y] = wt;
        w[y][x] = wt;
    });
    f[0] = w;
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                f[k + 1][i][j] = f[k][i][j].min(f[k][i][k] + f[k][k][j]);
            }
        }
    }

    let (mut ans, mut min_cnt) = (0, n);
    for i in 0..n {
        let mut cnt = 0;
        for j in 0..n {
            if j != i && f[n][i][j] <= distance_threshold {
                cnt += 1;
            }
        }
        if cnt <= min_cnt {
            min_cnt = cnt;
            ans = i;
        }
    }
    ans as _
}

/// 1335
pub fn min_difficulty(job_difficulty: Vec<i32>, d: i32) -> i32 {
    let d = d as usize;
    if job_difficulty.len() < d {
        return -1;
    }
    let n = job_difficulty.len();
    let mut dp = vec![vec![i32::MAX; n + 1]; d + 1];

    dp[0][0] = 0;
    for i in 0..d {
        for j in i..n {
            let mut job = job_difficulty[j];
            for k in j + 1..n + 1 {
                dp[i + 1][k] = dp[i + 1][k].min(dp[i][j].saturating_add(job));
                if k < n {
                    job = job.max(job_difficulty[k]);
                }
            }
        }
    }
    dp[d][n]
}

/// 1337
pub fn k_weakest_rows(mat: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
    let mut s = mat
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.iter().sum()))
        .collect::<Vec<(usize, i32)>>();
    s.sort_by(|a, b| a.1.cmp(&b.1));
    s.iter()
        .map(|k| k.0 as i32)
        .take(k as usize)
        .collect::<Vec<i32>>()
}

/// 1338
pub fn min_set_size(arr: Vec<i32>) -> i32 {
    let mut cnt = arr
        .iter()
        .fold(HashMap::new(), |mut acc, &cur| {
            *acc.entry(cur).or_insert(0) += 1;
            acc
        })
        .into_iter()
        .collect::<Vec<(i32, usize)>>();
    cnt.sort_by(|a, b| b.1.cmp(&a.1));
    let (mut total, mut cur_size) = (0, 0);
    for c in cnt {
        total = total + 1;
        cur_size = cur_size + c.1;
        if cur_size * 2 >= arr.len() {
            break;
        }
    }
    total
}

/// 1339
pub fn max_product(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut res = 0;
    if let Some(inner) = root {
        let mut stack = vec![(inner.clone(), false)];
        while let Some((node, flag)) = stack.pop() {
            if flag {
                let mut sum = 0;
                if let Some(leaf) = node.borrow().right.clone() {
                    sum += leaf.borrow().val;
                }
                if let Some(leaf) = node.borrow().left.clone() {
                    sum += leaf.borrow().val;
                }
                node.borrow_mut().val += sum;
            } else {
                stack.push((node.clone(), true));
                if let Some(leaf) = node.borrow().right.clone() {
                    stack.push((leaf, false));
                }
                if let Some(leaf) = node.borrow().left.clone() {
                    stack.push((leaf, false));
                }
            }
        }
        let sum = inner.clone().borrow().val;
        let mut stack = vec![inner];
        while let Some(node) = stack.pop() {
            let val = node.borrow().val;
            let diff = sum - 2 * val;
            res = res.max(val as i64 * (sum - val) as i64);
            if diff.abs() <= 1 {
                break;
            } else if diff < -1 {
                if let Some(leaf) = node.borrow().right.clone() {
                    stack.push(leaf);
                }
                if let Some(leaf) = node.borrow().left.clone() {
                    stack.push(leaf);
                }
            }
        }
    }
    (res % (10i64.pow(9) + 7)) as i32
}

/// 1340
/// timeout
pub fn max_jumps(arr: Vec<i32>, d: i32) -> i32 {
    let mut pairs = arr.into_iter().enumerate().collect::<Vec<(usize, i32)>>();
    pairs.sort_by(|a, b| a.1.cmp(&b.1));
    let mut dp = vec![1; pairs.len()];
    for i in 0..pairs.len() {
        for j in 0..i {
            let barrier = pairs.iter().find(|p| {
                p.0 > pairs[i].0.min(pairs[j].0)
                    && p.0 < pairs[i].0.max(pairs[j].0)
                    && p.1 >= pairs[i].1
            });
            if barrier.is_none()
                && (pairs[i].0 as i32 - pairs[j].0 as i32).abs() <= d
                && pairs[i].1 > pairs[j].1
            {
                dp[i] = dp[i].max(dp[j] + 1);
            }
        }
    }
    dp.into_iter().max().unwrap()
}

/// 1342
pub fn number_of_steps(mut num: i32) -> i32 {
    let mut step = 0;
    while num != 0 {
        if num % 2 == 0 {
            num = num / 2;
        } else {
            num = num - 1;
        }
        step = step + 1;
    }
    step
}

/// 1343
pub fn num_of_subarrays(arr: Vec<i32>, k: i32, threshold: i32) -> i32 {
    let k = k as usize;
    let total = k as i32 * threshold;
    let (mut sum, mut cnt) = (0, 0);
    for i in 0..k as usize {
        sum = sum + arr[i]
    }
    if sum >= total {
        cnt = cnt + 1;
    }
    for j in k..arr.len() {
        sum = sum - arr[j - k];
        sum = sum + arr[j];
        if sum >= total {
            cnt = cnt + 1;
        }
    }
    cnt
}

/// 1344
pub fn angle_clock(h: i32, m: i32) -> f64 {
    let tmp: f64 = (30_f64 * h as f64 - 5.5_f64 * m as f64).abs();
    if tmp > 180_f64 { 360_f64 - tmp } else { tmp }
}

/// 1345 timeout
pub fn min_jumps(arr: Vec<i32>) -> i32 {
    fn bfs(arr: &Vec<i32>, min_steps: &mut Vec<i32>, start: usize, next: &mut Vec<usize>) {
        let height = arr[start];
        let same_height = arr
            .iter()
            .enumerate()
            .filter(|(i, h)| **h == height && *i != start && min_steps[*i] == i32::MAX)
            .map(|v| v.0)
            .collect::<Vec<usize>>();
        same_height.iter().for_each(|&i| {
            min_steps[i] = min_steps[start] + 1;
            next.push(i);
        });
        if start - 1 < arr.len() && min_steps[start - 1] == i32::MAX {
            min_steps[start - 1] = min_steps[start] + 1;
            next.push(start - 1);
        }
        if start + 1 < arr.len() && min_steps[start + 1] == i32::MAX {
            min_steps[start + 1] = min_steps[start] + 1;
            next.push(start + 1);
        }
    }
    let mut min_steps = vec![i32::MAX; arr.len()];
    min_steps[0] = 0;
    let mut next = vec![0];
    while next.len() != 0 {
        for _ in 0..next.len() {
            bfs(&arr, &mut min_steps, next.remove(0), &mut next);
        }
    }
    min_steps[arr.len() - 1]
}

/// 1346
pub fn check_if_exist(arr: Vec<i32>) -> bool {
    let mut record = HashSet::new();
    for i in arr {
        if record.contains(&(i * 2)) || (i % 2 == 0 && record.contains(&(i / 2))) {
            return true;
        }
        record.insert(i);
    }
    return false;
}

/// 1347
pub fn min_steps(s: String, t: String) -> i32 {
    let mut cnt = s.chars().fold(HashMap::new(), |mut acc, cur| {
        *acc.entry(cur).or_insert(0) += 1;
        acc
    });
    t.chars().for_each(|c| {
        cnt.entry(c).and_modify(|e| {
            if *e > 0 {
                *e = *e - 1
            }
        });
    });
    cnt.iter().fold(0, |acc, cur| acc + cur.1)
}

/// 1349
pub fn max_students(seats: Vec<Vec<char>>) -> i32 {
    let m = seats.len();
    let n = seats[0].len();
    let mut a = vec![0; m];
    for (i, row) in seats.iter().enumerate() {
        for (j, &c) in row.iter().enumerate() {
            if c == '.' {
                a[i] |= 1 << j;
            }
        }
    }

    let mut memo = vec![vec![-1; 1 << n]; m];
    fn dfs(i: usize, j: usize, memo: &mut Vec<Vec<i32>>, a: &Vec<usize>) -> i32 {
        if memo[i][j] != -1 {
            return memo[i][j];
        }
        if i == 0 {
            if j == 0 {
                return 0;
            }
            let lb = (j as i32 & -(j as i32)) as usize;
            memo[i][j] = dfs(i, j & !(lb * 3), memo, a) + 1;
            return memo[i][j];
        }
        let mut res = dfs(i - 1, a[i - 1], memo, a);
        let mut s = j;
        while s > 0 {
            if (s & (s >> 1)) == 0 {
                let t = a[i - 1] & !(s << 1 | s >> 1);
                res = res.max(dfs(i - 1, t, memo, a) + s.count_ones() as i32);
            }
            s = (s - 1) & j;
        }
        memo[i][j] = res;
        res
    }
    dfs(m - 1, a[m - 1], &mut memo, &a)
}

/// 1351
pub fn count_negatives(grid: Vec<Vec<i32>>) -> i32 {
    let width = grid[0].len();
    let mut ans = 0;
    'a: for r in grid {
        for (i, &c) in r.iter().enumerate() {
            if c < 0 {
                ans = ans + width - i;
                continue 'a;
            }
        }
    }
    ans as i32
}

/// 1352
struct ProductOfNumbers {
    data: Vec<i32>,
}

impl ProductOfNumbers {
    fn new() -> Self {
        Self { data: vec![] }
    }

    fn add(&mut self, num: i32) {
        self.data.push(self.data.last().unwrap_or(&1) * num);
    }

    fn get_product(&self, k: i32) -> i32 {
        self.data.last().unwrap_or(&1) / self.data.get(self.data.len() - k as usize).unwrap_or(&1)
    }
}

/// 1353
pub fn max_events(events: Vec<Vec<i32>>) -> i32 {
    let mut mx = 0;
    for e in &events {
        mx = mx.max(e[1]);
    }

    let mut groups = vec![vec![]; (mx + 1) as usize];
    for e in events {
        groups[e[0] as usize].push(e[1]);
    }

    let mut ans = 0;
    let mut h = BinaryHeap::<i32>::new();
    for (i, g) in groups.into_iter().enumerate() {
        while let Some(end_day) = h.peek() {
            if -end_day >= i as i32 {
                break;
            }
            h.pop();
        }
        for end_day in g {
            h.push(-end_day);
        }
        if let Some(_) = h.pop() {
            ans += 1;
        }
    }
    ans
}

/// 1354
pub fn is_possible(target: Vec<i32>) -> bool {
    let mut sum = target.iter().map(|&x| x as i64).sum();
    let mut pq = BinaryHeap::from(target);

    while *pq.peek().unwrap() > 1 {
        let x = pq.pop().unwrap() as i64;
        sum -= x;

        if sum == 0 || x <= sum {
            return false;
        }
        let x = (x - 1) % sum + 1;
        sum += x;
        pq.push(x as i32);
    }

    true
}

/// 1356
pub fn sort_by_bits(mut arr: Vec<i32>) -> Vec<i32> {
    arr.sort_by(|a, b| match a.count_ones().cmp(&b.count_ones()) {
        Ordering::Equal => a.cmp(&b),
        Ordering::Greater => Ordering::Greater,
        Ordering::Less => Ordering::Less,
    });
    arr
}

/// 1357
struct Cashier {
    n: i32,
    now_customer: i32,
    discount: i32,
    price: std::collections::HashMap<i32, i32>,
}

impl Cashier {
    fn new(n: i32, discount: i32, products: Vec<i32>, prices: Vec<i32>) -> Self {
        Self {
            n,
            now_customer: 0,
            discount,
            price: products.into_iter().zip(prices.into_iter()).collect(),
        }
    }
    fn get_bill(&mut self, product: Vec<i32>, amount: Vec<i32>) -> f64 {
        self.now_customer += 1;
        let pay = product
            .iter()
            .zip(amount.iter())
            .fold(0, |acc, (product_id, this_amo)| {
                acc + self.price[&product_id] * this_amo
            });
        if self.now_customer % self.n == 0 {
            (pay as f64) * (1.0 - self.discount as f64 / 100.0)
        } else {
            pay as f64
        }
    }
}

/// 1358
pub fn number_of_substrings(s: String) -> i32 {
    let (chars, mut total, mut l) = (s.as_bytes(), 0, 0);
    let mut cnt = vec![0; 3];
    for r in 0..chars.len() {
        cnt[(chars[r] - b'a') as usize] += 1;
        while cnt[0] > 0 && cnt[1] > 0 && cnt[2] > 0 {
            cnt[(chars[l] - b'a') as usize] -= 1;
            l = l + 1;
        }
        total += l;
    }
    total as i32
}

/// 1360
pub fn days_between_dates(date1: String, date2: String) -> i32 {
    fn zeller_days(year: i32, month: i32, day: i32) -> i32 {
        let mut y = year;
        let mut m = month;
        let d = day;
        if m < 3 {
            y -= 1;
            m += 12;
        }
        (y - 1) * 365 + y / 4 - y / 100 + y / 400 + (m - 1) * 28 + 13 * (m + 1) / 5 - 7 + d
    }

    let dv1: Vec<i32> = date1
        .split('-')
        .map(|s| s.parse::<i32>().unwrap())
        .collect();
    let dv2: Vec<i32> = date2
        .split('-')
        .map(|s| s.parse::<i32>().unwrap())
        .collect();
    (zeller_days(dv2[0], dv2[1], dv2[2]) - zeller_days(dv1[0], dv1[1], dv1[2])).abs()
}

/// 1361
pub fn validate_binary_tree_nodes(n: i32, left_child: Vec<i32>, right_child: Vec<i32>) -> bool {
    let n = n as usize;
    let mut in_deg = vec![0; n];
    left_child
        .iter()
        .zip(right_child.iter())
        .for_each(|(&l, &r)| {
            if l != -1 {
                in_deg[l as usize] = in_deg[l as usize] + 1;
            }
            if r != -1 {
                in_deg[r as usize] = in_deg[r as usize] + 1;
            }
        });
    let root = in_deg.iter().enumerate().find(|(_, deg)| **deg == 0);
    if let Some((node, _)) = root {
        let mut visited = HashSet::new();
        visited.insert(node);
        let mut queue = VecDeque::new();
        queue.push_back(node);
        while !queue.is_empty() {
            let last = queue.pop_front().unwrap();
            if left_child[last] != -1 {
                if visited.contains(&(left_child[last] as usize)) {
                    return false;
                } else {
                    visited.insert(left_child[last] as usize);
                    queue.push_back(left_child[last] as usize);
                }
            }
            if right_child[last] != -1 {
                if visited.contains(&(right_child[last] as usize)) {
                    return false;
                } else {
                    visited.insert(right_child[last] as usize);
                    queue.push_back(right_child[last] as usize);
                }
            }
        }
        return visited.len() == n;
    } else {
        false
    }
}

/// 1362
pub fn closest_divisors(num: i32) -> Vec<i32> {
    [num + 1, num + 2]
        .iter()
        .fold(vec![], |mut acc, &target| {
            for i in 1..target.isqrt() + 1 {
                if target % i == 0 {
                    acc.push([i, target / i]);
                }
            }
            acc
        })
        .iter()
        .fold(vec![0, i32::MAX], |acc, cur| {
            if (cur[0] - cur[1]).abs() < (acc[0] - acc[1]).abs() {
                vec![cur[0], cur[1]]
            } else {
                acc
            }
        })
}

/// 1363
pub fn largest_multiple_of_three(digits: Vec<i32>) -> String {
    let mut digits = digits;
    digits.sort();
    let tot: i32 = digits.iter().sum();
    let mut modify_digits = |miss: i32| -> bool {
        let c1 = miss + 1;
        let c2 = (1 ^ miss) + 1;
        let mut found = false;
        for (index, num) in digits.iter_mut().enumerate() {
            if *num % 3 == c1 {
                digits.remove(index);
                return true;
            }
        }
        let mut pos1 = -1;
        let mut pos2 = -1;
        for (index, num) in digits.iter_mut().enumerate() {
            if *num % 3 == c2 {
                if pos1 >= 0 {
                    pos2 = index as i32;
                    found = true;
                    break;
                } else {
                    pos1 = index as i32;
                }
            }
        }
        if found {
            digits.remove(pos2 as usize);
            digits.remove(pos1 as usize);
            return true;
        } else {
            return false;
        }
    };
    if tot % 3 == 1 {
        if !modify_digits(0) {
            return "".to_string();
        }
    } else if tot % 3 == 2 {
        if !modify_digits(1) {
            return "".to_string();
        }
    }
    if digits.is_empty() {
        return "".to_string();
    }
    while *digits.last().unwrap() == 0 && digits.len() >= 2 {
        digits.pop();
    }
    String::from_utf8(
        digits
            .iter_mut()
            .map(|digit: &mut i32| -> u8 {
                return *digit as u8 + b'0';
            })
            .rev()
            .collect::<Vec<u8>>(),
    )
    .unwrap()
}

/// 1365
pub fn smaller_numbers_than_current(nums: Vec<i32>) -> Vec<i32> {
    let mut cnt = vec![0; 101];
    nums.iter().for_each(|&n| {
        cnt[n as usize] = cnt[n as usize] + 1;
    });
    for i in 1..cnt.len() {
        cnt[i] = cnt[i - 1] + cnt[i];
    }
    nums.iter().fold(vec![], |mut acc, &cur| {
        if cur == 0 {
            acc.push(0);
        } else {
            acc.push(cnt[cur as usize - 1])
        }
        acc
    })
}

/// 1366
pub fn rank_teams(votes: Vec<String>) -> String {
    let _ = votes.len();
    let mut ranking: HashMap<char, Vec<i32>> = HashMap::new();
    for vid in votes[0].chars() {
        ranking.entry(vid).or_insert(vec![0; votes[0].len()]);
    }
    for vote in votes {
        for (i, c) in vote.chars().enumerate() {
            if let Some(rank) = ranking.get_mut(&c) {
                rank[i] += 1;
            }
        }
    }
    let mut result: Vec<(char, Vec<i32>)> = ranking.into_iter().collect();
    result.sort_by(|a, b| {
        for i in 0..a.1.len() {
            if a.1[i] != b.1[i] {
                return b.1[i].cmp(&a.1[i]);
            }
        }
        a.0.cmp(&b.0)
    });

    let mut ans = String::new();
    for (vid, _) in result {
        ans.push(vid);
    }
    ans
}

/// 1367
pub fn is_sub_path(head: Option<Box<ListNode>>, root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn dfs(
        head: &Option<Box<ListNode>>,
        l: &Option<Box<ListNode>>,
        t: &Option<Rc<RefCell<TreeNode>>>,
    ) -> bool {
        if l.is_none() {
            return true;
        }
        if let Some(node) = t {
            let node = node.borrow();
            node.val == l.as_ref().unwrap().val
                && (dfs(head, &l.as_ref().unwrap().next, &node.left)
                    || dfs(head, &l.as_ref().unwrap().next, &node.right))
                || l == head && (dfs(head, head, &node.left) || dfs(head, head, &node.right))
        } else {
            false
        }
    }
    dfs(&head, &head, &root)
}

/// 1368
pub fn min_cost(grid: Vec<Vec<i32>>) -> i32 {
    let n = grid.len();
    let m = grid[0].len();
    let mut costs = vec![vec![i32::MAX; m]; n];
    let mut seen = vec![vec![false; m]; n];
    let mut queue = BinaryHeap::new();

    costs[0][0] = 0;
    queue.push(Reverse((0, 0, 0)));

    while let Some(Reverse((cost, i, j))) = queue.pop() {
        if seen[i][j] {
            continue;
        }
        seen[i][j] = true;

        if i == n - 1 && j == m - 1 {
            return costs[i][j];
        }

        let cost1 = if grid[i][j] == 4 { 0 } else { 1 };
        if i > 0 && cost + cost1 < costs[i - 1][j] {
            costs[i - 1][j] = cost + cost1;
            queue.push(Reverse((costs[i - 1][j], i - 1, j)));
        }

        let cost2 = if grid[i][j] == 3 { 0 } else { 1 };
        if i < n - 1 && cost + cost2 < costs[i + 1][j] {
            costs[i + 1][j] = cost + cost2;
            queue.push(Reverse((costs[i + 1][j], i + 1, j)));
        }

        let cost3 = if grid[i][j] == 2 { 0 } else { 1 };
        if j > 0 && cost + cost3 < costs[i][j - 1] {
            costs[i][j - 1] = cost + cost3;
            queue.push(Reverse((costs[i][j - 1], i, j - 1)));
        }

        let cost4 = if grid[i][j] == 1 { 0 } else { 1 };
        if j < m - 1 && cost + cost4 < costs[i][j + 1] {
            costs[i][j + 1] = cost + cost4;
            queue.push(Reverse((costs[i][j + 1], i, j + 1)));
        }
    }

    costs[n - 1][m - 1]
}

/// 1370
pub fn sort_string(s: String) -> String {
    let mut cnt = vec![0; 26];
    s.chars().for_each(|c| {
        cnt[(c as usize).abs_diff(b'a' as usize)] += 1;
    });
    let mut ans = "".to_string();
    while ans.len() < s.len() {
        (0..26).for_each(|i| {
            if cnt[i] > 0 {
                ans.push((i as u8 + b'a') as char);
                cnt[i] -= 1;
            }
        });
        (0..26).rev().for_each(|i| {
            if cnt[i] > 0 {
                ans.push((i as u8 + b'a') as char);
                cnt[i] -= 1;
            }
        });
    }
    ans
}

/// 1371
pub fn find_the_longest_substring(s: String) -> i32 {
    let (mut ans, mut t, mut d) = (0, 0, [50001; 32]);
    d[0] = -1;
    for (i, c) in s.char_indices() {
        match c {
            'a' => t ^= 1,
            'e' => t ^= 2,
            'i' => t ^= 4,
            'o' => t ^= 8,
            'u' => t ^= 16,
            _ => (),
        }
        match d[t] {
            dt if dt != 50001 => ans = ans.max(i as i32 - dt),
            _ => d[t] = i as i32,
        }
    }
    ans
}

/// 1372
pub fn longest_zig_zag(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(node: &Rc<RefCell<TreeNode>>, max_length: &mut i32) -> (i32, i32) {
        let (mut left, mut right) = (0, 0);
        if let Some(left_node) = node.borrow().left.as_ref() {
            left = dfs(left_node, max_length).1 + 1;
            *max_length = (*max_length).max(left);
        }
        if let Some(right_node) = node.borrow().right.as_ref() {
            right = dfs(right_node, max_length).0 + 1;
            *max_length = (*max_length).max(right);
        }
        (left, right)
    }
    if let Some(node) = root.as_ref() {
        let mut max_length = 0;
        dfs(node, &mut max_length);
        max_length
    } else {
        0
    }
}

/// 1373
pub fn max_sum_bst(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        if root.is_none() {
            return vec![i32::MAX, i32::MIN, 0, i32::MIN];
        };
        let (left, right) = (
            dfs(&root.as_ref().unwrap().borrow().left),
            dfs(&root.as_ref().unwrap().borrow().right),
        );
        let val = root.as_ref().unwrap().borrow().val;
        if val <= left[1] || val >= right[0] {
            return vec![
                i32::MIN,
                i32::MAX,
                left[2].max(right[2]),
                left[3].max(right[3]),
            ];
        }
        let (min, max) = (left[0].min(val), right[1].max(val));
        let sum = left[2] + right[2] + val;
        let max_sum = sum.max(left[3].max(right[3]));
        vec![min, max, sum, max_sum]
    }
    let ret = dfs(&root);
    ret[3].max(0)
}

/// 1374
pub fn generate_the_string(n: i32) -> String {
    let n = n as usize;
    match n % 2 {
        1 => (0..n).fold("".to_string(), |mut acc, _| {
            acc.push('a');
            acc
        }),
        0 => {
            let mut s = (0..n - 1).fold("".to_string(), |mut acc, _| {
                acc.push('a');
                acc
            });
            s.push('b');
            s
        }
        _ => "".to_string(),
    }
}

/// 1375
pub fn num_times_all_blue(flips: Vec<i32>) -> i32 {
    let (mut ans, mut max_digit) = (0, 0);
    flips.iter().enumerate().for_each(|(i, &f)| {
        max_digit = max_digit.max(f as usize);
        if max_digit == i + 1 {
            ans += 1;
        }
    });
    ans
}

/// 1376
pub fn num_of_minutes(n: i32, head_id: i32, manager: Vec<i32>, inform_time: Vec<i32>) -> i32 {
    let (n, head_id) = (n as usize, head_id as usize);
    let (mut ans, mut queue, mut graph) = (0, vec![], vec![vec![]; n]);
    manager.iter().enumerate().for_each(|(i, &m)| {
        if m != -1 {
            graph[m as usize].push(i);
        }
    });
    queue.push((head_id, 0));
    while let Some(cur) = queue.pop() {
        ans = ans.max(cur.1 as usize);
        graph[cur.0]
            .iter()
            .for_each(|&c| queue.push((c, cur.1 + inform_time[cur.0])));
    }
    ans as i32
}

/// 1377
pub fn frog_position(n: i32, edges: Vec<Vec<i32>>, t: i32, target: i32) -> f64 {
    let (n, target) = (n as usize, target as usize);
    let (mut graph, mut visited) = (vec![vec![]; n + 1], vec![false; n + 1]);
    edges.iter().for_each(|e| {
        graph[e[0] as usize].push(e[1] as usize);
        graph[e[1] as usize].push(e[0] as usize);
    });
    fn dfs(g: &Vec<Vec<usize>>, v: &mut Vec<bool>, e: usize, t: i32, target: usize) -> f64 {
        let next = if e == 1 { g[e].len() } else { g[e].len() - 1 };
        if t == 0 || next == 0 {
            return if e == target { 1.0 } else { 0.0 };
        }
        v[e] = true;
        let mut ans = 0.0;
        g[e].iter().for_each(|&i| {
            if v[i] == false {
                ans += dfs(g, v, i, t - 1, target);
            }
        });
        return ans / next as f64;
    }
    dfs(&graph, &mut visited, 1, t, target)
}

/// 1380
pub fn lucky_numbers(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let mut ret = Vec::new();
    let (m, n) = (matrix.len(), matrix[0].len());
    for i in 0..m {
        let mut min = matrix[i][0];
        let mut min_idx = 0;
        for j in 1..n {
            if min > matrix[i][j] {
                min = matrix[i][j];
                min_idx = j;
            }
        }
        let mut max = min;
        for j in 0..m {
            max = max.max(matrix[j][min_idx]);
        }
        if min == max {
            ret.push(max);
        }
    }
    ret
}

/// 1381
struct CustomStack {
    data: Vec<i32>,
    max_size: usize,
    cur_size: usize,
}

impl CustomStack {
    fn new(max_size: i32) -> Self {
        Self {
            data: vec![],
            max_size: max_size as usize,
            cur_size: 0,
        }
    }

    fn push(&mut self, x: i32) {
        if self.cur_size < self.max_size {
            self.data.push(x);
            self.cur_size += 1;
        }
    }

    fn pop(&mut self) -> i32 {
        if self.cur_size > 0 {
            self.cur_size -= 1;
            self.data.pop().unwrap()
        } else {
            -1
        }
    }

    fn increment(&mut self, k: i32, val: i32) {
        let n = (k as usize).min(self.cur_size);
        for i in 0..n {
            self.data[i] += val;
        }
    }
}

/// 1382
pub fn balance_bst(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    let mut vals = vec![];
    fn in_order(root: Option<Rc<RefCell<TreeNode>>>, vals: &mut Vec<i32>) {
        if let Some(node) = root.as_ref() {
            in_order(node.borrow().left.clone(), vals);
            vals.push(node.borrow().val);
            in_order(node.borrow().right.clone(), vals);
        }
    }
    in_order(root, &mut vals);
    fn build(l: usize, r: usize, vals: &Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        if l <= r {
            let mid = (l + r) / 2;
            let node = Rc::new(RefCell::new(TreeNode::new(vals[mid])));
            if mid != l {
                node.borrow_mut().left = build(l, mid - 1, vals);
            }
            if mid != r {
                node.borrow_mut().right = build(mid + 1, r, vals);
            }
            Some(node)
        } else {
            None
        }
    }
    build(0, vals.len() - 1, &vals)
}

/// 1383
pub fn max_performance(speed: Vec<i32>, efficiency: Vec<i32>, k: i32) -> i32 {
    struct Engineer {
        speed: i32,
        efficiency: i32,
    }
    const M: u64 = 1e9 as u64 + 7;

    let mut engineers: Vec<Engineer> = speed
        .iter()
        .zip(efficiency.iter())
        .map(|(s, e)| Engineer {
            speed: *s,
            efficiency: *e,
        })
        .collect();
    engineers.sort_by_key(|e| Reverse(e.efficiency));
    let mut max_performance = 0;
    let mut min_heap = BinaryHeap::new();
    for engineer in engineers.iter() {
        min_heap.push(Reverse(engineer.speed));
        if min_heap.len() > k as usize {
            min_heap.pop();
        }
        let speeds: u64 = min_heap.iter().map(|Reverse(i)| *i as u64).sum();
        let perf: u64 = speeds as u64 * engineer.efficiency as u64;
        max_performance = max_performance.max(perf)
    }

    (max_performance % M) as i32
}

/// 1385
pub fn find_the_distance_value(arr1: Vec<i32>, arr2: Vec<i32>, d: i32) -> i32 {
    arr1.iter().fold(0, |mut acc, c1| {
        if arr2.iter().all(|&c2| c1.abs_diff(c2) > (d as u32)) {
            acc = acc + 1
        };
        acc
    })
}

/// 1386
pub fn max_number_of_families(n: i32, reserved_seats: Vec<Vec<i32>>) -> i32 {
    use std::collections::HashMap;
    let mut hash = HashMap::<i32, i32>::new();

    for seat in reserved_seats {
        let x = seat[0];
        let y = seat[1];
        if y != 1 && y != 10 {
            hash.entry(x)
                .and_modify(|bits| *bits |= 1 << (y - 2))
                .or_insert(1 << (y - 2));
        }
    }
    let mut res = (n - hash.len() as i32) * 2;

    let left = 0b11110000;
    let right = 0b00001111;
    let mid = 0b00111100;
    for &s in hash.values() {
        if !(s & left).is_positive() || !(s & mid).is_positive() || !(s & right).is_positive() {
            res += 1;
        }
    }
    res
}

/// 1387
pub fn get_kth(lo: i32, hi: i32, k: i32) -> i32 {
    let (lo, hi, k) = (lo as usize, hi as usize, k as usize);
    let mut with_weight = (lo..=hi)
        .map(|val| {
            let (mut v, mut w) = (val, 0);
            while v != 1 {
                if v % 2 == 0 {
                    v = v / 2;
                } else {
                    v = v * 3 + 1;
                }
                w = w + 1;
            }
            (val, w)
        })
        .collect::<Vec<(usize, usize)>>();
    with_weight.sort_by(|a, b| a.1.cmp(&b.1));
    with_weight[k - 1].0 as i32
}

/// 1388
pub fn max_size_slices(slices: Vec<i32>) -> i32 {
    fn calc(s: &[i32]) -> i32 {
        let mut pre = vec![0; s.len() + 2];
        for _ in 0..s.len() / 3 + 1 {
            let mut cur = vec![0; s.len() + 2];
            for i in 2..pre.len() {
                cur[i] = cur[i - 1].max(pre[i - 2] + s[i - 2]);
            }
            pre = cur;
        }
        pre[s.len() + 1]
    }
    calc(&slices[0..slices.len() - 1]).max(calc(&slices[1..]))
}

/// 1389
pub fn create_target_array(nums: Vec<i32>, index: Vec<i32>) -> Vec<i32> {
    let mut target = vec![i32::MAX; nums.len()];
    nums.iter().zip(index).for_each(|(&n, i)| {
        let i = i as usize;
        if target[i] == i32::MAX {
            target[i] = n;
        } else {
            target.insert(i, n);
        }
    });
    target[0..nums.len()].to_vec()
}

/// 1390
pub fn sum_four_divisors(nums: Vec<i32>) -> i32 {
    let mut ans = 0;
    nums.iter().for_each(|&n| {
        let (mut cnt, mut sum) = (0, 0);
        let mut i = 1;
        while i * i <= n {
            if n % i == 0 {
                cnt = cnt + 1;
                sum = sum + i;
                if i * i != n {
                    cnt = cnt + 1;
                    sum = sum + n / i;
                }
            }
            i = i + 1;
        }
        if cnt == 4 {
            ans = ans + sum;
        }
    });
    ans
}

/// 1391
pub fn has_valid_path(grid: Vec<Vec<i32>>) -> bool {
    let n = grid.len();
    let m = grid[0].len();
    let mut vis = vec![vec![false; m]; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((0, 0));

    while let Some((i, j)) = queue.pop_front() {
        vis[i][j] = true;

        if vis[n - 1][m - 1] {
            return true;
        }

        if i > 0
            && !vis[i - 1][j]
            && matches!(grid[i][j], 2 | 5 | 6)
            && matches!(grid[i - 1][j], 2 | 3 | 4)
        {
            queue.push_back((i - 1, j));
        }

        if j > 0
            && !vis[i][j - 1]
            && matches!(grid[i][j], 1 | 3 | 5)
            && matches!(grid[i][j - 1], 1 | 4 | 6)
        {
            queue.push_back((i, j - 1));
        }

        if i + 1 < n
            && !vis[i + 1][j]
            && matches!(grid[i][j], 2 | 3 | 4)
            && matches!(grid[i + 1][j], 2 | 5 | 6)
        {
            queue.push_back((i + 1, j));
        }

        if j + 1 < m
            && !vis[i][j + 1]
            && matches!(grid[i][j], 1 | 4 | 6)
            && matches!(grid[i][j + 1], 1 | 3 | 5)
        {
            queue.push_back((i, j + 1));
        }
    }

    false
}

/// 1392
pub fn longest_prefix(s: String) -> String {
    let chars = s.chars().collect::<Vec<char>>();
    let (mut left, mut right, mut ans) = (0, chars.len() - 1, 0);
    while right > 0 {
        if chars[0..=left] == chars[right..=chars.len() - 1] {
            ans = left;
        }
        left = left + 1;
        right = right - 1;
    }
    if ans == chars.len() - 1 || (ans == 0 && chars[0] != chars[chars.len() - 1]) {
        return "".to_string();
    }
    chars[0..=ans].iter().collect::<String>()
}

/// 1394
pub fn find_lucky(arr: Vec<i32>) -> i32 {
    arr.iter()
        .fold(HashMap::new(), |mut acc, &cur| {
            *acc.entry(cur).or_insert(0) += 1;
            acc
        })
        .into_iter()
        .fold(-1, |mut acc, (k, v)| {
            if k == v {
                acc = acc.max(k)
            }
            acc
        })
}

/// 1396
struct UndergroundSystem {
    times: HashMap<String, HashMap<String, (i32, i32)>>,
    passengers: HashMap<i32, (String, i32)>,
}

impl UndergroundSystem {
    fn new() -> Self {
        Self {
            times: HashMap::new(),
            passengers: HashMap::new(),
        }
    }

    fn check_in(&mut self, id: i32, start_station: String, start_t: i32) {
        self.passengers.insert(id, (start_station, start_t));
    }

    fn check_out(&mut self, id: i32, end_station: String, end_t: i32) {
        if let Some((start_station, start_t)) = self.passengers.remove(&id) {
            let (sum, count) = self
                .times
                .entry(start_station)
                .or_default()
                .entry(end_station)
                .or_default();
            *sum += end_t - start_t;
            *count += 1;
        }
    }

    fn get_average_time(&mut self, start_station: String, end_station: String) -> f64 {
        let (sum, count) = self
            .times
            .entry(start_station)
            .or_default()
            .entry(end_station)
            .or_default();
        *sum as f64 / *count as f64
    }
}
