use std::{
    cell::RefCell,
    collections::{HashSet, VecDeque},
    hash::{DefaultHasher, Hasher},
    rc::Rc,
};

use rand::rand_core::le;

use crate::common::TreeNode;

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
