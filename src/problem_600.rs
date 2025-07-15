use std::cell::{Ref, RefCell};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::rc::Rc;

use crate::common::TreeNode;

/// p501
pub fn find_mode(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut frequency: HashMap<i32, i32> = HashMap::new();
    fn dfs(root: Rc<RefCell<TreeNode>>, frequncy: &mut HashMap<i32, i32>) {
        let count = frequncy.entry(root.borrow().val).or_insert(0);
        *count = *count + 1;
        if let Some(left) = root.borrow().left.as_ref() {
            dfs(left.clone(), frequncy);
        }
        if let Some(right) = root.borrow().right.as_ref() {
            dfs(right.clone(), frequncy);
        }
    }
    if let Some(val) = root {
        dfs(val, &mut frequency);
    }
    let mut max: i32 = 0;
    let mut keys: Vec<i32> = vec![];
    frequency.iter().for_each(|(key, val)| {
        if *val > max {
            keys.clear();
            keys.push(*key);
            max = *val;
        } else if *val == max {
            keys.push(*key)
        }
    });
    keys
}

/// p502
pub fn find_maximized_capital(k: i32, w: i32, profits: Vec<i32>, capital: Vec<i32>) -> i32 {
    let mut map: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for (key, val) in capital.into_iter().zip(profits.into_iter()) {
        map.entry(key).or_insert(Vec::new()).push(val);
    }
    let mut max_heap: BinaryHeap<i32> = BinaryHeap::new();
    insert(&map, -1, w, &mut max_heap);

    let mut cur_profit = w;
    let mut count = 0;
    while count < k {
        match max_heap.pop() {
            Some(val) if val > 0 => {
                cur_profit += val;
                insert(&map, cur_profit - val, cur_profit, &mut max_heap);
                count += 1;
            }
            _ => break,
        }
    }

    cur_profit
}

fn insert(map: &BTreeMap<i32, Vec<i32>>, left: i32, right: i32, max_heap: &mut BinaryHeap<i32>) {
    if left + 1 > right {
        return;
    }
    for (_, vals) in map.range(left + 1..=right) {
        for &v in vals.iter() {
            max_heap.push(v);
        }
    }
}

/// p503
pub fn next_greater_elements(nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len();
    let mut ans = vec![-1; n];
    let mut st = vec![];
    for i in (0..2 * n).rev() {
        let x = nums[i % n];
        while let Some(&top) = st.last() {
            if x < top {
                break;
            }
            st.pop();
        }
        if i < n && !st.is_empty() {
            ans[i] = *st.last().unwrap();
        }
        st.push(x);
    }
    ans
}

/// p504
pub fn convert_to_base7(num: i32) -> String {
    if num == 0 {
        return "0".to_string();
    }
    let mut symbols: Vec<char> = vec![];
    let mut remain = num.abs();

    while remain != 0 {
        symbols.push((remain % 7).to_string().chars().collect::<Vec<char>>()[0]);
        remain = remain / 7;
    }
    if num < 0 {
        symbols.push('-');
    }
    symbols.reverse();
    symbols.iter().collect::<String>()
}

/// p506
pub fn find_relative_ranks(score: Vec<i32>) -> Vec<String> {
    let mut with_position = score.into_iter().enumerate().collect::<Vec<(usize, i32)>>();
    with_position.sort_by(|a, b| b.1.cmp(&a.1));
    let mut ranking: Vec<String> = vec!["".to_string(); with_position.len()];
    with_position
        .iter()
        .enumerate()
        .for_each(|(index, score)| match index {
            0 => ranking[score.0] = "Gold Medal".to_string(),
            1 => ranking[score.0] = "Silver Medal".to_string(),
            2 => ranking[score.0] = "Bronze Medal".to_string(),
            r => ranking[score.0] = (r + 1).to_string(),
        });

    ranking
}

/// p507
pub fn check_perfect_number(num: i32) -> bool {
    let mut divisor: Vec<i32> = vec![];
    for i in 1..=num / 2 {
        if num % i == 0 {
            divisor.push(i)
        }
    }
    divisor.iter().sum::<i32>() == num
}

/// p508
pub fn find_frequent_tree_sum(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut sum_frequency: HashMap<i32, usize> = HashMap::new();
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>, sum_frequency: &mut HashMap<i32, usize>) -> i32 {
        if let Some(node) = root.as_ref() {
            let mut sum = node.borrow().val;
            sum = sum + dfs(&node.borrow().left, sum_frequency);
            sum = sum + dfs(&node.borrow().right, sum_frequency);
            let count = sum_frequency.entry(sum).or_insert(0);
            *count = *count + 1;
            return sum;
        } else {
            return 0;
        }
    }

    dfs(&root, &mut sum_frequency);

    let mut accuracy: usize = 0;
    let mut vals: Vec<i32> = vec![];
    sum_frequency.iter().for_each(|(v, c)| {
        if *c > accuracy {
            accuracy = *c;
            vals.clear();
            vals.push(*v);
        } else if *c == accuracy {
            vals.push(*v);
        }
    });
    vals
}

/// p509
pub fn fib(n: i32) -> i32 {
    match n {
        0 => return 0,
        1 => return 1,
        v => {
            let mut sum_pre = 0;
            let mut sum = 1;
            let mut temp = 0;
            for i in 0..=v {
                if i >= 2 {
                    temp = sum;
                    sum = sum + sum_pre;
                    sum_pre = temp;
                }
            }
            return sum;
        }
    }
}

/// p513
pub fn find_bottom_left_value(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut ans: i32 = 0;
    if let Some(val) = root.as_ref() {
        let mut queue: Vec<Rc<RefCell<TreeNode>>> = vec![val.clone()];
        while queue.len() != 0 {
            let first = queue.remove(0);
            if let Some(right) = first.borrow().right.as_ref() {
                queue.push(right.clone());
            }
            if let Some(left) = first.borrow().left.as_ref() {
                queue.push(left.clone());
            }
            ans = first.borrow().val;
        }
    }
    ans
}

/// p514
pub fn find_rotate_steps(ring: String, key: String) -> i32 {
    let (n, m) = (ring.len(), key.len());
    let (ring, key) = (
        ring.chars().collect::<Vec<_>>(),
        key.chars().collect::<Vec<_>>(),
    );
    // pos 用于记录 ring 中每个字符出现的所有位置
    let mut pos = vec![vec![]; 26];
    for (i, &c) in ring.iter().enumerate() {
        pos[(c as u8 - b'a') as usize].push(i);
    }
    // dp[i][j] 表示拼写 key 的前 i 个字符，ring 的指针在 j 位置时的最小操作次数
    let mut dp = vec![vec![std::i32::MAX; n]; m];

    // 初始化 dp[0][...]，处理 key 的第一个字符
    for &i in pos[(key[0] as u8 - b'a') as usize].iter() {
        dp[0][i] = i.min(n - i) as i32 + 1; // +1 是按下按钮的操作
    }

    for i in 1..m {
        // 当前字符在 ring 中的位置
        for &j in pos[(key[i] as u8 - b'a') as usize].iter() {
            // 上一个字符在 ring 中的位置
            for &k in pos[(key[i - 1] as u8 - b'a') as usize].iter() {
                dp[i][j] = std::cmp::min(
                    dp[i][j],
                    dp[i - 1][k]
                            + std::cmp::min((j + n - k) % n, (k + n - j) % n) as i32 // 旋转 ring 的最小距离
                            + 1, // +1 是按下按钮的操作
                );
            }
        }
    }
    // 返回拼写完整个 key 时，ring 的指针在任意位置的最小操作次数
    *dp[m - 1].iter().min().unwrap()
}

/// p515
pub fn largest_values(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut ans: Vec<i32> = vec![];
    if let Some(val) = root.as_ref() {
        let mut queue: Vec<Rc<RefCell<TreeNode>>> = vec![val.clone()];
        while queue.len() != 0 {
            let mut max = std::i32::MIN;
            for i in 0..queue.len() {
                let first = queue.remove(0);
                max = max.max(first.borrow().val);
                if let Some(left) = first.borrow().left.as_ref() {
                    queue.push(left.clone());
                }
                if let Some(right) = first.borrow().right.as_ref() {
                    queue.push(right.clone());
                }
            }
            ans.push(max);
        }
    }
    ans
}

/// p516
pub fn longest_palindrome_subseq(s: String) -> i32 {
    let length = s.len();
    let chars = s.chars().collect::<Vec<char>>();
    let mut dp: Vec<Vec<usize>> = vec![vec![0; length]; length];
    for i in (0..length).rev() {
        dp[i][i] = 1;
        let c_1 = chars[i];
        for j in i + 1..length {
            let c_2 = chars[j];
            if c_1 == c_2 {
                dp[i][j] = dp[i + 1][j - 1] + 2
            } else {
                dp[i][j] = dp[i + 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[0][length - 1] as i32
}

/// p517
pub fn find_min_moves(machines: Vec<i32>) -> i32 {
    let n = machines.len() as i32;
    let sum = machines.iter().sum::<i32>();
    if sum % n != 0 {
        return -1;
    }

    let avg = sum / n;
    let mut result = 0;
    let mut sum = 0;
    for &num in machines.iter() {
        let cur_diff = num - avg;
        sum += cur_diff;
        result = i32::max(result, i32::max(sum.abs(), cur_diff));
    }

    result
}

/// p518
pub fn change(amount: i32, coins: Vec<i32>) -> i32 {
    let mut dp: Vec<i32> = vec![0; (amount as usize) + 1];
    dp[0] = 1;
    coins.iter().for_each(|&coin| {
        for i in (coin as usize)..=(amount as usize) {
            dp[i] = dp[i] + dp[i - (coin as usize)];
        }
    });
    return dp[amount as usize];
}

/// p520
pub fn detect_capital_use(word: String) -> bool {
    let cnt = word.bytes().filter(|c| c.is_ascii_uppercase()).count();
    cnt == 0 || cnt == word.len() || cnt == 1 && word.as_bytes()[0].is_ascii_uppercase()
}

/// p521
pub fn find_lu_slength(a: String, b: String) -> i32 {
    if a != b {
        return a.len().max(b.len()) as i32;
    } else {
        -1
    }
}

/// p522
pub fn find_lu_slength_2(strs: Vec<String>) -> i32 {
    let (n, mut ans) = (strs.len(), -1);
    fn is_sub_sequence(s: &[u8], t: &[u8]) -> bool {
        if s == t {
            return true;
        }
        let mut i = 0;
        for &c in t {
            if c == s[i] {
                i += 1;
                if i == s.len() {
                    return true;
                }
            }
        }
        false
    }
    for i in 0..n {
        if strs[i].len() as i32 <= ans {
            continue;
        }
        let mut j = 0;
        while j < n {
            if j != i && is_sub_sequence(&strs[i].as_bytes(), &strs[j].as_bytes()) {
                break;
            }
            j += 1;
        }
        if j == n {
            ans = ans.max(strs[i].len() as i32);
        }
    }
    ans
}

/// p523
pub fn check_subarray_sum(nums: Vec<i32>, k: i32) -> bool {
    let mut m = HashMap::from([(0, 0)]);
    let mut acc = 0;
    for (i, num) in nums.into_iter().enumerate() {
        acc = (acc + num) % k;
        if let Some(&x) = m.get(&acc) {
            if i + 1 - x >= 2 {
                return true;
            }
            continue;
        }
        m.insert(acc, i + 1);
    }
    false
}

// p524
pub fn find_longest_word(s: String, dictionary: Vec<String>) -> String {
    use std::cmp::Ordering;
    let mut s: Vec<char> = s.chars().collect();
    let mut dic = dictionary;
    dic.sort_unstable_by(|a, b| {
        let cmp = b.len().cmp(&a.len());
        if cmp == Ordering::Equal {
            a.cmp(b)
        } else {
            cmp
        }
    });
    for d in dic {
        let mut chars: Vec<char> = d.chars().collect();
        let mut i = 0;
        let mut j = 0;
        while i < chars.len() && j < s.len() {
            if chars[i] == s[j] {
                i += 1;
            }
            j += 1;
        }
        if i == chars.len() {
            return d;
        }
    }
    String::from("")
}

/// p525
pub fn find_max_length(nums: Vec<i32>) -> i32 {
    let mut max_len = 0;
    let mut count = 0;
    let mut map: HashMap<i32, i32> = HashMap::new();
    map.insert(count, -1);

    for (i, &num) in nums.iter().enumerate() {
        if num == 0 {
            count -= 1;
        } else {
            count += 1;
        }

        match map.get(&count) {
            Some(&idx) => {
                max_len = i32::max(max_len, i as i32 - idx);
            }
            None => {
                map.insert(count, i as i32);
            }
        }
    }

    max_len
}

/// p526
pub fn count_arrangement(n: i32) -> i32 {
    fn dfs(s: usize, n: i32, memo: &mut Vec<i32>) -> i32 {
        if s == (1 << n) - 1 {
            return 1;
        }
        if memo[s] != -1 {
            // 之前计算过
            return memo[s];
        }
        let mut res = 0;
        let i = s.count_ones() as i32 + 1;
        for j in 1..=n {
            if (s >> (j - 1) & 1) == 0 && (i % j == 0 || j % i == 0) {
                res += dfs(s | (1 << (j - 1)), n, memo);
            }
        }
        memo[s] = res; // 记忆化
        res
    }
    let mut memo = vec![-1; 1 << n]; // -1 表示没有计算过
    dfs(0, n, &mut memo)
}

/// p529
pub fn update_board(mut board: Vec<Vec<char>>, click: Vec<i32>) -> Vec<Vec<char>> {
    let (a, b) = (click[0] as usize, click[1] as usize);
    if board[a][b] == 'M' {
        board[a][b] = 'X'
    } else if board[a][b] == 'E' {
        let d = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ];
        let (m, n) = (board.len(), board[0].len());
        let mut q = HashSet::new();
        q.insert((a, b));
        while !q.is_empty() {
            let mut p = HashSet::new();
            for (i, j) in q.into_iter() {
                let mut c = 48;
                let mut t = vec![];
                for &(di, dj) in d.iter() {
                    match (i as i32 + di, j as i32 + dj) {
                        (x, y) if x == -1 || x == m as i32 || y == -1 || y == n as i32 => (),
                        (x, y) => {
                            let (x, y) = (x as usize, y as usize);
                            c += (board[x][y] == 'M') as u8;
                            if board[x][y] == 'E' {
                                t.push((x, y));
                            }
                        }
                    }
                }
                board[i][j] = if c > 48 {
                    c as char
                } else {
                    p.extend(t);
                    'B'
                };
            }
            q = p;
        }
    }
    board
}

/// p530
pub fn get_minimum_difference(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut vals: Vec<i32> = vec![];
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>, vals: &mut Vec<i32>) {
        if let Some(v) = root.as_ref() {
            dfs(&v.borrow().left.clone(), vals);
            vals.push(v.borrow().val);
            dfs(&v.borrow().right.clone(), vals)
        }
    }
    dfs(&root, &mut vals);

    let mut min = i32::MAX;
    vals.iter().enumerate().for_each(|(i, v)| {
        if i != 0 {
            min = min.min(v - vals[i - 1])
        }
    });
    min
}

/// p532
pub fn find_pairs(nums: Vec<i32>, k: i32) -> i32 {
    let mut num_count: HashMap<i32, usize> = HashMap::new();
    nums.iter().for_each(|n| {
        let count = num_count.entry(*n).or_insert(0);
        *count = *count + 1
    });
    let mut pair = 0;
    if k == 0 {
        num_count.values().for_each(|c| {
            if *c > 1 {
                pair = pair + 1;
            }
        });
    } else {
        for key in num_count.keys() {
            if num_count.contains_key(&(key + k)) {
                pair = pair + 1;
            }
        }
    }

    pair
}

/// p537
pub fn complex_number_multiply(num1: String, num2: String) -> String {
    let (m, n) = (
        num1[..num1.len() - 1].split_once('+').unwrap(),
        num2[..num2.len() - 1].split_once('+').unwrap(),
    );
    let (a, b, c, d) = (
        m.0.parse::<i32>().unwrap(),
        m.1.parse::<i32>().unwrap(),
        n.0.parse::<i32>().unwrap(),
        n.1.parse::<i32>().unwrap(),
    );
    format!("{}+{}i", a * c - b * d, a * d + b * c)
}

/// p538
pub fn convert_bst(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    let mut acc: i32 = 0;
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>, acc: &mut i32) {
        if let Some(node) = root.as_ref() {
            dfs(&node.borrow().right.clone(), acc);
            let old_val = node.borrow_mut().val;
            node.borrow_mut().val = old_val + *acc;
            *acc = old_val + *acc;
            dfs(&node.borrow().left.clone(), acc);
        }
    }
    dfs(&root, &mut acc);
    root
}

/// p539
pub fn find_min_difference(time_points: Vec<String>) -> i32 {
    if time_points.len() > 1440 {
        return 0;
    }
    let mut min = i32::MAX;

    let mut cache: Vec<i32> = time_points
        .iter()
        .map(|time_point| {
            time_point[0..2].parse::<i32>().unwrap() * 60 + time_point[3..].parse::<i32>().unwrap()
        })
        .collect();

    cache.sort();
    (1..cache.len()).for_each(|i| min = min.min(cache[i] - cache[i - 1]));
    min.min(cache[0] + 1440 - cache[cache.len() - 1])
}

/// p540
pub fn single_non_duplicate(nums: Vec<i32>) -> i32 {
    nums.iter().fold(0, |acc, &n| acc ^ n)
}
