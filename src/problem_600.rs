use std::cell::RefCell;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::i32;
use std::rc::Rc;
use std::str::FromStr;

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
    let s: Vec<char> = s.chars().collect();
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
        let chars: Vec<char> = d.chars().collect();
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

/// p541
pub fn reverse_str(s: String, k: i32) -> String {
    let mut s = s.chars().collect::<Vec<char>>();
    for i in (0..s.len()).step_by(2 * k as usize) {
        let len = if s.len() - i > k as usize {
            k as usize
        } else {
            s.len() - i
        };
        for j in 0..len / 2 {
            s.swap(i + j, i + len - j - 1);
        }
    }
    s.iter().collect()
}

/// p542
pub fn update_matrix(mat: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut distance = mat.clone();
    let height = mat.len();
    let width = mat[0].len();
    let mut visited: Vec<Vec<bool>> = vec![vec![false; width]; height];
    let mut temp: Vec<[usize; 2]> = vec![];
    for i in 0..height {
        for j in 0..width {
            if mat[i][j] == 0 {
                visited[i][j] = true;
                temp.push([i, j])
            }
        }
    }

    while temp.len() != 0 {
        let [i, j] = temp.remove(0);
        for [r, c] in [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]] {
            if r < height && c < width && visited[r][c] == false {
                visited[r][c] = true;
                distance[r][c] = distance[i][j] + 1;
                temp.push([r, c])
            }
        }
    }
    distance
}

/// p543
pub fn diameter_of_binary_tree(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>) -> Option<(i32, i32)> {
        if let Some(node) = root.as_ref() {
            let left_depth = dfs(&node.borrow().left).unwrap_or((-1, -1));
            let right_depth = dfs(&node.borrow().right).unwrap_or((-1, -1));
            return Some((
                left_depth.0.max(right_depth.0) + 1,
                (left_depth.0 + 1 + right_depth.0 + 1)
                    .max(left_depth.1)
                    .max(right_depth.1),
            ));
        }
        None
    }
    return dfs(&root).unwrap_or((0, 0)).1;
}

/// p546
pub fn remove_boxes(boxes: Vec<i32>) -> i32 {
    fn calculate_points(
        boxes: &[i32],
        dp: &mut [[[u16; 100]; 100]; 100],
        l: usize,
        mut r: usize,
        mut k: usize,
    ) -> u16 {
        if l as isize > r as isize {
            0
        } else if dp[l][r][k] != 0 {
            dp[l][r][k]
        } else {
            while r > l && boxes[r] == boxes[r - 1] {
                r -= 1;
                k += 1;
            }
            dp[l][r][k] =
                calculate_points(boxes, dp, l, r - 1, 0) + (k as u16 + 1) * (k as u16 + 1);
            for i in l..r {
                if boxes[i] == boxes[r] {
                    dp[l][r][k] = dp[l][r][k].max(
                        calculate_points(boxes, dp, l, i, k + 1)
                            + calculate_points(boxes, dp, i + 1, r - 1, 0),
                    )
                }
            }
            dp[l][r][k]
        }
    }
    let mut dp = [[[0; 100]; 100]; 100];

    calculate_points(&boxes, &mut dp, 0, boxes.len() - 1, 0) as i32
}

/// p547
pub fn find_circle_num(is_connected: Vec<Vec<i32>>) -> i32 {
    fn dfs(is_connected: &Vec<Vec<i32>>, visited: &mut Vec<bool>, o: usize) {
        for i in 0..is_connected.len() {
            if is_connected[o][i] == 1 && !visited[i] {
                visited[i] = true;
                dfs(is_connected, visited, i);
            }
        }
    }
    let mut visited = vec![false; is_connected.len()];
    let mut ans = 0;
    for r in 0..is_connected.len() {
        for c in r..is_connected[r].len() {
            if is_connected[r][c] == 1 && !visited[c] {
                visited[c] = true;
                ans += 1;
                dfs(&is_connected, &mut visited, c);
            }
        }
    }
    ans
}

/// p551
pub fn check_record(s: String) -> bool {
    let mut late = false;
    let mut absence: usize = 0;
    let chars = s.chars().collect::<Vec<char>>();
    let length = chars.len();
    for (i, &c) in chars.iter().enumerate() {
        match c {
            'A' => absence = absence + 1,
            'L' => {
                if length > 2 && i < length - 2 && chars[i + 1] == 'L' && chars[i + 2] == 'L' {
                    late = true;
                }
            }
            _ => (),
        }
    }
    !late && absence < 2
}

/// p552
pub fn check_record_2(n: i32) -> i32 {
    const MOD: i64 = 1_000_000_007;
    let n = n as usize;
    if n == 1 {
        return 3;
    }
    let mut dp = vec![0; n + 3];
    dp[1] = 1;
    dp[2] = 1;
    let mut sum = 0;
    for i in 0..n {
        dp[i + 3] = (dp[i + 2] + dp[i + 1] + dp[i]) % MOD;
    }
    for i in 0..=n {
        if i == 0 || i == n - 1 {
            sum = (sum + dp[n + 1]) % MOD;
        } else if i == n {
            sum = (sum + dp[n + 2]) % MOD;
        } else {
            sum = (sum + (dp[i + 2]) * (dp[n - i + 1])) % MOD;
        }
    }
    sum as i32
}

/// p553
pub fn optimal_division(nums: Vec<i32>) -> String {
    (1..nums.len()).fold(nums[0].to_string(), |ret, i| match i {
        1 if nums.len() > 2 => format!("{}/({}", ret, nums[i]),
        _ => {
            if i == nums.len() - 1 && nums.len() > 2 {
                format!("{}/{})", ret, nums[i])
            } else {
                format!("{}/{}", ret, nums[i])
            }
        }
    })
}

/// p554
pub fn least_bricks(wall: Vec<Vec<i32>>) -> i32 {
    let mut end_point: HashMap<i32, i32> = HashMap::new();
    let mut max: i32 = 0;
    for row in wall.iter() {
        let mut acc: i32 = 0;
        let length = row.len();
        for (i, &brick) in row.iter().enumerate() {
            if i != length - 1 {
                acc = acc + brick;
                let count = end_point.entry(acc).or_insert(0);
                *count = *count + 1;
                max = max.max(*count)
            }
        }
    }
    wall.len() as i32 - max
}

/// p556
pub fn next_greater_element(n: i32) -> i32 {
    let mut ch_arr = n.to_string().chars().collect::<Vec<char>>();
    let (mut i, mut j): (i32, usize) = (ch_arr.len() as i32 - 2, ch_arr.len() - 1);
    while i >= 0 as i32 && ch_arr[i as usize] >= ch_arr[i as usize + 1] {
        i -= 1;
    }
    if i == -1 {
        return -1;
    }
    while j < ch_arr.len() && ch_arr[i as usize] >= ch_arr[j] {
        j -= 1;
    }
    ch_arr.swap(i as usize, j);
    let (mut prefix, mut tmp) = (
        ch_arr[..i as usize + 1].to_vec(),
        ch_arr[i as usize + 1..].to_vec(),
    );
    tmp.sort();
    prefix.append(&mut tmp);
    return if let Ok(num) = prefix.iter().map(|c| *c).collect::<String>().parse::<i32>() {
        num
    } else {
        -1
    };
}

// p557
pub fn reverse_words(s: String) -> String {
    let contents = s.split(" ").collect::<Vec<&str>>();
    contents
        .iter()
        .map(|&s| {
            return s
                .chars()
                .map(|c| c.to_string())
                .rev()
                .collect::<Vec<String>>()
                .concat();
        })
        .collect::<Vec<String>>()
        .join(" ")
}

/// p560
pub fn subarray_sum(nums: Vec<i32>, k: i32) -> i32 {
    let mut ans = 0;
    let mut s = 0;
    let mut cnt = HashMap::with_capacity(nums.len());
    for x in nums {
        *cnt.entry(s).or_insert(0) += 1;
        s += x;
        if let Some(&c) = cnt.get(&(s - k)) {
            ans += c;
        }
    }
    return ans;
}

/// p561
pub fn array_pair_sum(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort();
    let length = nums.len();
    nums.chunks(2).fold(0, |acc, cur| acc + cur[0])
}

/// p563
pub fn find_tilt(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut acc: i32 = 0;
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>, acc: &mut i32) -> (i32, i32) {
        if let Some(node) = root.as_ref() {
            let (left_sum, left_diff) = dfs(&node.borrow().left, acc);
            let (right_sum, right_diff) = dfs(&node.borrow().right, acc);
            *acc = left_diff + right_diff + *acc;
            return (
                node.borrow().val + left_sum + right_sum,
                (left_sum - right_sum).abs(),
            );
        } else {
            (0, 0)
        }
    }
    let (_, diff) = dfs(&root, &mut acc);
    acc + diff
}

/// p564
pub fn nearest_palindromic(n: String) -> String {
    fn get_palindrome(mut left: i64, mark: bool) -> i64 {
        let mut ret = left;
        if !mark {
            left /= 10;
        }
        while left > 0 {
            ret = ret * 10 + left % 10;
            left /= 10;
        }
        ret
    }
    let n_len = n.len();
    let i = if (n_len & 1) == 1 {
        n_len >> 1
    } else {
        (n_len >> 1) - 1
    };
    let half = i64::from_str(&n[..i + 1]).unwrap();
    let mark = (n_len & 1) == 0;

    let mut cache: Vec<i64> = vec![0; 5];
    cache.push(((10 as i64).pow(n_len as u32 - 1) - 1) as i64);
    cache.push(((10 as i64).pow(n_len as u32) + 1) as i64);
    cache.push(get_palindrome(half, mark));
    cache.push(get_palindrome(half - 1, mark));
    cache.push(get_palindrome(half + 1, mark));

    let nl = i64::from_str(&n).unwrap();
    let mut ret = 0;
    let mut diff = i64::MAX;
    while let Some(v) = cache.pop() {
        if v == nl {
            continue;
        }
        let curr_diff = ((v - nl) as i64).abs();
        if curr_diff == diff {
            ret = v.min(ret);
        } else if curr_diff < diff {
            ret = v;
            diff = curr_diff;
        }
    }
    ret.to_string()
}

/// p565
pub fn array_nesting(mut nums: Vec<i32>) -> i32 {
    nums.clone()
        .iter()
        .enumerate()
        .scan(0, |max_len, (i, num)| {
            let (mut index, mut cnt): (usize, i32) = (i, 0);
            if *num != -1 {
                while nums[index] != -1 {
                    let tmp = nums[index];
                    nums[index] = -1;
                    index = tmp as usize;
                    cnt += 1;
                }
            }
            Some(cnt.max(*max_len))
        })
        .max()
        .unwrap_or(0)
}

/// p566
pub fn matrix_reshape(mat: Vec<Vec<i32>>, r: i32, c: i32) -> Vec<Vec<i32>> {
    let (r, c) = (r as usize, c as usize);
    let row = mat.len();
    if row == 0 {
        return mat;
    }
    let col = mat[0].len();
    if r * c != row * col {
        return mat;
    }
    let mut new_mat: Vec<Vec<i32>> = vec![vec![0; c]; r];
    for i in 0..row {
        for j in 0..col {
            let length = col * i + j;
            let row = length / c;
            let col = length % c;
            new_mat[row][col] = mat[i][j];
        }
    }
    new_mat
}

/// p567
pub fn check_inclusion(s1: String, s2: String) -> bool {
    let mut s1_content: HashMap<char, usize> = HashMap::new();
    for c in s1.chars() {
        *s1_content.entry(c).or_insert(0) += 1;
    }
    let mut find_target = false;
    let mut temp_content: HashMap<char, usize> = HashMap::new();
    s2.chars()
        .collect::<Vec<char>>()
        .windows(s1.len())
        .enumerate()
        .for_each(|(i, content)| {
            if i == 0 {
                for &c in content {
                    *temp_content.entry(c).or_insert(0) += 1;
                }
            } else if let Some(&c) = content.last() {
                *temp_content.entry(c).or_insert(0) += 1;
            }
            if temp_content == s1_content {
                find_target = true;
            }
            if let Some(&c) = content.first() {
                let count = temp_content.entry(c).or_insert(0);
                if *count == 1 {
                    temp_content.remove(&c);
                } else {
                    *count = *count - 1;
                }
            }
        });
    find_target
}

/// p572
pub fn is_subtree(
    root: Option<Rc<RefCell<TreeNode>>>,
    sub_root: Option<Rc<RefCell<TreeNode>>>,
) -> bool {
    fn dfs(a: &Option<Rc<RefCell<TreeNode>>>, b: &Option<Rc<RefCell<TreeNode>>>) -> bool {
        if a == b {
            return true;
        }
        if a.is_none() || b.is_none() {
            return false;
        }
        let (a, b) = (a.as_ref().unwrap().borrow(), b.as_ref().unwrap().borrow());
        a.val == b.val && dfs(&a.left, &b.left) && dfs(&a.right, &b.right)
    }
    if sub_root.is_none() {
        return true;
    }
    if let Some(r) = root {
        let node = r.borrow();
        dfs(&Some(r.clone()), &sub_root)
            || is_subtree(node.left.clone(), sub_root.clone())
            || is_subtree(node.right.clone(), sub_root.clone())
    } else {
        false
    }
}

/// p575
pub fn distribute_candies(candy_type: Vec<i32>) -> i32 {
    let set = candy_type.iter().collect::<HashSet<_>>();
    (candy_type.len() / 2).min(set.len()) as i32
}

/// p576
pub fn find_paths(m: i32, n: i32, max_move: i32, start_row: i32, start_column: i32) -> i32 {
    let mut dp = vec![vec![vec![0; n as usize]; m as usize]; max_move as usize + 1];
    dp[0][start_row as usize][start_column as usize] = 1;
    let mut out_counts = 0;
    let mod_num = 1_000_000_007;

    for i in 0..max_move {
        for j in 0..m {
            for k in 0..n {
                if dp[i as usize][j as usize][k as usize] > 0 {
                    for &(dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)].iter() {
                        let new_x = j + dx;
                        let new_y = k + dy;
                        if new_x >= 0 && new_x < m && new_y >= 0 && new_y < n {
                            dp[i as usize + 1][new_x as usize][new_y as usize] = (dp
                                [i as usize + 1][new_x as usize][new_y as usize]
                                + dp[i as usize][j as usize][k as usize])
                                % mod_num;
                        } else {
                            out_counts =
                                (out_counts + dp[i as usize][j as usize][k as usize]) % mod_num;
                        }
                    }
                }
            }
        }
    }
    out_counts
}

/// p581
pub fn find_unsorted_subarray(nums: Vec<i32>) -> i32 {
    let mut pre = nums[0];
    let ir = nums[1..].iter().enumerate().fold(0, |ir, (i, &x)| {
        if x >= pre {
            pre = x;
            ir
        } else {
            i + 1
        }
    });
    if 0 == ir {
        return 0;
    }
    pre = nums[ir];
    nums[..ir].iter().rev().enumerate().fold(0, |ans, (i, &x)| {
        if x <= pre {
            pre = x;
            ans
        } else {
            i + 2
        }
    }) as i32
}

/// p583
pub fn min_distance(word1: String, word2: String) -> i32 {
    let char1 = word1.chars().collect::<Vec<char>>();
    let char2 = word2.chars().collect::<Vec<char>>();
    let len1 = char1.len();
    let len2 = char2.len();
    let mut dp = vec![vec![-1; len2 + 1]; len1 + 1];
    dp[0][0] = 0;
    for i in 1..=len1 {
        dp[i][0] = i as i32;
    }
    for j in 1..=len2 {
        dp[0][j] = j as i32;
    }
    for i in 1..=len1 {
        let c1 = char1[i - 1];
        for j in 1..=len2 {
            let c2 = char2[j - 1];
            if c1 == c2 {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = dp[i][j - 1].min(dp[i - 1][j]) + 1;
            }
        }
    }

    dp[len1][len2]
}
