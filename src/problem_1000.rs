use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    i32,
    rc::Rc,
};

use crate::common::TreeNode;

///901
#[derive(Default)]
struct StockSpanner {
    queue: Vec<Vec<i32>>,
}

impl StockSpanner {
    fn new() -> Self {
        Self::default()
    }

    fn next(&mut self, price: i32) -> i32 {
        let mut ret = 1;
        while !self.queue.is_empty() && self.queue[self.queue.len() - 1][0] <= price {
            ret += self.queue.pop().unwrap()[1];
        }
        self.queue.push(vec![price, ret]);
        ret
    }
}

///902
pub fn at_most_n_given_digit_set(digits: Vec<String>, mut n: i32) -> i32 {
    let digits = digits
        .into_iter()
        .map(|i| i.into_bytes()[0] - b'0')
        .collect::<Vec<u8>>();

    let len = digits.len() as i32;
    let mut last_a = 1;
    let mut result = 1;

    let mut n_len = 0;
    let mut b;
    while n != 0 {
        b = (n % 10) as u8;
        n /= 10;
        n_len += 1;

        let mut t = 0;
        for d in &digits {
            t += match d.cmp(&b) {
                Ordering::Greater => break,
                Ordering::Equal => result,
                Ordering::Less => last_a,
            };
        }
        result = t;
        last_a *= len;
    }

    result
        + if len == 1 {
            n_len as i32 - 1
        } else {
            (len.pow(n_len) - len) / (len - 1)
        }
}

///904
pub fn total_fruit(fruits: Vec<i32>) -> i32 {
    let length = fruits.len();
    if length == 0 {
        return 0;
    }
    let (mut left, mut right) = (0, 0);
    let mut total = 0;
    let mut count = HashMap::<i32, i32>::new();
    while right < length {
        let size = count.entry(fruits[right]).or_insert(0);
        *size = *size + 1;
        while count.len() > 2 {
            let size = count.entry(fruits[left]).or_insert(0);
            if *size == 1 {
                count.remove(&fruits[left]);
            } else {
                *size = *size - 1;
            }
            left = left + 1;
        }
        total = total.max(count.values().sum::<i32>());
        right = right + 1;
    }
    total
}

///905
pub fn sort_array_by_parity(nums: Vec<i32>) -> Vec<i32> {
    let (mut even, mut odd) = (vec![], vec![]);
    nums.iter().for_each(|n| {
        if n % 2 == 0 {
            even.push(*n);
        } else {
            odd.push(*n);
        }
    });
    [even, odd].concat()
}

///906
pub fn superpalindromes_in_range(left: String, right: String) -> i32 {
    fn is_palindrome(inp: i64) -> bool {
        let mut rx = 0;
        let mut x = inp;
        while x > 0 {
            rx = 10 * rx + x % 10;
            x /= 10;
        }
        rx == inp
    }
    let left = left.parse::<i64>().unwrap();
    let right = right.parse::<i64>().unwrap();
    const MAGIC: i32 = 1e5 as i32;
    let mut res = 0;
    for k in 1..MAGIC {
        let s = format!("{}", k);
        let s = s.clone() + &s.chars().rev().collect::<String>();
        let v = s.parse::<i64>().unwrap();
        let v = v * v;
        if v > right {
            break;
        }
        if v >= left && is_palindrome(v) {
            res += 1;
        }
    }
    for k in 1..MAGIC {
        let s = format!("{}", k);
        let s = s.clone() + &s.chars().rev().skip(1).collect::<String>();
        let v = s.parse::<i64>().unwrap();
        let v = v * v;
        if v > right {
            break;
        }
        if v >= left && is_palindrome(v) {
            res += 1;
        }
    }
    res
}

///907
pub fn sum_subarray_mins(arr: Vec<i32>) -> i32 {
    let n = arr.len();
    let mut left = vec![-1; n];
    let mut right = vec![n as i32; n];
    let mut stk: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        while !stk.is_empty() && arr[*stk.back().unwrap()] >= arr[i] {
            stk.pop_back();
        }
        if let Some(&top) = stk.back() {
            left[i] = top as i32;
        }
        stk.push_back(i);
    }

    stk.clear();
    for i in (0..n).rev() {
        while !stk.is_empty() && arr[*stk.back().unwrap()] > arr[i] {
            stk.pop_back();
        }
        if let Some(&top) = stk.back() {
            right[i] = top as i32;
        }
        stk.push_back(i);
    }

    const MOD: i64 = 1_000_000_007;
    let mut ans: i64 = 0;
    for i in 0..n {
        ans +=
            ((((right[i] - (i as i32)) * ((i as i32) - left[i])) as i64) * (arr[i] as i64)) % MOD;
        ans %= MOD;
    }
    ans as i32
}

///908
pub fn smallest_range_i(nums: Vec<i32>, k: i32) -> i32 {
    let (max, min) = nums
        .iter()
        .fold((i32::MIN, i32::MAX), |acc, &n| (acc.0.max(n), acc.1.min(n)));
    return if max - min > 2 * k {
        max - min - 2 * k
    } else {
        0
    };
}

///909
pub fn snakes_and_ladders(board: Vec<Vec<i32>>) -> i32 {
    let n = board.len();
    let mut vis = vec![false; n * n + 1];
    vis[1] = true;
    let mut q = vec![1];
    for step in 0.. {
        if q.is_empty() {
            break;
        }
        let tmp = q;
        q = vec![];
        for x in tmp {
            if x == n * n {
                return step;
            }
            for y in x + 1..=(x + 6).min(n * n) {
                let r = (y - 1) / n;
                let mut c = (y - 1) % n;
                if r % 2 > 0 {
                    c = n - 1 - c;
                }
                let mut nxt = board[n - 1 - r][c];
                if nxt < 0 {
                    nxt = y as i32;
                }
                let nxt = nxt as usize;
                if !vis[nxt] {
                    vis[nxt] = true;
                    q.push(nxt);
                }
            }
        }
    }
    -1
}

///910
pub fn smallest_range_ii(mut nums: Vec<i32>, k: i32) -> i32 {
    nums.sort();
    let n = nums.len();
    let mut ans = nums[n - 1] - nums[0];
    for i in (1..n).into_iter() {
        let mx = (nums[i - 1] + k).max(nums[n - 1] - k);
        let mn = (nums[0] + k).min(nums[i] - k);
        ans = ans.min(mx - mn);
    }
    ans
}

///911
struct TopVotedCandidate {
    times: Vec<i32>,
    dp: Vec<i32>,
}

impl TopVotedCandidate {
    fn new(persons: Vec<i32>, times: Vec<i32>) -> Self {
        let n = persons.len();
        let mut map: HashMap<i32, i32> = HashMap::new();
        let mut dp: Vec<i32> = vec![0; n];

        for i in 0..n {
            let count = map.entry(persons[i]).or_insert(0);
            *count += 1;

            dp[i] = if i == 0 {
                persons[i]
            } else {
                let cur_count = *count;
                let prev_person = dp[i - 1];
                let prev_count = map.get(&prev_person).map(|count| *count).unwrap();
                if cur_count >= prev_count {
                    persons[i]
                } else {
                    prev_person
                }
            };
        }

        Self { times, dp }
    }

    fn q(&self, t: i32) -> i32 {
        match self.times.binary_search(&t) {
            Ok(idx) => self.dp[idx],
            Err(idx) => self.dp[idx - 1],
        }
    }
}

///912
pub fn sort_array(nums: Vec<i32>) -> Vec<i32> {
    fn merge(arr_1: &Vec<i32>, arr_2: &Vec<i32>) -> Vec<i32> {
        let (size_1, size_2) = (arr_1.len(), arr_2.len());
        let mut res = vec![];
        let (mut i, mut j) = (0, 0);
        while i < size_1 || j < size_2 {
            if i == size_1 {
                res.push(arr_2[j]);
                j = j + 1;
            } else if j == size_2 {
                res.push(arr_1[i]);
                i = i + 1;
            } else if arr_1[i] <= arr_2[j] {
                res.push(arr_1[i]);
                i = i + 1;
            } else {
                res.push(arr_2[j]);
                j = j + 1;
            }
        }
        res
    }
    let mut ans = nums.iter().map(|&n| vec![n]).collect::<Vec<Vec<i32>>>();
    let mut size = ans.len();
    while size > 1 {
        println!("{:?}", ans);
        let mut merged_arr = vec![];
        for i in (0..size).step_by(2) {
            if i < size - 1 {
                merged_arr.push(merge(&ans[i], &ans[i + 1]));
            }
        }
        if size % 2 != 0 {
            merged_arr.push(ans.last().unwrap().clone())
        }
        ans = merged_arr;
        size = ans.len();
    }
    ans[0].to_vec()
}

///913
pub fn cat_mouse_game(graph: Vec<Vec<i32>>) -> i32 {
    let n = graph.len();
    if n <= 2 {
        return if n == 0 { 0 } else { 1 };
    }

    if graph[1].is_empty() {
        return 2;
    }
    if graph[2].is_empty() {
        return 1;
    }

    let mut dp = vec![0u8; n * n * 2];
    let mut queue = VecDeque::with_capacity(n * n * 2);

    for mouse in 0..n {
        for cat in 0..n {
            let idx = mouse * n * 2 + cat * 2;
            if mouse == 0 {
                dp[idx] = 1;
                dp[idx + 1] = 1;
                queue.push_back((mouse, cat, 0, 1));
                queue.push_back((mouse, cat, 1, 1));
            } else if mouse == cat {
                dp[idx] = 2;
                dp[idx + 1] = 2;
                queue.push_back((mouse, cat, 0, 2));
                queue.push_back((mouse, cat, 1, 2));
            }
        }
    }

    while let Some((mouse, cat, turn, result)) = queue.pop_front() {
        if mouse == 1 && cat == 2 && turn == 0 {
            return result as i32;
        }

        let prev_turn = turn ^ 1;
        let prev_turn_idx = prev_turn as usize;

        if prev_turn == 0 {
            for &prev_mouse in &graph[mouse] {
                let prev_mouse = prev_mouse as usize;
                let idx = prev_mouse * n * 2 + cat * 2 + prev_turn_idx;

                if dp[idx] == 0 {
                    if result == 1 {
                        dp[idx] = 1;
                        queue.push_back((prev_mouse, cat, prev_turn, 1));
                    } else {
                        let mut all_cat_win = true;
                        for &next in &graph[prev_mouse] {
                            let next_idx =
                                next as usize * n * 2 + cat * 2 + (prev_turn ^ 1) as usize;
                            if dp[next_idx] != 2 {
                                all_cat_win = false;
                                break;
                            }
                        }
                        if all_cat_win {
                            dp[idx] = 2;
                            queue.push_back((prev_mouse, cat, prev_turn, 2));
                        }
                    }
                }
            }
        } else {
            for &prev_cat in &graph[cat] {
                let prev_cat = prev_cat as usize;
                if prev_cat == 0 {
                    continue;
                }

                let idx = mouse * n * 2 + prev_cat * 2 + prev_turn_idx;

                if dp[idx] == 0 {
                    if result == 2 {
                        dp[idx] = 2;
                        queue.push_back((mouse, prev_cat, prev_turn, 2));
                    } else {
                        let mut all_mouse_win = true;
                        for &next in &graph[prev_cat] {
                            if next == 0 {
                                continue;
                            }
                            let next_idx =
                                mouse * n * 2 + next as usize * 2 + (prev_turn ^ 1) as usize;
                            if dp[next_idx] != 1 {
                                all_mouse_win = false;
                                break;
                            }
                        }
                        if all_mouse_win {
                            dp[idx] = 1;
                            queue.push_back((mouse, prev_cat, prev_turn, 1));
                        }
                    }
                }
            }
        }
    }

    dp[1 * n * 2 + 2 * 2] as i32
}

///914
pub fn has_groups_size_x(deck: Vec<i32>) -> bool {
    fn gcd(a: usize, b: usize) -> usize {
        return if a == 0 { b } else { gcd(b % a, a) };
    }
    let mut count = HashMap::<i32, usize>::new();
    deck.iter().for_each(|&n| {
        let count = count.entry(n).or_insert(0);
        *count = *count + 1;
    });
    let mut all_gcd = 0;
    count.values().for_each(|&v| {
        if all_gcd == 0 {
            all_gcd = v;
        } else {
            all_gcd = gcd(all_gcd, v);
        }
    });
    return all_gcd >= 2;
}

///915
pub fn partition_disjoint(nums: Vec<i32>) -> i32 {
    let size = nums.len();
    let mut suffix_min = vec![i32::MAX; size];
    nums.iter().enumerate().rev().for_each(|(i, &n)| {
        if i == size - 1 {
            suffix_min[i] = n;
        } else {
            suffix_min[i] = suffix_min[i + 1].min(n);
        }
    });
    let mut prefix_max = vec![i32::MIN; size];
    for i in 0..size {
        if i == 0 {
            prefix_max[i] = nums[i];
        } else {
            prefix_max[i] = prefix_max[i - 1].max(nums[i]);
        }
        if i < size - 1 && prefix_max[i] <= suffix_min[i + 1] {
            return (i + 1) as i32;
        }
    }
    return -1;
}

/// 916
pub fn word_subsets(words1: Vec<String>, words2: Vec<String>) -> Vec<String> {
    fn get_count(word: &str) -> [i32; 26] {
        let mut records = [0; 26];
        for idx in word.bytes().map(|b| (b - b'a') as usize) {
            records[idx] += 1;
        }
        records
    }

    let mut totals = [0; 26];
    for word in words2.iter() {
        let records = get_count(word);
        for i in 0..26 {
            totals[i] = i32::max(totals[i], records[i]);
        }
    }

    for word in words1.iter() {
        let records = get_count(word);
        if (0..26).all(|idx| records[idx] >= totals[idx]) {}
    }

    words1
        .into_iter()
        .filter(|word| -> bool {
            let records = get_count(word);
            (0..26).all(|idx| records[idx] >= totals[idx])
        })
        .collect()
}

/// 917
pub fn reverse_only_letters(s: String) -> String {
    let mut chars = s.chars().collect::<Vec<char>>();
    let (mut left, mut right) = (0, chars.len() - 1);
    while left < right {
        if !chars[left].is_alphabetic() {
            left = left + 1;
        } else if !chars[right].is_alphabetic() {
            right = right - 1;
        } else {
            let temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left = left + 1;
            right = right - 1;
        }
    }
    chars.iter().collect::<String>()
}

/// 918
pub fn max_subarray_sum_circular(nums: Vec<i32>) -> i32 {
    let (mut sum, mut curr_max, mut curr_min, mut max_sum, mut min_sum) = (0, 0, 0, nums[0], 0);
    for num in nums {
        curr_max = num.max(curr_max + num);
        max_sum = max_sum.max(curr_max);
        curr_min = num.min(curr_min + num);
        min_sum = min_sum.min(curr_min);
        sum += num;
    }
    if max_sum > 0 {
        max_sum.max(sum - min_sum)
    } else {
        max_sum
    }
}

/// 919
struct CBTInserter {
    nodes: Vec<Option<Rc<RefCell<TreeNode>>>>,
}

impl CBTInserter {
    fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut nodes = Vec::new();
        nodes.push(root);
        let mut i = 0;
        while i < nodes.len() {
            let (left, right) = (
                nodes[i].as_ref().unwrap().borrow().left.clone(),
                nodes[i].as_ref().unwrap().borrow().right.clone(),
            );
            if left.is_some() {
                nodes.push(left);
            }
            if right.is_some() {
                nodes.push(right);
            }
            i += 1;
        }
        CBTInserter { nodes }
    }

    fn insert(&mut self, val: i32) -> i32 {
        let pos = self.nodes.len();
        let node = Rc::new(RefCell::new(TreeNode::new(val)));
        self.nodes.push(Some(node.clone()));
        if (pos & 1) == 1 {
            self.nodes[(pos - 1) / 2]
                .as_ref()
                .unwrap()
                .borrow_mut()
                .left = Some(node.clone())
        } else {
            self.nodes[(pos - 1) / 2]
                .as_ref()
                .unwrap()
                .borrow_mut()
                .right = Some(node.clone())
        }
        self.nodes[(pos - 1) / 2].as_ref().unwrap().borrow().val
    }

    fn get_root(&self) -> Option<Rc<RefCell<TreeNode>>> {
        self.nodes[0].clone()
    }
}

/// 920
pub fn num_music_playlists(n: i32, goal: i32, k: i32) -> i32 {
    let mut dp = vec![vec![0_i64; n as usize + 1]; goal as usize + 1];
    dp[0][0] = 1;

    for i in 1..=goal as usize {
        for j in 1..=n as usize {
            dp[i][j] += dp[i - 1][j - 1] * (n as i64 - j as i64 + 1);
            dp[i][j] += dp[i - 1][j] * 0.max(j as i64 - k as i64);
            dp[i][j] %= 1_000_000_007;
        }
    }

    dp[goal as usize][n as usize] as i32
}

/// 921
pub fn min_add_to_make_valid(s: String) -> i32 {
    let (chars, mut stack, mut extra_left) = (s.chars().collect::<Vec<char>>(), vec![], 0);
    for c in chars {
        match c {
            '(' => stack.push('('),
            ')' => {
                if stack.len() > 0 {
                    stack.pop();
                } else {
                    extra_left = extra_left + 1;
                }
            }
            _ => {}
        }
    }
    return extra_left + (stack.len() as i32);
}

/// 922
pub fn sort_array_by_parity_ii(mut nums: Vec<i32>) -> Vec<i32> {
    let mut odd = 1;
    for even in (0..nums.len()).step_by(2) {
        if nums[even] % 2 == 1 {
            while nums[odd] % 2 == 1 {
                odd = odd + 2;
            }
            let temp = nums[even];
            nums[even] = nums[odd];
            nums[odd] = temp;
        }
    }
    nums
}

/// 923
pub fn three_sum_multi(arr: Vec<i32>, target: i32) -> i32 {
    let target = target as usize;
    const MOD: i64 = 1e9 as i64 + 7;
    let mut mp = vec![0; 101];
    for num in arr {
        mp[num as usize] += 1;
    }
    let mut res = 0i64;
    for i in 0..101usize {
        if i + i > target {
            break;
        }
        let t = target - i - i;
        if t == i {
            res += mp[i] * (mp[i] - 1) * (mp[i] - 2) / 6;
        } else if t > i && t <= 100 {
            res += mp[i] * (mp[i] - 1) * mp[t] / 2;
        }
        res = res % MOD;
        for j in i + 1..101 {
            if i + j > target {
                break;
            }
            let t = target - i - j;
            if t == j {
                res += mp[i] * mp[j] * (mp[j] - 1) / 2;
            } else if t > j && t <= 100 {
                res += mp[i] * mp[j] * mp[t];
            }
            res = res % MOD;
        }
    }
    res as i32
}

/// 924
pub fn min_malware_spread(graph: Vec<Vec<i32>>, initial: Vec<i32>) -> i32 {
    let n = graph.len();
    let mut ids: Vec<i32> = vec![0; n];
    let mut id_to_size: HashMap<i32, i32> = HashMap::new();
    let mut id = 0;
    for i in 0..n {
        if ids[i] == 0 {
            id += 1;
            let mut q = VecDeque::from([i]);
            ids[i] = id;
            let mut size = 1;
            while let Some(u) = q.pop_front() {
                for v in 0..n {
                    if graph[u][v] == 1 && ids[v] == 0 {
                        size += 1;
                        q.push_back(v);
                        ids[v] = id;
                    }
                }
            }
            id_to_size.insert(id, size);
        }
    }
    let mut id_to_initials: HashMap<i32, i32> = HashMap::new();
    for &u in initial.iter() {
        *id_to_initials.entry(ids[u as usize]).or_insert(0) += 1;
    }
    let mut ans = n as i32 + 1;
    let mut ans_removed = 0;
    for &u in initial.iter() {
        let removed = if id_to_initials[&ids[u as usize]] == 1 {
            id_to_size[&ids[u as usize]]
        } else {
            0
        };
        if removed > ans_removed || (removed == ans_removed && u < ans) {
            ans = u;
            ans_removed = removed;
        }
    }
    ans
}

/// 925
pub fn is_long_pressed_name(name: String, typed: String) -> bool {
    let (true_name, typed_name) = (
        name.chars().collect::<Vec<char>>(),
        typed.chars().collect::<Vec<char>>(),
    );
    let (mut i, mut j) = (0, 0);
    while j < typed_name.len() {
        if i < true_name.len() && true_name[i] == typed_name[j] {
            i = i + 1;
            j = j + 1;
        } else if j > 0 && typed_name[j] == typed_name[j - 1] {
            j = j + 1;
        } else {
            return false;
        }
    }
    i == true_name.len()
}

/// 926
pub fn min_flips_mono_incr(s: String) -> i32 {
    let chars = s.chars().collect::<Vec<char>>();
    let mut dp = vec![[i32::MAX; 2]; chars.len()];
    for (i, c) in chars.iter().enumerate() {
        match c {
            '1' => {
                if i == 0 {
                    dp[i][0] = 1;
                    dp[i][1] = 0;
                } else {
                    dp[i][0] = dp[i - 1][0] + 1;
                    dp[i][1] = dp[i - 1][0].min(dp[i - 1][1]);
                }
            }
            '0' => {
                if i == 0 {
                    dp[i][0] = 0;
                    dp[i][1] = 1;
                } else {
                    dp[i][0] = dp[i - 1][0];
                    dp[i][1] = dp[i - 1][0].min(dp[i - 1][1]) + 1;
                }
            }
            _ => (),
        }
    }
    let last = dp.last().unwrap();
    last[0].min(last[1])
}

/// 927
pub fn three_equal_parts(arr: Vec<i32>) -> Vec<i32> {
    let (one_total, mut one_cnt) = (arr.iter().filter(|&&x| x == 1).count() as i32, 0);
    if one_total == 0 {
        return vec![0, 2];
    }
    if one_total % 3 != 0 {
        return vec![-1, -1];
    }

    let (mut x, mut y, mut z) = arr
        .iter()
        .enumerate()
        .fold((0, 0, 0), |(x, y, z), (i, &v)| {
            if v == 1 {
                if one_cnt == 0 {
                    one_cnt += 1;
                    (i, y, z)
                } else if one_cnt == one_total / 3 {
                    one_cnt += 1;
                    (x, i, z)
                } else if one_cnt == one_total / 3 * 2 {
                    one_cnt += 1;
                    (x, y, i)
                } else {
                    one_cnt += 1;
                    (x, y, z)
                }
            } else {
                (x, y, z)
            }
        });
    while z < arr.len() {
        if arr[x] != arr[y] || arr[y] != arr[z] {
            return vec![-1, -1];
        }
        x += 1;
        y += 1;
        z += 1;
    }
    vec![x as i32 - 1, y as i32]
}

/// 928
pub fn min_malware_spread_1(graph: Vec<Vec<i32>>, initial: Vec<i32>) -> i32 {
    fn dfs(graph: &Vec<Vec<i32>>, visited: &mut Vec<bool>, start: usize, remove: usize) -> i32 {
        if start == remove {
            return 0;
        }
        let mut count = 1;
        visited[start] = true;
        for i in 0..graph.len() {
            if graph[start][i] == 1 && !visited[i] && i != remove {
                count += dfs(graph, visited, i, remove);
            }
        }
        count
    }
    let n = graph.len();
    let mut initial = initial;
    initial.sort();
    let mut min_malware_size = std::i32::MAX;
    let mut node_to_remove = initial[0];

    for &infected in &initial {
        let mut visited = vec![false; n];
        let mut size_of_malware_spread = 0;

        for &start in &initial {
            if start != infected && !visited[start as usize] {
                let count = dfs(&graph, &mut visited, start as usize, infected as usize);
                size_of_malware_spread += count;
            }
        }

        if size_of_malware_spread < min_malware_size {
            min_malware_size = size_of_malware_spread;
            node_to_remove = infected;
        }
    }

    node_to_remove
}

/// 929
pub fn num_unique_emails(emails: Vec<String>) -> i32 {
    let mut uniq = HashSet::new();
    for email in emails {
        let (mut is_plus, mut is_end, mut final_email) = (false, false, String::new());
        email.chars().for_each(|ch| {
            if ch == '+' {
                is_plus = true
            }
            if ch == '@' {
                is_end = true
            }
            if is_end || (!is_plus && ch != '.') {
                final_email.push(ch)
            }
        });
        uniq.insert(final_email);
    }
    uniq.len() as i32
}

/// 930
pub fn num_subarrays_with_sum(nums: Vec<i32>, goal: i32) -> i32 {
    let size = nums.len();
    let (mut surffix_sum, mut count) = (vec![0; size], 0);
    nums.iter().enumerate().for_each(|(i, &n)| {
        if i == 0 {
            surffix_sum[i] = n;
        } else {
            surffix_sum[i] = surffix_sum[i - 1] + n;
        }
        if surffix_sum[i] == goal {
            count = count + 1;
        }
        for j in 0..i {
            if surffix_sum[i] - surffix_sum[j] == goal {
                count = count + 1;
            }
        }
    });
    count
}
