use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    i32, iter,
    rc::Rc,
};

use crate::common::{ListNode, TreeNode};

/// p401
fn read_binary_watch(turned_on: i32) -> Vec<String> {
    let mut res = Vec::new();
    for h in 0..12 as i32 {
        for m in 0..60 as i32 {
            if h.count_ones() + m.count_ones() == (turned_on as u32) {
                res.push(format!("{}:{:02}", h, m))
            }
        }
    }
    res
}

/// p402
pub fn remove_kdigits(num: String, mut k: i32) -> String {
    let mut stack = vec![];

    for b in num.bytes() {
        while k > 0 {
            match stack.last() {
                Some(&prev_b) if prev_b > b => {
                    stack.pop();
                    k -= 1;
                }
                _ => break,
            }
        }

        if !stack.is_empty() || b != b'0' {
            stack.push(b);
        }
    }

    let n = stack.len();
    let k = k as usize;
    if n > k {
        String::from_utf8_lossy(&stack[..n - k]).to_string()
    } else {
        "0".to_string()
    }
}

/// p403
pub fn can_cross(stones: Vec<i32>) -> bool {
    let mut maps: HashMap<(usize, i32), bool> = HashMap::new();
    fn dfs(
        stones: &Vec<i32>,
        maps: &mut HashMap<(usize, i32), bool>,
        cur: usize,
        last_distance: i32,
    ) -> bool {
        if cur + 1 == stones.len() {
            return true;
        }
        if let Some(&flag) = maps.get(&(cur, last_distance)) {
            return flag;
        }

        for next_distance in last_distance - 1..=last_distance + 1 {
            if next_distance <= 0 {
                continue;
            }
            let next = stones[cur] + next_distance;
            if let Ok(next_idx) = stones.binary_search(&next) {
                if dfs(stones, maps, next_idx, next_distance) {
                    maps.insert((cur, last_distance), true);
                    return true;
                }
            }
        }

        maps.insert((cur, last_distance), false);
        false
    }
    dfs(&stones, &mut maps, 0, 0)
}

/// p404
pub fn sum_of_left_leaves(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut res = 0;
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>, is_left: bool, res: &mut i32) {
        if let Some(content) = root {
            let left_node = &content.borrow().left;
            let right_node = &content.borrow().right;
            if *left_node == None && *right_node == None {
                if is_left {
                    *res = *res + content.borrow().val
                }
            } else {
                dfs(left_node, true, res);
                dfs(right_node, false, res);
            }
        }
    }
    dfs(&root, false, &mut res);
    res
}

/// p405
pub fn to_hex(num: i32) -> String {
    format!("{:x}", num)
}

/// p406
pub fn reconstruct_queue(people: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut people = people;
    let mut ans = vec![];
    people.sort_by(|a, b| b[0].cmp(&a[0]).then(a[1].cmp(&b[1])));
    for p in people.iter() {
        ans.insert(p[1] as usize, p.to_vec())
    }
    ans
}

/// p407
pub fn trap_rain_water(mut height_map: Vec<Vec<i32>>) -> i32 {
    let m = height_map.len();
    let n = height_map[0].len();
    let mut h = BinaryHeap::new();
    for (i, row) in height_map.iter_mut().enumerate() {
        for (j, height) in row.iter_mut().enumerate() {
            if i == 0 || i == m - 1 || j == 0 || j == n - 1 {
                h.push((-*height, i, j)); // 取相反数变成最小堆
                *height = -1; // 标记 (i,j) 访问过
            }
        }
    }

    let mut ans = 0;
    while let Some((min_height, i, j)) = h.pop() {
        // 去掉短板
        let min_height = -min_height; // min_height 是木桶的短板
        for (x, y) in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)] {
            if x < m && y < n && height_map[x][y] >= 0 {
                // (x,y) 没有访问过
                // 如果 (x,y) 的高度小于 min_height，那么接水量为 min_height - heightMap[x][y]
                ans += 0.max(min_height - height_map[x][y]);
                // 给木桶新增一块高为 max(min_height, heightMap[x][y]) 的木板
                h.push((-min_height.max(height_map[x][y]), x, y));
                height_map[x][y] = -1; // 标记 (x,y) 访问过
            }
        }
    }
    ans
}

/// p409
pub fn longest_palindrome(s: String) -> i32 {
    let chars = s.chars();
    let mut record: HashMap<char, usize> = HashMap::new();
    for c in chars {
        if let Some(count) = record.get_mut(&c) {
            *count = *count + 1;
        } else {
            record.insert(c, 1);
        }
    }
    let mut length = 0;
    for l in record.values() {
        length = length + l / 2 * 2;
        if l % 2 == 1 && length % 2 == 0 {
            length = length + 1
        }
    }
    length as i32
}

/// p410
pub fn split_array(nums: Vec<i32>, k: i32) -> i32 {
    let check = |mx: i32| -> bool {
        let mut cnt = 1;
        let mut s = 0;
        for &x in &nums {
            if s + x <= mx {
                s += x;
            } else {
                if cnt == k {
                    return false;
                }
                cnt += 1;
                s = x;
            }
        }
        true
    };

    let mut right = nums.iter().sum::<i32>();
    let mut left = (*nums.iter().max().unwrap() - 1).max((right - 1) / k);
    while left + 1 < right {
        let mid = left + (right - left) / 2;
        if check(mid) {
            right = mid;
        } else {
            left = mid;
        }
    }
    right
}

/// p412
pub fn fizz_buzz(n: i32) -> Vec<String> {
    let mut ans: Vec<String> = vec![];
    for i in 1..=n {
        if i % 15 == 0 {
            ans.push("FizzBuzz".to_string());
        } else if i % 5 == 0 {
            ans.push("Buzz".to_string());
        } else if i % 3 == 0 {
            ans.push("Fizz".to_string())
        } else {
            ans.push(i.to_string())
        }
    }
    ans
}

/// p413
pub fn number_of_arithmetic_slices(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    if n == 1 {
        return 0;
    }
    let (mut d, mut t, mut ans) = (nums[0] - nums[1], 0, 0);
    for i in 2..n {
        if nums[i - 1] - nums[i] == d {
            t = t + 1;
        } else {
            d = nums[i - 1] - nums[i];
            t = 0;
        }
        ans = ans + t;
    }
    return ans;
}

/// p414
pub fn third_max(nums: Vec<i32>) -> i32 {
    let (mut first, mut second, mut third) = (i64::MIN, i64::MIN, i64::MIN);
    for n in nums {
        let num = n as i64;
        if num > first {
            third = second;
            second = first;
            first = num;
        } else if num < first && num > second {
            third = second;
            second = num;
        } else if num < second && num > third {
            third = num;
        }
    }
    return if third == i64::MIN {
        first as i32
    } else {
        third as i32
    };
}

/// p415
pub fn add_strings(num1: String, num2: String) -> String {
    use std::iter::repeat;
    if num2.len() > num1.len() {
        return add_strings(num2, num1);
    }
    let mut prev = 0;
    let mut ret = num1
        .chars()
        .rev()
        .zip(
            num2.chars()
                .rev()
                .chain(repeat('0').take(num1.len().saturating_sub(num2.len()))),
        )
        .map(|(a, b)| {
            let curr = prev + a.to_digit(10).unwrap() + b.to_digit(10).unwrap();
            prev = curr / 10;
            char::from_digit(curr % 10, 10).unwrap()
        })
        .collect::<Vec<_>>();

    if prev == 1 {
        ret.push((1u8 + b'0') as char);
    }
    ret.iter().rev().collect::<_>()
}

/// p416
pub fn can_partition(nums: Vec<i32>) -> bool {
    let s = nums.iter().sum::<i32>() as usize;
    if s % 2 != 0 {
        return false;
    }
    fn dfs(i: usize, j: usize, nums: &[i32], memo: &mut [Vec<i32>]) -> bool {
        if i == nums.len() {
            return j == 0;
        }
        if memo[i][j] != -1 {
            // 之前计算过
            return memo[i][j] == 1;
        }
        let x = nums[i] as usize;
        let res = j >= x && dfs(i + 1, j - x, nums, memo) || dfs(i + 1, j, nums, memo);
        memo[i][j] = if res { 1 } else { 0 }; // 记忆化
        res
    }
    let n = nums.len();
    let mut memo = vec![vec![-1; s / 2 + 1]; n]; // -1 表示没有计算过
    // 为方便起见，改成 i 从 0 开始
    dfs(0, s / 2, &nums, &mut memo)
}

/// p417
pub fn pacific_atlantic(heights: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut to_pacific: HashSet<Vec<usize>> = HashSet::new();
    let mut to_atlantic: HashSet<Vec<usize>> = HashSet::new();
    let mut ans = Vec::new();

    fn dfs(r: usize, c: usize, heights: &Vec<Vec<i32>>, record: &mut HashSet<Vec<usize>>) {
        if record.contains(&vec![r, c]) {
            return;
        }
        record.insert(vec![r, c]);
        if r as i32 - 1 >= 0 && heights[r - 1][c] >= heights[r][c] {
            dfs(r - 1, c, heights, record)
        }
        if r + 1 < heights.len() && heights[r + 1][c] >= heights[r][c] {
            dfs(r + 1, c, heights, record)
        }
        if c as i32 - 1 >= 0 && heights[r][c - 1] >= heights[r][c] {
            dfs(r, c - 1, heights, record)
        }
        if c + 1 < heights[0].len() && heights[r][c + 1] >= heights[r][c] {
            dfs(r, c + 1, heights, record)
        }
    }

    for left in 0..heights.len() {
        dfs(left, 0, &heights, &mut to_pacific);
    }
    for top in 0..heights[0].len() {
        dfs(0, top, &heights, &mut to_pacific);
    }
    for right in 0..heights.len() {
        dfs(right, heights[0].len() - 1, &heights, &mut to_atlantic);
    }
    for bottom in 0..heights[0].len() {
        dfs(heights.len() - 1, bottom, &heights, &mut to_atlantic);
    }

    for p in to_pacific.iter() {
        for a in to_atlantic.iter() {
            if p[0] == a[0] && p[1] == a[1] {
                ans.push(vec![p[0] as i32, p[1] as i32])
            }
        }
    }
    ans
}

/// p419
pub fn count_battleships(board: Vec<Vec<char>>) -> i32 {
    let mut ans = 0;
    let row = board.len();
    let col = board[0].len();
    for r in 0..row {
        for c in 0..col {
            if board[r][c] == 'X' {
                if r > 0 && board[r - 1][c] == 'X' {
                    continue;
                }
                if c > 0 && board[r][c - 1] == 'X' {
                    continue;
                }
                ans = ans + 1;
            }
        }
    }
    ans
}

/// p420
pub fn strong_password_checker(password: String) -> i32 {
    let mut miss_types = 3;

    if password.chars().any(|c| c.is_lowercase()) {
        miss_types -= 1;
    }

    if password.chars().any(|c| c.is_uppercase()) {
        miss_types -= 1;
    }

    if password.chars().any(|c| c.is_digit(10)) {
        miss_types -= 1;
    }

    let n = password.len();
    let s = password.as_bytes();
    let (mut a, mut b, mut r) = (0, 0, 0);
    let mut i = 2;

    while i < n {
        if s[i] == s[i - 1] && s[i] == s[i - 2] {
            let mut tmp = 3;

            while i + 1 < n && s[i] == s[i + 1] {
                tmp += 1;
                i += 1;
            }

            if tmp % 3 == 0 {
                a += 1;
            } else if tmp % 3 == 1 {
                b += 1;
            }

            r += tmp / 3;
        }

        i += 1;
    }

    if n < 6 {
        return miss_types.max(6 - n as i32);
    }

    if n <= 20 {
        return miss_types.max(r);
    }

    let d = n as i32 - 20;
    r -= a.min(d);

    if d > a {
        r -= b.min((d - a) / 2);
    }

    if d > a + 2 * b {
        r -= (d - a - 2 * b) / 3;
    }

    d + r.max(miss_types)
}

/// p421
pub fn find_maximum_xor(nums: Vec<i32>) -> i32 {
    let mx = nums.iter().max().unwrap();
    let high_bit = 31 - mx.leading_zeros() as i32;

    let mut ans = 0;
    let mut mask = 0;
    let mut seen = HashSet::new();
    for i in (0..=high_bit).rev() {
        // 从最高位开始枚举
        seen.clear();
        mask |= 1 << i;
        let new_ans = ans | (1 << i); // 这个比特位可以是 1 吗？
        for &x in &nums {
            let x = x & mask; // 低于 i 的比特位置为 0
            if seen.contains(&(new_ans ^ x)) {
                ans = new_ans; // 这个比特位可以是 1
                break;
            }
            seen.insert(x);
        }
    }
    ans
}

/// p423
pub fn original_digits(s: String) -> String {
    let mut c: HashMap<char, usize> = HashMap::new();
    s.chars().for_each(|ch| {
        if let Some(count) = c.get_mut(&ch) {
            *count = *count + 1;
        } else {
            c.insert(ch, 1);
        }
    });
    let mut cnt = vec![0 as usize; 10];
    cnt[0] = *c.get(&'z').unwrap_or(&0);
    cnt[2] = *c.get(&'w').unwrap_or(&0);
    cnt[4] = *c.get(&'u').unwrap_or(&0);
    cnt[6] = *c.get(&'x').unwrap_or(&0);
    cnt[8] = *c.get(&'g').unwrap_or(&0);

    cnt[3] = *c.get(&'h').unwrap_or(&0) - cnt[8];
    cnt[5] = *c.get(&'f').unwrap_or(&0) - cnt[4];
    cnt[7] = *c.get(&'s').unwrap_or(&0) - cnt[6];

    cnt[1] = *c.get(&'o').unwrap_or(&0) - cnt[0] - cnt[2] - cnt[4];
    cnt[9] = *c.get(&'i').unwrap_or(&0) - cnt[5] - cnt[6] - cnt[8];

    let mut ans = "".to_string();
    for i in 0..10 {
        for _ in 0..cnt[i] {
            ans = ans + &i.to_string();
        }
    }
    return ans;
}

/// p424
pub fn character_replacement(s: String, k: i32) -> i32 {
    let mut records = [0; 26];
    let mut max_count = 0;
    let mut max_len = 0;

    let mut l = 0;
    for (i, b) in s.bytes().enumerate() {
        let idx = (b - b'A') as usize;
        records[idx] += 1;
        max_count = i32::max(max_count, records[idx]);

        while (i - l + 1) as i32 - max_count > k {
            records[(s.as_bytes()[l] - b'A') as usize] -= 1;
            l += 1;
        }

        max_len = i32::max(max_len, (i - l + 1) as i32);
    }

    max_len
}

/// p432
#[derive(Debug, PartialEq, Clone, Eq)]
struct Node {
    key: String,
    count: i32,
    prev: Option<Rc<RefCell<Node>>>,
    next: Option<Rc<RefCell<Node>>>,
}
impl Node {
    fn new(key: String) -> Self {
        Self {
            key,
            count: 1,
            prev: None,
            next: None,
        }
    }
}

struct DoubleList {
    head: Rc<RefCell<Node>>,
    tail: Rc<RefCell<Node>>,
}
impl DoubleList {
    fn new() -> Self {
        let mut head = Rc::new(RefCell::new(Node::new("".to_string())));
        let mut tail = Rc::new(RefCell::new(Node::new("".to_string())));
        head.borrow_mut().next = Some(tail.clone());
        tail.borrow_mut().prev = Some(head.clone());
        Self { head, tail }
    }

    fn get_max_key(&self) -> String {
        let next = self.head.borrow().next.clone().unwrap();
        if next == self.tail {
            "".to_string()
        } else {
            next.borrow().key.clone()
        }
    }
    fn get_min_key(&self) -> String {
        let prev = self.tail.borrow().prev.clone().unwrap();
        if prev == self.head {
            "".to_string()
        } else {
            prev.borrow().key.clone()
        }
    }

    fn insert_tail(&mut self, node: Rc<RefCell<Node>>) {
        let prev = self.tail.borrow_mut().prev.take().unwrap();
        prev.borrow_mut().next = Some(node.clone());
        self.tail.borrow_mut().prev = Some(node.clone());
        node.borrow_mut().prev = Some(prev);
        node.borrow_mut().next = Some(self.tail.clone());
    }

    fn remove(&mut self, node: Rc<RefCell<Node>>) {
        let prev = node.borrow_mut().prev.take().unwrap();
        let next = node.borrow_mut().next.take().unwrap();
        prev.borrow_mut().next = Some(next.clone());
        next.borrow_mut().prev = Some(prev);
    }
    fn insert(&mut self, node: Rc<RefCell<Node>>) {
        let count = node.borrow().count;
        let mut cur = self.head.borrow().next.clone().unwrap();
        while cur != self.tail && cur.borrow().count > count {
            let next = cur.borrow().next.clone().unwrap();
            cur = next;
        }

        let prev = cur.borrow_mut().prev.take().unwrap();
        prev.borrow_mut().next = Some(node.clone());
        cur.borrow_mut().prev = Some(node.clone());
        node.borrow_mut().prev = Some(prev);
        node.borrow_mut().next = Some(cur);
    }
    fn rearrange(&mut self, node: Rc<RefCell<Node>>) {
        self.remove(node.clone());
        self.insert(node);
    }
}

struct AllOne {
    maps: HashMap<String, Rc<RefCell<Node>>>,
    list: DoubleList,
}
impl AllOne {
    fn new() -> Self {
        Self {
            maps: HashMap::new(),
            list: DoubleList::new(),
        }
    }

    fn inc(&mut self, key: String) {
        match self.maps.get_mut(&key) {
            Some(node) => {
                node.borrow_mut().count += 1;
                self.list.rearrange(node.clone());
            }
            None => {
                let node = Rc::new(RefCell::new(Node::new(key.clone())));
                self.list.insert_tail(node.clone());
                self.maps.insert(key, node);
            }
        }
    }

    fn dec(&mut self, key: String) {
        let node = self.maps.get_mut(&key).unwrap();
        node.borrow_mut().count -= 1;
        if node.borrow().count == 0 {
            self.list.remove(self.maps.remove(&key).unwrap());
        } else {
            self.list.rearrange(node.clone());
        }
    }

    fn get_max_key(&self) -> String {
        self.list.get_max_key()
    }
    fn get_min_key(&self) -> String {
        self.list.get_min_key()
    }
}

/// p433
pub fn min_mutation(start_gene: String, end_gene: String, bank: Vec<String>) -> i32 {
    if start_gene == end_gene {
        return 0;
    }
    if !bank.contains(&end_gene) {
        return -1;
    }
    let gene = vec![
        "A".to_string(),
        "G".to_string(),
        "T".to_string(),
        "C".to_string(),
    ];
    let mut all_mut: Vec<Vec<String>> = vec![];
    let mut current_mut: Vec<String> = vec![start_gene.clone()];
    fn dfs(
        current_mut: &mut Vec<String>,
        all_mut: &mut Vec<Vec<String>>,
        bank: &Vec<String>,
        gene: &Vec<String>,
        start_gene: &String,
        end_gene: &String,
    ) {
        let last = current_mut[current_mut.len() - 1].to_string();
        for g in gene {
            for i in 0..start_gene.len() {
                let mut new = last.clone();
                new.replace_range(i..i + 1, &g.to_string());
                if new == *end_gene {
                    current_mut.push(new);
                    all_mut.push(current_mut.clone());
                    current_mut.pop();
                } else if bank.contains(&new) && !current_mut.contains(&new) {
                    current_mut.push(new);
                    dfs(current_mut, all_mut, bank, gene, start_gene, end_gene);
                    current_mut.pop();
                }
            }
        }
    }

    dfs(
        &mut current_mut,
        &mut all_mut,
        &bank,
        &gene,
        &start_gene,
        &end_gene,
    );
    if all_mut.len() == 0 {
        return -1;
    } else {
        (all_mut.iter().fold(usize::MAX, |acc, path| {
            if path.len() < acc {
                return path.len();
            } else {
                acc
            }
        }) - 1) as i32
    }
}

/// p434
pub fn count_segments(s: String) -> i32 {
    s.split_whitespace().count() as i32
}

/// p435
pub fn erase_overlap_intervals(intervals: Vec<Vec<i32>>) -> i32 {
    let length = intervals.len();
    if length == 0 {
        return 0;
    }
    let mut intervals = intervals;
    let mut count = 1;
    intervals.sort_by(|a, b| return a[1].cmp(&b[1]));
    let mut right = intervals[0][1];
    for i in 1..length {
        if intervals[i][0] >= right {
            right = intervals[i][1];
            count = count + 1;
        }
    }
    return length as i32 - count;
}

/// p446
pub fn find_right_interval(intervals: Vec<Vec<i32>>) -> Vec<i32> {
    let mut ans = vec![];
    for interval in intervals.iter() {
        let mut index = -1;
        let mut min = i32::MAX;
        for (location, pick) in intervals.iter().enumerate() {
            if pick[0] >= interval[1] && pick[0] < min {
                index = location as i32;
                min = pick[0];
            }
        }
        ans.push(index);
    }
    ans
}

/// p447
pub fn path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> i32 {
    fn dfs(
        node: &Option<Rc<RefCell<TreeNode>>>,
        s: i64,
        target_sum: i64,
        ans: &mut i32,
        cnt: &mut HashMap<i64, i32>,
    ) {
        if let Some(node) = node {
            let node = node.borrow();
            let s = s + node.val as i64;
            *ans += *cnt.get(&(s - target_sum)).unwrap_or(&0);
            *cnt.entry(s).or_insert(0) += 1;
            dfs(&node.left, s, target_sum, ans, cnt);
            dfs(&node.right, s, target_sum, ans, cnt);
            *cnt.entry(s).or_insert(0) -= 1;
        }
    }
    let mut ans = 0;
    let mut cnt = HashMap::new();
    cnt.insert(0, 1);
    dfs(&root, 0, target_sum as i64, &mut ans, &mut cnt);
    ans
}

/// p448
pub fn find_anagrams(s: String, p: String) -> Vec<i32> {
    // 1.
    let mut records = [0; 26];
    for b in p.bytes() {
        records[(b - b'a') as usize] += 1;
    }

    // 2.
    let mut result = vec![];
    for (i, w) in s.as_bytes().windows(p.len()).enumerate() {
        if i == 0 {
            for &b in w.iter() {
                records[(b - b'a') as usize] -= 1;
            }
        } else {
            records[(*w.last().unwrap() - b'a') as usize] -= 1;
        }

        if records.iter().all(|&count| count == 0) {
            result.push(i as i32);
        }
        records[(*w.first().unwrap() - b'a') as usize] += 1;
    }

    // 3.
    result
}

/// p440
pub fn find_kth_number(n: i32, mut k: i32) -> i32 {
    let mut curr: i32 = 1;
    while k > 1 {
        let (mut step, mut p, mut q): (i64, i64, i64) = (0, curr as i64, curr as i64 + 1);
        // 计算节点数
        while p <= n as i64 {
            step += q.min(n as i64 + 1) - p;
            p *= 10;
            q *= 10;
        }
        // 根据结果在下一棵树还是当前树做不同的处理
        if k - 1 >= step as i32 {
            k -= step as i32;
            curr += 1;
        } else {
            k -= 1;
            curr *= 10;
        }
    }
    curr as i32
}

/// p441
pub fn arrange_coins(n: i32) -> i32 {
    if n == 1 {
        return 1;
    }
    let mut rows = 1;
    let mut remain = n;
    while remain >= rows {
        remain = remain - rows;
        rows = rows + 1;
    }
    rows - 1
}

/// p442
pub fn find_duplicates(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    let n = nums.len();
    let mut ans = vec![];
    for i in 0..n {
        let x = nums[i].abs();
        if nums[(x - 1) as usize] > 0 {
            nums[(x - 1) as usize] = -nums[(x - 1) as usize];
        } else {
            ans.push(x);
        }
    }
    ans
}

/// p443
pub fn compress(chars: &mut Vec<char>) -> i32 {
    let n = chars.len();
    let mut idx = 0;
    let mut count = 1;

    for i in 1..n {
        if chars[i - 1] == chars[i] {
            count += 1;
        } else {
            chars[idx] = chars[i - 1];
            idx += 1;
            if count > 1 {
                for c in count.to_string().chars() {
                    chars[idx] = c;
                    idx += 1;
                }
            }
            count = 1;
        }
    }

    chars[idx] = chars[n - 1];
    idx += 1;
    if count > 1 {
        for c in count.to_string().chars() {
            chars[idx] = c;
            idx += 1;
        }
    }

    idx as i32
}

/// p445
pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut n_1 = vec![];
    let mut n_2 = vec![];
    let mut p_1 = &l1;
    let mut p_2 = &l2;
    while let Some(node) = p_1 {
        n_1.push(node.val);
        p_1 = &node.next;
    }
    while let Some(node) = p_2 {
        n_2.push(node.val);
        p_2 = &node.next;
    }
    let mut longer = n_1;
    let mut shorter = n_2;
    if longer.len() < shorter.len() {
        let temp = longer;
        longer = shorter;
        shorter = temp;
    }
    let mut remain = 0;
    let mut total = longer
        .iter()
        .rev()
        .zip(shorter.iter().rev().chain(iter::repeat(&0)))
        .map(|(num1, num2)| {
            let mut total = num1 + num2 + remain;
            if total >= 10 {
                remain = 1;
                total = total - 10;
            } else {
                remain = 0;
            }
            total
        })
        .collect::<Vec<i32>>();
    if remain != 0 {
        total.push(remain)
    }
    total.reverse();
    let mut res: Option<Box<ListNode>> = None;
    let mut p = &mut res;
    for n in total {
        let new_node = ListNode::new(n);
        if let Some(old_node) = p {
            old_node.next = Some(Box::new(new_node));
            p = &mut old_node.next;
        } else {
            *p = Some(Box::new(new_node));
        }
    }
    res
}

/// p446
pub fn number_of_arithmetic_slices_2(nums: Vec<i32>) -> i32 {
    let mut dp: Vec<HashMap<i64, i32>> = vec![HashMap::new(); nums.len()];

    let mut total_count = 0;
    for (i, &num) in nums.iter().enumerate() {
        for (j, &prev_num) in nums.iter().enumerate().take(i) {
            let d = num as i64 - prev_num as i64;
            let count = dp[j].get(&d).map(|count| *count).unwrap_or(0);
            *dp[i].entry(d).or_insert(0) += count + 1;
            total_count += count;
        }
    }

    total_count
}

/// p447
pub fn number_of_boomerangs(points: Vec<Vec<i32>>) -> i32 {
    let length = points.len();
    if length < 3 {
        return 0;
    } else {
        let mut ans = 0;
        for i in 0..length {
            let mut record: HashMap<i32, i32> = HashMap::new();
            for j in 0..length {
                let distance =
                    (points[i][0] - points[j][0]).pow(2) + (points[i][1] - points[j][1]).pow(2);
                if let Some(val) = record.get_mut(&distance) {
                    *val = *val + 1;
                } else {
                    record.insert(distance, 1);
                }
            }
            record.iter().for_each(|(_, val)| {
                if *val > 1 {
                    ans = ans + val * (val - 1);
                }
            });
        }
        ans
    }
}

/// p448
pub fn find_disappeared_numbers(nums: Vec<i32>) -> Vec<i32> {
    let mut cnt = vec![0; nums.len() + 1];
    for n in nums {
        cnt[n as usize] += 1;
    }
    let mut ans = vec![];
    for i in 1..cnt.len() {
        if cnt[i] == 0 {
            ans.push(i as i32);
        }
    }
    ans
}

/// p449
struct Codec {}

impl Codec {
    fn new() -> Self {
        Self {}
    }

    fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        let mut s = String::new();

        if let Some(root) = root {
            let mut q = VecDeque::new();
            s.push_str(&root.borrow().val.to_string());
            q.push_back(root);

            while let Some(node) = q.pop_front() {
                s.push('-');
                match node.borrow_mut().left.take() {
                    Some(left) => {
                        s.push_str(&left.borrow().val.to_string());
                        q.push_back(left);
                    }
                    None => {
                        s.push('#');
                    }
                }

                s.push('-');
                match node.borrow_mut().right.take() {
                    Some(right) => {
                        s.push_str(&right.borrow().val.to_string());
                        q.push_back(right);
                    }
                    None => {
                        s.push('#');
                    }
                }
            }
        }

        s
    }

    fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
        if data.is_empty() {
            return None;
        }

        // 1.
        let mut nodes = data
            .split('-')
            .map(|num| -> Option<Rc<RefCell<TreeNode>>> {
                match num.parse::<i32>() {
                    Ok(val) => Some(Rc::new(RefCell::new(TreeNode::new(val)))),
                    Err(_) => None,
                }
            })
            .collect::<Vec<_>>();

        // 2.
        let n = nodes.len();
        let (mut l, mut r) = (0, 1);

        while r < n {
            nodes[l].as_mut().unwrap().borrow_mut().left = nodes[r].clone();
            nodes[l].as_mut().unwrap().borrow_mut().right = nodes[r + 1].clone();

            r += 2;
            l += 1;
            while l < n && nodes[l].is_none() {
                l += 1;
            }
        }

        // 3.
        nodes[0].clone()
    }
}

/// p450
pub fn delete_node(root: Option<Rc<RefCell<TreeNode>>>, key: i32) -> Option<Rc<RefCell<TreeNode>>> {
    use std::cmp;
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, key: i32) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() {
            return None;
        }
        let mut node = root.as_ref().unwrap().borrow_mut();
        match node.val.cmp(&key) {
            cmp::Ordering::Greater => {
                node.left = dfs(node.left.take(), key);
            }
            cmp::Ordering::Less => {
                node.right = dfs(node.right.take(), key);
            }
            cmp::Ordering::Equal => {
                return match (node.left.is_none(), node.right.is_none()) {
                    (true, false) => node.right.take(),
                    (false, true) => node.left.take(),
                    (false, false) => {
                        let mut min_node = node.right.clone().unwrap();
                        while min_node.borrow().left.is_some() {
                            let t = min_node.borrow_mut().left.clone();
                            min_node = t.unwrap();
                        }
                        min_node.borrow_mut().left = node.left.take();
                        node.right.take()
                    }
                    _ => None,
                };
            }
        };
        root.clone()
    }
    dfs(root, key)
}

/// p451
pub fn frequency_sort(s: String) -> String {
    let mut record = HashMap::new();
    s.chars().for_each(|c| *record.entry(c).or_insert(0) += 1);
    let mut list = record.iter().map(|(c, _)| *c).collect::<Vec<char>>();
    list.sort_by_key(|c| -record[c]);
    list.iter()
        .map(|c| c.to_string().repeat(record[c] as usize))
        .collect::<Vec<String>>()
        .join("")
}

/// p452
pub fn find_min_arrow_shots(points: Vec<Vec<i32>>) -> i32 {
    if points.len() == 0 {
        return 0;
    }
    let mut points = points;
    points.sort_by(|a, b| a[1].cmp(&b[1]));
    let mut pos = points[0][1];
    let mut ans = 1;
    for balloon in points {
        if balloon[0] > pos {
            pos = balloon[1];
            ans += 1;
        }
    }
    return ans;
}

/// p453
pub fn min_moves(nums: Vec<i32>) -> i32 {
    let record = nums
        .iter()
        .fold((0, i32::MAX), |acc, cur| (acc.0 + cur, (*cur).min(acc.1)));
    record.0 - (nums.len() as i32) * record.1
}

/// p454
pub fn four_sum_count(nums1: Vec<i32>, nums2: Vec<i32>, nums3: Vec<i32>, nums4: Vec<i32>) -> i32 {
    let mut first_two = HashMap::new();
    nums1.iter().for_each(|n1| {
        nums2.iter().for_each(|n2| {
            let v = first_two.entry(n1 + n2).or_insert(0);
            *v = *v + 1;
        });
    });
    let mut count = 0;
    nums3.iter().for_each(|n3| {
        nums4.iter().for_each(|n4| {
            let c = first_two.get(&(-n3 - n4));
            if let Some(v) = c {
                count = count + v;
            }
        });
    });
    count
}

/// p455
pub fn find_content_children(g: Vec<i32>, s: Vec<i32>) -> i32 {
    let (mut g, mut s) = (g, s);
    g.sort_by(|a, b| a.cmp(&b));
    s.sort_by(|a, b| a.cmp(&b));
    let mut i = 0;
    for x in s {
        if i < g.len() && g[i] <= x {
            i += 1;
        }
    }
    i as i32
}

/// p456
pub fn find132pattern(nums: Vec<i32>) -> bool {
    let mut candidate_k = vec![];
    let mut max_k = i32::MIN;

    nums.into_iter().rev().any(|num| -> bool {
        if num < max_k {
            return true;
        }

        while let Some(val) = candidate_k.pop() {
            if val >= num {
                candidate_k.push(val);
                break;
            } else {
                max_k = val;
            }
        }
        if num > max_k {
            candidate_k.push(num);
        }

        false
    })
}

/// p457
fn circular_array_loop(mut nums: Vec<i32>) -> bool {
    let n = nums.len();

    fn next(nums: &[i32], i: usize) -> usize {
        let n = nums.len();
        let i = i as i32 + nums[i];
        let i = if i < 0 {
            n as i32 + (i % n as i32)
        } else {
            i % n as i32
        };

        (i as usize) % n
    }

    for i in 0..n {
        if next(&nums, i) == i {
            nums[i] = 0;
        }
    }

    for i in 0..n {
        if nums[i] == 0 {
            continue;
        }

        let mut sp = i;
        let mut fp = i;

        while nums[sp] * nums[next(&nums, fp)] > 0
            && nums[sp] * nums[next(&nums, next(&nums, fp))] > 0
        {
            sp = next(&nums, sp);
            fp = next(&nums, next(&nums, fp));

            if sp == fp {
                return true;
            }
        }

        let mut j = i;
        let v = nums[i];

        while nums[j] * v > 0 {
            let next = next(&nums, j);
            nums[j] = 0;
            j = next;
        }
    }

    false
}

/// p458
pub fn poor_pigs(buckets: i32, minutes_to_die: i32, minutes_to_test: i32) -> i32 {
    let r_plus_1 = minutes_to_test / minutes_to_die + 1;
    let mut p = 0;
    let mut b = 1;
    while b < buckets {
        p += 1;
        b *= r_plus_1;
    }
    p
}

/// p459
pub fn repeated_substring_pattern(s: String) -> bool {
    let s = s.chars().collect::<Vec<char>>();
    let len = s.len();
    if len == 0 {
        return false;
    };
    let mut next = vec![0; len];
    fn get_next(next: &mut Vec<usize>, s: &Vec<char>) {
        let len = s.len();
        let mut j = 0;
        for i in 1..len {
            while j > 0 && s[i] != s[j] {
                j = next[j - 1];
            }
            if s[i] == s[j] {
                j += 1;
            }
            next[i] = j;
        }
    }
    get_next(&mut next, &s);
    if next[len - 1] != 0 && len % (len - (next[len - 1])) == 0 {
        return true;
    }
    return false;
}

/// p460
struct LFUNode {
    key: i32,
    value: i32,
    freq: i32,
    prev: Option<Rc<RefCell<LFUNode>>>,
    next: Option<Rc<RefCell<LFUNode>>>,
}

impl LFUNode {
    fn new(key: i32, value: i32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(LFUNode {
            key,
            value,
            freq: 1,
            prev: None,
            next: None,
        }))
    }
}

struct LFUCache {
    capacity: usize,
    min_freq: i32,
    key_to_node: HashMap<i32, Rc<RefCell<LFUNode>>>,
    freq_to_dummy: HashMap<i32, Rc<RefCell<LFUNode>>>,
}

impl LFUCache {
    fn new(capacity: i32) -> Self {
        LFUCache {
            capacity: capacity as usize,
            min_freq: 0,
            key_to_node: HashMap::new(),
            freq_to_dummy: HashMap::new(),
        }
    }

    fn get_node(&mut self, key: i32) -> Option<Rc<RefCell<LFUNode>>> {
        if let Some(node) = self.key_to_node.get(&key) {
            // 有这本书
            let node = node.clone();
            Self::remove(node.clone()); // 把这本书抽出来
            let freq = node.borrow().freq;
            let dummy = self.freq_to_dummy.get(&freq).unwrap();
            if Rc::ptr_eq(dummy, dummy.borrow().prev.as_ref().unwrap()) {
                // 抽出来后，这摞书是空的
                self.freq_to_dummy.remove(&freq); // 移除空链表
                if self.min_freq == freq {
                    // 这摞书是最左边的
                    self.min_freq += 1;
                }
            }
            node.borrow_mut().freq += 1; // 看书次数 +1
            self.push_front(freq + 1, node.clone()); // 放在右边这摞书的最上面
            return Some(node);
        }
        None // 没有这本书
    }

    fn get(&mut self, key: i32) -> i32 {
        if let Some(node) = self.get_node(key) {
            // 有这本书
            return node.borrow().value;
        }
        -1 // 没有这本书
    }

    fn put(&mut self, key: i32, value: i32) {
        if let Some(node) = self.get_node(key) {
            // 有这本书
            node.borrow_mut().value = value; // 更新 value
            return;
        }
        if self.key_to_node.len() == self.capacity {
            // 书太多了
            let dummy = self.freq_to_dummy.get(&self.min_freq).unwrap();
            let back_node = dummy.borrow().prev.clone().unwrap(); // 最左边那摞书的最下面的书
            let key = back_node.borrow().key;
            self.key_to_node.remove(&key);
            Self::remove(back_node); // 移除
            if Rc::ptr_eq(dummy, dummy.borrow().prev.as_ref().unwrap()) {
                // 抽出来后，这摞书是空的
                self.freq_to_dummy.remove(&self.min_freq); // 移除空链表
            }
        }
        let node = LFUNode::new(key, value); // 新书
        self.key_to_node.insert(key, node.clone());
        self.push_front(1, node.clone()); // 放在「看过 1 次」的最上面
        self.min_freq = 1;
    }

    // 创建一个新的双向链表
    fn new_list() -> Rc<RefCell<LFUNode>> {
        let dummy = LFUNode::new(0, 0);
        dummy.borrow_mut().prev = Some(dummy.clone());
        dummy.borrow_mut().next = Some(dummy.clone());
        dummy
    }

    // 在链表头添加一个节点（把一本书放在最上面）
    fn push_front(&mut self, freq: i32, x: Rc<RefCell<LFUNode>>) {
        let dummy = self
            .freq_to_dummy
            .entry(freq)
            .or_insert_with(|| Self::new_list());
        let next = dummy.borrow().next.clone();
        x.borrow_mut().prev = Some(dummy.clone());
        x.borrow_mut().next = next.clone();
        dummy.borrow_mut().next = Some(x.clone());
        next.unwrap().borrow_mut().prev = Some(x);
    }

    fn remove(x: Rc<RefCell<LFUNode>>) {
        let prev = x.borrow().prev.clone().unwrap();
        let next = x.borrow().next.clone().unwrap();
        prev.borrow_mut().next = Some(next.clone());
        next.borrow_mut().prev = Some(prev);
    }
}

/// p461
pub fn hamming_distance(x: i32, y: i32) -> i32 {
    (x ^ y).count_ones() as i32
}

/// p462
pub fn min_moves2(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort();
    let (mut left, mut right) = (0, nums.len() - 1);
    let mut ans = 0;
    while left < right {
        ans += nums[right] - nums[left];
        left = left + 1;
        right = right - 1;
    }
    return ans;
}

/// p463
pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
    let mut perimeter = 0;
    for r in 0..grid.len() {
        for c in 0..grid[0].len() {
            if grid[r][c] == 1 {
                let mut contribution = 4;
                if (r as i32) - 1 >= 0 && grid[r - 1][c] == 1 {
                    contribution = contribution - 1;
                }
                if r + 1 < grid.len() && grid[r + 1][c] == 1 {
                    contribution = contribution - 1;
                }
                if (c as i32) - 1 >= 0 && grid[r][c - 1] == 1 {
                    contribution = contribution - 1;
                }
                if c + 1 < grid[0].len() && grid[r][c + 1] == 1 {
                    contribution = contribution - 1;
                }
                perimeter = perimeter + contribution;
            }
        }
    }
    perimeter
}

/// p464
pub fn can_i_win(max_choosable_integer: i32, desired_total: i32) -> bool {
    if desired_total == 0 {
        return true;
    }
    if max_choosable_integer * (1 + max_choosable_integer) / 2 < desired_total {
        return false;
    }
    fn dfs(curr: i32, choosable: i32, total: i32, dp: &mut Vec<i32>) -> bool {
        if total <= 0 {
            return false;
        }
        if dp[curr as usize] != 0 {
            return dp[curr as usize] == 1;
        }
        for i in 0..choosable {
            if (curr & (1 << i)) == 0 && !dfs(curr | (1 << i), choosable, total - i - 1, dp) {
                dp[curr as usize] = 1;
                return true;
            }
        }
        dp[curr as usize] = -1;
        false
    }
    dfs(
        0,
        max_choosable_integer,
        desired_total,
        &mut vec![0; 1 << max_choosable_integer],
    )
}

/// p466
pub fn get_max_repetitions(s1: String, n1: i32, s2: String, n2: i32) -> i32 {
    let mut map = vec![(-1, -1); s1.len()];
    let (bs1, bs2) = (s1.as_bytes(), s2.as_bytes());
    let (mut cnt1, mut cnt2) = (0, 0);
    let (mut idx1, mut idx2) = (0, 0);
    while cnt1 < n1 {
        idx2 += (bs1[idx1] == bs2[idx2]) as usize;
        idx1 += 1;
        if idx1 == bs1.len() {
            idx1 = 0;
            cnt1 += 1;
        }
        if idx2 == bs2.len() {
            idx2 = 0;
            cnt2 += 1;
            match map[idx1] {
                (-1, -1) => map[idx1] = (cnt1, cnt2),
                (prev_cnt1, prev_cnt2) => {
                    let repeat = (n1 - 1 - cnt1) / (cnt1 - prev_cnt1);
                    cnt1 += repeat * (cnt1 - prev_cnt1);
                    cnt2 += repeat * (cnt2 - prev_cnt2);
                    break;
                }
            }
        }
    }
    while cnt1 < n1 {
        idx2 += (bs1[idx1] == bs2[idx2]) as usize;
        idx1 += 1;
        if idx1 == bs1.len() {
            idx1 = 0;
            cnt1 += 1;
        }
        if idx2 == bs2.len() {
            idx2 = 0;
            cnt2 += 1;
        }
    }
    cnt2 / n2
}

/// p467
pub fn find_substring_in_wrapround_string(p: String) -> i32 {
    let (mut dp, p_arr, mut cnt, mut ret) = (vec![0; 32], p.as_bytes(), 0, 0);
    for i in 0..p.len() {
        if i > 0 && (p_arr[i] - p_arr[i - 1] == 1 || p_arr[i - 1] - p_arr[i] == 25) {
            cnt += 1;
        } else {
            cnt = 1
        }
        dp[(p_arr[i] - 'a' as u8) as usize] = cnt.max(dp[(p_arr[i] - 'a' as u8) as usize])
    }
    dp.iter().sum::<i32>()
}

/// p468
pub fn valid_ip_address(query_ip: String) -> String {
    use std::str::FromStr;
    fn is_ipv4(arr: Vec<&str>) -> bool {
        if arr.len() != 4 {
            return false;
        }
        for ip in arr {
            if ip.len() < 1 || ip.len() > 3 {
                return false;
            }
            let ip_num = ip.parse::<i32>();
            if ip_num.is_err()
                || ip_num.as_ref().unwrap() > &255
                || ip_num.as_ref().unwrap().to_string().len() != ip.len()
            {
                return false;
            }
        }
        true
    }
    fn is_ipv6(arr: Vec<&str>) -> bool {
        if arr.len() != 8 {
            return false;
        }
        for ip in arr {
            if ip.len() < 1 || ip.len() > 4 {
                return false;
            }
            for ch in ip.chars() {
                if !char::is_ascii_digit(&ch) && !(ch >= 'a' && ch <= 'f' || ch >= 'A' && ch <= 'F')
                {
                    return false;
                }
            }
        }
        true
    }

    if query_ip.contains(".") {
        return if !is_ipv4(query_ip.split(".").collect::<Vec<_>>()) {
            "Neither".to_string()
        } else {
            "IPv4".to_string()
        };
    }
    return if !is_ipv6(query_ip.split(":").collect::<Vec<_>>()) {
        "Neither".to_string()
    } else {
        "IPv6".to_string()
    };
}

/// p472
pub fn find_all_concatenated_words_in_a_dict(words: Vec<String>) -> Vec<String> {
    let mut set = HashSet::new();
    let mut ret = Vec::new();

    words.iter().for_each(|word| {
        set.insert(word);
    });

    fn can_break(s: &String, set: &HashSet<&String>) -> bool {
        let n = s.len();
        if set.is_empty() || n == 0 {
            return false;
        }

        let mut dp = vec![false; n + 1];
        dp[0] = true;

        for i in 1..n + 1 {
            for j in 0..i {
                if !dp[j] {
                    continue;
                }
                if set.contains(&s[j..i].to_string()) {
                    dp[i] = true;
                    break;
                }
            }
        }
        dp[n]
    }

    for word in words.iter() {
        if "" == word {
            continue;
        }
        set.remove(&word);

        if can_break(&word, &set) {
            ret.push(word.clone());
        }

        set.insert(&word);
    }

    ret
}

/// p473
pub fn makesquare(mut matchsticks: Vec<i32>) -> bool {
    let (mut m, n) = (matchsticks.len() as i32, matchsticks.iter().sum::<i32>());
    if n % 4 != 0 {
        return false;
    }

    fn dfs(m: i32, n: i32, side: &mut Vec<i32>, matchsticks: &Vec<i32>) -> bool {
        if m == -1 {
            return true;
        }
        for i in 0..4 {
            if side[i] == n
                || side[i] + matchsticks[m as usize] > n
                || (i > 0 && side[i] == side[i - 1])
            {
                continue;
            }
            side[i] += matchsticks[m as usize];
            if dfs(m - 1, n, side, matchsticks) {
                return true;
            }
            side[i] -= matchsticks[m as usize];
        }
        false
    }
    matchsticks.sort();
    dfs(m - 1, n / 4, &mut vec![0; 4], &matchsticks)
}

/// p474
pub fn find_max_form(strs: Vec<String>, m: i32, n: i32) -> i32 {
    fn count(s: &str) -> (usize, usize) {
        let m = s
            .chars()
            .fold(0, |acc, c| acc + if c == '0' { 1 } else { 0 });
        (m, s.len() - m)
    }
    let mut dp = vec![vec![0; m as usize + 1]; n as usize + 1];
    strs.iter().for_each(|s| {
        let (ms, ns) = count(s);
        for ni in (ns..=n as usize).rev() {
            for mi in (ms..=m as usize).rev() {
                dp[ni][mi] = dp[ni][mi].max(dp[ni - ns][mi - ms] + 1);
            }
        }
    });
    dp[n as usize][m as usize]
}

/// p475
pub fn find_radius(houses: Vec<i32>, heaters: Vec<i32>) -> i32 {
    let (mut houses, mut heaters) = (houses, heaters);
    houses.sort();
    heaters.sort();

    let mut i = 0;
    let mut ans = 0;
    for house in houses {
        let mut tmp = i32::MAX;
        while i < heaters.len() && (house - heaters[i]).abs() <= tmp {
            tmp = (house - heaters[i]).abs();
            i += 1;
        }
        ans = ans.max(tmp);
        i -= 1;
    }
    ans
}

/// p476
pub fn find_complement(num: i32) -> i32 {
    let count = 32 - num.leading_zeros();
    let mut new_num = 0;
    for i in 0..count {
        if num & (1 << i) == 0 {
            new_num |= 1 << i;
        }
    }
    new_num
}

/// p477
pub fn total_hamming_distance(nums: Vec<i32>) -> i32 {
    let mut result = 0;
    let mut zero = 0;
    for i in 0..32 {
        zero = 0;
        for j in 0..nums.len() {
            if (nums[j] >> i) & 1 == 1 {
                zero += 1;
            }
        }
        result += (nums.len() - zero) * zero;
    }
    result as i32
}
