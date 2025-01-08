use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashMap, HashSet},
    rc::Rc,
};

use crate::common::TreeNode;

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
    let mut step = -1;

    step
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

#[test]
fn test_eq() {
    original_digits("owoztneoer".to_string());
}
