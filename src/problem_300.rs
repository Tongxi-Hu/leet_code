use std::{cell::RefCell, rc::Rc};

use crate::common::{ListNode, TreeNode};

///p201
pub fn range_bitwise_and(left: i32, right: i32) -> i32 {
    let m = 32 - (left ^ right).leading_zeros();
    left & !((1 << m) - 1)
}

///p202
pub fn is_happy(n: i32) -> bool {
    let mut n = n;
    let mut history: std::collections::HashSet<i32> = std::collections::HashSet::new();
    while n != 1 {
        n = n
            .to_string()
            .chars()
            .map(|char| char.to_digit(10).unwrap())
            .fold(0, |acc, cur| return acc + (cur * cur) as i32);
        if n == 1 {
            return true;
        };
        match history.get(&n) {
            Some(_) => {
                return false;
            }
            None => {
                history.insert(n);
            }
        }
    }
    return true;
}

///p203
pub fn remove_elements(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut fake = Box::new(ListNode { val: 0, next: head });
    let mut pointer: &mut Box<ListNode> = &mut fake;
    while let Some(ref mut node) = pointer.next {
        if node.val == val {
            pointer.next = node.next.take();
        } else {
            pointer = pointer.next.as_mut().unwrap();
        }
    }
    return fake.next;
}

///p204
pub fn count_primes(n: i32) -> i32 {
    let n = n as usize;
    let mut d = vec![true; n];
    let mut count = 0;
    for i in 2..n {
        if d[i] {
            count += 1;
            let mut j = i * i;
            while j < n {
                d[j] = false;
                j += i;
            }
        }
    }
    count
}

/// p205
pub fn is_isomorphic(s: String, t: String) -> bool {
    let s_char = s.chars().collect::<Vec<char>>();
    let t_char = t.chars().collect::<Vec<char>>();
    if s_char.len() != t_char.len() {
        return false;
    };
    let mut s_to_t: std::collections::HashMap<char, char> = std::collections::HashMap::new();
    let mut t_to_s: std::collections::HashMap<char, char> = std::collections::HashMap::new();
    for i in 0..s_char.len() {
        match (s_to_t.get(&s_char[i]), t_to_s.get(&t_char[i])) {
            (Some(t), Some(s)) => {
                if *t == t_char[i] && *s == s_char[i] {
                    continue;
                } else {
                    return false;
                }
            }
            (None, Some(_)) | (Some(_), None) => {
                return false;
            }
            (None, None) => {
                s_to_t.insert(s_char[i], t_char[i]);
                t_to_s.insert(t_char[i], s_char[i]);
                continue;
            }
        }
    }
    return true;
}

/// p206
pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut pre = None;
    let mut cur = head;
    while let Some(mut node) = cur {
        let next = node.next;
        node.next = pre;
        pre = Some(node);
        cur = next;
    }
    return pre;
}

/// p207
pub fn can_finish(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
    let mut income = vec![0; num_courses as usize];
    let mut map = std::collections::HashMap::new();
    prerequisites.iter().for_each(|x| {
        let (a, b) = (x[0] as usize, x[1] as usize);
        income[a] += 1;
        map.entry(b).or_insert(vec![]).push(a);
    });
    let mut courses = std::collections::VecDeque::new();
    for (i, &n) in income.iter().enumerate() {
        if n == 0 {
            courses.push_back(i);
        }
    }

    let mut ans = 0;
    while !courses.is_empty() {
        ans += 1;
        let course = courses.pop_front().unwrap();
        if let Some(next) = map.get(&course) {
            for &n in next {
                income[n] -= 1;
                if income[n] == 0 {
                    courses.push_back(n);
                }
            }
        }
    }
    ans == num_courses
}

/// p208
#[derive(Default)]
pub struct Trie {
    root: Node,
}

#[derive(Default)]
struct Node {
    // Some(Box(next_node)) -> 父节点是子节点的所有者
    children: [Option<Box<Node>>; 26],
    is_word: bool,
}

impl Trie {
    pub fn new() -> Self {
        Self::default()
        // Trie {
        //     root: Node {
        //         children: [None; 26], -> 直接这样写是不行的，这种写法要求实现了Copy trait
        //         is_word: false,
        //     }
        // }
    }

    pub fn insert(&mut self, word: String) {
        let mut node = &mut self.root;
        for c in word.as_bytes() {
            let idx = (c - b'a') as usize;
            let next = &mut node.children[idx];
            // next.is_some() -> 直接取可变引用
            // next.is_none() -> 插入新的节点，再取其可变引用
            node = next.get_or_insert_with(Box::<Node>::default);
        }
        node.is_word = true;
    }

    pub fn search(&self, word: String) -> bool {
        self.get_node(&word).map_or(false, |w| w.is_word)
        // match self.get_node(&word) {
        //     Some(w) if w.is_word => true,
        //     _ => false
        // }
    }

    pub fn starts_with(&self, prefix: String) -> bool {
        self.get_node(&prefix).is_some()
    }

    /// 取 `s` 对应的节点，如果不存在则返回 `None`
    fn get_node(&self, s: &str) -> Option<&Node> {
        let mut node = &self.root;
        for c in s.as_bytes() {
            let idx = (c - b'a') as usize;
            match &node.children[idx] {
                Some(next) => node = next.as_ref(),
                None => return None,
            }
        }
        Some(node)
    }
}

/// p209
pub fn min_sub_array_len(target: i32, nums: Vec<i32>) -> i32 {
    let n = nums.len();
    let mut ans = n + 1;
    let mut sum = 0; // 子数组元素和
    let mut left = 0; // 子数组左端点
    for (right, &x) in nums.iter().enumerate() {
        // 枚举子数组右端点
        sum += x;
        while sum >= target {
            // 满足要求
            ans = ans.min(right - left + 1);
            sum -= nums[left]; // 左端点右移
            left += 1;
        }
    }
    if ans <= n {
        ans as i32
    } else {
        0
    }
}

/// p210
pub fn find_order(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> Vec<i32> {
    let mut indeg = vec![0; num_courses as usize];
    let mut m: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();

    for x in prerequisites {
        indeg[x[0] as usize] += 1;
        m.entry(x[1] as usize).or_default().push(x[0] as usize);
    }

    let mut count = 0;
    let mut ans: Vec<i32> = Vec::new();
    let mut q = std::collections::VecDeque::new();

    for i in 0..num_courses as usize {
        if indeg[i] == 0 {
            q.push_back(i);
        }
    }

    while !q.is_empty() {
        let v = q.pop_front().unwrap();
        count += 1;
        ans.push(v as i32);

        if let Some(x) = m.get(&v) {
            for &i in x {
                indeg[i] -= 1;
                if indeg[i] == 0 {
                    q.push_back(i);
                }
            }
        }
    }

    if count != num_courses {
        return vec![];
    }

    ans
}

/// p211
struct DictTree {
    is_end: bool,
    son: Vec<Option<DictTree>>,
}

impl DictTree {
    fn new(is_end: bool) -> Self {
        let mut son = Vec::with_capacity(26);
        (0..26).for_each(|_| son.push(None));
        Self { is_end, son }
    }
}

struct WordDictionary {
    tab: DictTree,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl WordDictionary {
    fn new() -> Self {
        Self {
            tab: DictTree::new(false),
        }
    }

    fn add_word(&mut self, word: String) {
        fn dfs(dt: &mut DictTree, s: &[u8], i: usize) {
            if i < s.len() {
                let c = s[i] as usize - b'a' as usize;
                let t = if let Some(t) = dt.son[c].as_mut() {
                    t
                } else {
                    dt.son[c] = Some(DictTree::new(false));
                    dt.son[c].as_mut().unwrap()
                };
                dfs(t, s, i + 1)
            } else {
                dt.is_end = true
            }
        }
        dfs(&mut self.tab, &mut word.as_bytes(), 0);
    }

    fn search(&self, word: String) -> bool {
        fn dfs(dt: &DictTree, s: &[u8], i: usize) -> bool {
            if i < s.len() {
                if s[i] == b'.' {
                    for son in dt.son.iter() {
                        if let Some(t) = son {
                            if dfs(t, s, i + 1) {
                                return true;
                            }
                        }
                    }
                } else {
                    let c = s[i] as usize - b'a' as usize;
                    if let Some(t) = dt.son[c].as_ref() {
                        return dfs(t, s, i + 1);
                    }
                }
                false
            } else {
                if dt.is_end {
                    true
                } else {
                    false
                }
            }
        }
        dfs(&self.tab, &word.as_bytes(), 0)
    }
}

///p212

///p213
pub fn rob(nums: Vec<i32>) -> i32 {
    fn rob1(nums: &Vec<i32>, start: usize, end: usize) -> i32 {
        let mut f0 = 0;
        let mut f1 = 0;
        for i in start..end {
            let new_f = f1.max(f0 + nums[i]);
            f0 = f1;
            f1 = new_f;
        }
        f1
    }

    let n = nums.len();
    rob1(&nums, 1, n).max(nums[0] + rob1(&nums, 2, n - 1))
}

///p214
pub fn shortest_palindrome(s: String) -> String {
    let mut palindrome_end = s.len();
    let chars = s.chars().collect::<Vec<char>>();
    if palindrome_end <= 1 || is_palindrome(&chars) {
        return s;
    }
    while palindrome_end >= 0 {
        if is_palindrome(&chars[0..palindrome_end]) {
            break;
        }
        palindrome_end -= 1;
    }

    chars[palindrome_end..]
        .iter()
        .rev()
        .chain(&chars)
        .collect::<String>()
}

fn is_palindrome(source: &[char]) -> bool {
    if source.len() == 1 {
        return true;
    }
    let mut i = 0;
    let mut j = source.len() - 1;
    while i <= j {
        if source[i] != source[j] {
            return false;
        }
        i += 1;
        j -= 1;
    }

    true
}

///p215
pub fn find_kth_largest(nums: Vec<i32>, k: i32) -> i32 {
    let mut heap = std::collections::BinaryHeap::with_capacity(k as usize);

    for n in nums {
        if heap.len() == heap.capacity() {
            if *heap.peek().unwrap() > std::cmp::Reverse(n) {
                heap.pop();
            } else {
                continue;
            }
        }
        heap.push(std::cmp::Reverse(n));
    }
    heap.peek().unwrap().0
}

///p216
pub fn combination_sum3(k: i32, n: i32) -> Vec<Vec<i32>> {
    let mut ans: Vec<Vec<i32>> = vec![];
    let mut temp: Vec<i32> = vec![];
    fn dfs(
        left: i32,
        right: i32,
        num: i32,
        sum: i32,
        ans: &mut Vec<Vec<i32>>,
        temp: &mut Vec<i32>,
    ) {
        if temp.len() as i32 + right - left + 1 < num || temp.len() as i32 > num {
            return;
        };
        if temp.len() as i32 == num && temp.iter().fold(0, |acc, cur| acc + cur) == sum {
            ans.push(temp.clone());
            return;
        }
        temp.push(left);
        dfs(left + 1, right, num, sum, ans, temp);
        temp.pop();
        dfs(left + 1, right, num, sum, ans, temp);
    }
    dfs(1, 9, k, n, &mut ans, &mut temp);
    return ans;
}

/// p217
pub fn contains_duplicate(nums: Vec<i32>) -> bool {
    let mut set: std::collections::HashSet<i32> = std::collections::HashSet::new();
    return !nums.iter().all(|&x| set.insert(x));
}

/// p218
pub fn get_skyline(buildings: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    // 1.
    let mut max_heap: std::collections::BinaryHeap<(i32, i32)> =
        std::collections::BinaryHeap::new();

    // 2.
    let mut build_edges = vec![];
    for build in buildings.iter() {
        build_edges.push(build[0]);
        build_edges.push(build[1]);
    }
    build_edges.sort_unstable();

    // 3.
    let n = buildings.len();
    let mut idx = 0_usize;
    let mut result: Vec<Vec<i32>> = vec![];

    for &edge in build_edges.iter() {
        while idx < n && buildings[idx][0] <= edge {
            max_heap.push((buildings[idx][2], buildings[idx][1]));
            idx += 1;
        }
        while let Some(&(_, edge_end)) = max_heap.peek() {
            if edge_end <= edge {
                max_heap.pop();
            } else {
                break;
            }
        }

        let cur_max_height = match max_heap.peek() {
            Some(&(height, _)) => height,
            None => 0,
        };
        if result.is_empty() || result.last().unwrap()[1] != cur_max_height {
            result.push(vec![edge, cur_max_height]);
        }
    }

    // 4.
    result
}

/// p219
pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
    let mut record: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for (index, num) in nums.iter().enumerate() {
        match record.get(num) {
            Some(last) => {
                if index - last <= k as usize {
                    return true;
                } else {
                    record.insert(*num, index);
                }
            }
            None => {
                record.insert(*num, index);
            }
        }
    }
    return false;
}

/// p220
fn neighbors<T: Ord>(tree: &std::collections::BTreeSet<T>, val: T) -> (Option<&T>, Option<&T>) {
    let mut before = tree.range(..&val);
    let mut after = tree.range(&val..);

    (before.next_back(), after.next())
}

pub fn contains_nearby_almost_duplicate(nums: Vec<i32>, k: i32, t: i32) -> bool {
    let t = t as i64;
    let k = k as i64;
    let mut tree: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();

    for i in 0..nums.len() {
        let u = nums[i] as i64;
        let tmp = neighbors(&tree, u);
        let (l, r) = tmp;
        if let Some(&l) = l {
            if u - l <= t {
                return true;
            }
        }

        if let Some(&r) = r {
            if r - u <= t {
                return true;
            }
        }
        tree.insert(u);
        if i >= k as usize {
            tree.remove(&(nums[i - k as usize] as i64));
        };
    }

    false
}

/// p221
pub fn maximal_square(matrix: Vec<Vec<char>>) -> i32 {
    let row = matrix.len();
    let col = matrix[0].len();
    let mut dp: Vec<Vec<usize>> = vec![vec![0; col]; row];
    for (r, items) in matrix.into_iter().enumerate() {
        for (c, item) in items.into_iter().enumerate() {
            if item == '0' {
                continue;
            } else if r == 0 || c == 0 {
                dp[r][c] = 1;
            } else {
                dp[r][c] = dp[r - 1][c].min(dp[r][c - 1].min(dp[r - 1][c - 1])) + 1;
            }
        }
    }
    let max = dp.into_iter().flatten().max().unwrap_or(0);
    return (max * max) as i32;
}

///p222
pub fn count_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut total = 0;
    match root {
        Some(node) => {
            let left_total = count_nodes(node.borrow().left.clone());
            let right_total = count_nodes(node.borrow().right.clone());
            total = 1 + left_total + right_total;
        }
        None => (),
    }
    return total;
}

///p223
pub fn compute_area(
    ax1: i32,
    ay1: i32,
    ax2: i32,
    ay2: i32,
    bx1: i32,
    by1: i32,
    bx2: i32,
    by2: i32,
) -> i32 {
    let rec_1 = (ax2 - ax1) * (ay2 - ay1);
    let rec_2 = (bx2 - bx1) * (by2 - by1);
    let overlap = 0.max(ax2.min(bx2) - ax1.max(bx1)) * 0.max(ay2.min(by2) - ay1.max(by1));
    return rec_1 + rec_2 - overlap;
}

/// p225
use std::collections::VecDeque;

struct MyStack {
    q1: VecDeque<i32>,
    q2: VecDeque<i32>,
    out1: bool,
}

impl MyStack {
    fn new() -> Self {
        Self {
            q1: VecDeque::new(),
            q2: VecDeque::new(),
            out1: false,
        }
    }

    fn push(&mut self, x: i32) {
        let (q1, q2) = if !self.out1 {
            (&mut self.q1, &mut self.q2)
        } else {
            (&mut self.q2, &mut self.q1)
        };
        q1.push_back(x);
        while let Some(val) = q2.pop_front() {
            q1.push_back(val);
        }
        self.out1 = !self.out1;
    }

    fn pop(&mut self) -> i32 {
        if self.out1 {
            self.q1.pop_front().unwrap()
        } else {
            self.q2.pop_front().unwrap()
        }
    }

    fn top(&mut self) -> i32 {
        if self.out1 {
            *self.q1.front().unwrap()
        } else {
            *self.q2.front().unwrap()
        }
    }

    fn empty(&self) -> bool {
        self.q1.is_empty() && self.q2.is_empty()
    }
}

/// p226
pub fn invert_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    match root {
        None => return None,
        Some(node) => {
            let old_left = invert_tree(node.borrow_mut().left.take());
            let old_right = invert_tree(node.borrow_mut().right.take());
            node.borrow_mut().left = old_right;
            node.borrow_mut().right = old_left;
            return Some(node);
        }
    }
}

/// p228
pub fn summary_ranges(nums: Vec<i32>) -> Vec<String> {
    if nums.len() == 0 {
        return vec![];
    }
    let mut start = nums[0];
    let mut cur = nums[0];
    let mut ans: Vec<String> = vec![];
    fn update_ans(start: i32, cur: i32, ans: &mut Vec<String>) {
        if start == cur {
            ans.push(start.to_string());
        } else {
            ans.push(start.to_string() + &"->".to_string() + &cur.to_string());
        }
    }
    for (index, num) in nums.into_iter().enumerate() {
        if index == 0 {
            continue;
        }
        if num == cur + 1 {
            cur = num;
            continue;
        } else {
            update_ans(start, cur, &mut ans);
            start = num;
            cur = num;
        }
    }
    update_ans(start, cur, &mut ans);
    return ans;
}

/// p229
pub fn majority_element(nums: Vec<i32>) -> Vec<i32> {
    let mut count: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for num in nums.iter() {
        match count.get(&num) {
            Some(val) => {
                count.insert(*num, val + 1);
            }
            None => {
                count.insert(*num, 1);
            }
        }
    }
    let length = nums.len();
    let mut ans: Vec<i32> = vec![];
    for (val, num) in count.into_iter() {
        if num > length / 3 {
            ans.push(val)
        }
    }
    return ans;
}

/// p230
pub fn kth_smallest(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> i32 {
    let mut value: Vec<i32> = vec![];
    fn in_order(root: Option<Rc<RefCell<TreeNode>>>, value: &mut Vec<i32>) {
        match root {
            None => {
                return;
            }
            Some(node) => {
                in_order(node.borrow_mut().left.take(), value);
                value.push(node.borrow().val);
                in_order(node.borrow_mut().right.take(), value);
            }
        }
    }
    in_order(root, &mut value);
    return value[(k - 1) as usize];
}

/// p231
pub fn is_power_of_two(n: i32) -> bool {
    if n == 0 {
        return false;
    }
    let mut n = n;
    while n % 2 == 0 {
        n = n / 2;
    }
    return n == 1;
}

///p232
struct MyQueue {
    stack: Vec<i32>,
    out: Vec<i32>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl MyQueue {
    /** Initialize your data structure here. */
    fn new() -> Self {
        Self {
            stack: vec![],
            out: vec![],
        }
    }

    /** Push element x to the back of queue. */
    fn push(&mut self, x: i32) {
        self.stack.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    fn pop(&mut self) -> i32 {
        if let Some(out) = self.out.pop() {
            out
        } else {
            loop {
                if let Some(s) = self.stack.pop() {
                    self.out.push(s);
                } else {
                    return self.out.pop().unwrap();
                }
            }
        }
    }

    /** Get the front element. */
    fn peek(&mut self) -> i32 {
        if let Some(out) = self.out.last() {
            *out
        } else {
            let mut out = 0;
            loop {
                if let Some(s) = self.stack.pop() {
                    out = s;
                    self.out.push(s);
                } else {
                    return out;
                }
            }
        }
    }

    /** Returns whether the queue is empty. */
    fn empty(&self) -> bool {
        self.stack.is_empty() && self.out.is_empty()
    }
}

///p233
pub fn count_digit_one(n: i32) -> i32 {
    let mut m = 1;
    let mut ans = 0;
    while m <= n {
        ans += n / (m * 10) * m + m.min(0.max(n % (m * 10) - m + 1));
        m *= 10;
    }
    ans
}

///p234
pub fn is_palindrome_1(head: Option<Box<ListNode>>) -> bool {
    let mut head = head;
    let mut content: Vec<i32> = vec![];
    while let Some(node) = head {
        content.push(node.val);
        head = node.next;
    }
    return content
        .iter()
        .zip(content.iter().rev())
        .all(|(left, right)| *left == *right);
}

///p235
pub fn lowest_common_ancestor(
    root: Option<Rc<RefCell<TreeNode>>>,
    p: Option<Rc<RefCell<TreeNode>>>,
    q: Option<Rc<RefCell<TreeNode>>>,
) -> Option<Rc<RefCell<TreeNode>>> {
    let root_val = root.as_ref().unwrap().borrow().val;
    let p_val = p.as_ref().unwrap().borrow().val;
    let q_val = q.as_ref().unwrap().borrow().val;
    if root_val > p_val && root_val > q_val {
        return lowest_common_ancestor(root.unwrap().borrow_mut().left.take(), p, q);
    } else if root_val < p_val && root_val < q_val {
        return lowest_common_ancestor(root.unwrap().borrow_mut().right.take(), p, q);
    } else {
        return root;
    }
}

///p236
pub fn lowest_common_ancestor_1(
    root: Option<Rc<RefCell<TreeNode>>>,
    p: Option<Rc<RefCell<TreeNode>>>,
    q: Option<Rc<RefCell<TreeNode>>>,
) -> Option<Rc<RefCell<TreeNode>>> {
    if root.is_none() || root == p || root == q {
        return root;
    }
    let x = root.as_ref().unwrap();
    let left = lowest_common_ancestor(x.borrow_mut().left.take(), p.clone(), q.clone());
    let right = lowest_common_ancestor(x.borrow_mut().right.take(), p, q);
    if left.is_some() && right.is_some() {
        return root;
    }
    if left.is_some() {
        left
    } else {
        right
    }
}

/// p238
pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len();
    let mut pre = vec![1; n];
    for i in 1..n {
        pre[i] = pre[i - 1] * nums[i - 1];
    }

    let mut suf = vec![1; n];
    for i in (0..n - 1).rev() {
        suf[i] = suf[i + 1] * nums[i + 1];
    }

    pre.iter().zip(suf.iter()).map(|(&p, &s)| p * s).collect()
}

///p239
pub fn max_sliding_window(nums: Vec<i32>, k: i32) -> Vec<i32> {
    let k = k as usize;
    let mut ans = Vec::with_capacity(nums.len() - k + 1);
    let mut q = std::collections::VecDeque::new();
    for (i, &x) in nums.iter().enumerate() {
        while !q.is_empty() && nums[*q.back().unwrap()] <= x {
            q.pop_back();
        }
        q.push_back(i);
        if i - q[0] >= k {
            q.pop_front();
        }
        if i >= k - 1 {
            ans.push(nums[q[0]]);
        }
    }
    ans
}

///p240
pub fn search_matrix(matrix: Vec<Vec<i32>>, target: i32) -> bool {
    let (row, col) = (matrix.len(), matrix[0].len());
    let (mut r, mut c) = (0, (col - 1) as i32);
    while r < row && c >= 0 {
        match matrix[r][c as usize].cmp(&target) {
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Less => {
                r = r + 1;
            }
            std::cmp::Ordering::Greater => c = c - 1,
        }
    }
    return false;
}

///p241
pub fn diff_ways_to_compute(expression: String) -> Vec<i32> {
    let mut res: Vec<i32> = vec![];
    let chars = expression.chars().collect::<Vec<char>>();
    let length = chars.len();
    if length == 0 {
        return res;
    }

    for i in 0..length {
        match chars[i] {
            '+' | '-' | '*' => {
                let left = diff_ways_to_compute(expression[..i].to_owned());
                let right = diff_ways_to_compute(expression[i + 1..].to_owned());
                for l in left.iter() {
                    for r in right.iter() {
                        match chars[i] {
                            '+' => res.push(l + r),
                            '-' => res.push(l - r),
                            '*' => res.push(l * r),
                            _ => (),
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    if res.is_empty() {
        res.push(expression.parse::<i32>().unwrap())
    }

    return res;
}
