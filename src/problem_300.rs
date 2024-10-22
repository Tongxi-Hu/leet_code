use std::{cell::RefCell, collections::HashSet, rc::Rc};

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
