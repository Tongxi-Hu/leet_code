use std::{
    cell::RefCell,
    cmp::{Ordering, Reverse},
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque},
    i32,
    rc::Rc,
};

use crate::common::{ListNode, TreeNode};

/// p701
pub fn insert_into_bst(
    root: Option<Rc<RefCell<TreeNode>>>,
    val: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    let mut root = root;
    if let Some(node) = root.as_ref() {
        let mut tree = node.borrow_mut();
        let v = tree.val;
        if v > val {
            tree.left = insert_into_bst(tree.left.clone(), val);
        } else {
            tree.right = insert_into_bst(tree.right.clone(), val)
        }
    } else {
        root = Some(Rc::new(RefCell::new(TreeNode::new(val))))
    }
    root
}

/// p703
struct KthLargest {
    heap: BinaryHeap<Reverse<i32>>,
}

impl KthLargest {
    fn new(k: i32, nums: Vec<i32>) -> Self {
        let mut kth = Self {
            heap: BinaryHeap::with_capacity(k as usize),
        };
        nums.iter().for_each(|&n| kth.push(n));
        kth
    }

    fn add(&mut self, val: i32) -> i32 {
        self.push(val);
        self.peek()
    }

    fn push(&mut self, n: i32) {
        if self.heap.len() == self.heap.capacity() {
            if *self.heap.peek().unwrap() > Reverse(n) {
                self.heap.pop();
            } else {
                return;
            }
        }
        self.heap.push(Reverse(n));
    }

    fn peek(&self) -> i32 {
        self.heap.peek().unwrap().0
    }
}

/// p704
pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    let (mut l, mut r) = (0, nums.len());
    while l < r {
        let mid = l + (r - l) / 2;
        if nums[mid] == target {
            return mid as i32;
        } else if nums[mid] > target {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    -1
}

/// p705
struct MyHashSet {
    arr: Vec<bool>,
}

impl MyHashSet {
    fn new() -> Self {
        Self {
            arr: vec![false; 1000001],
        }
    }

    fn add(&mut self, key: i32) {
        self.arr[key as usize] = true;
    }

    fn remove(&mut self, key: i32) {
        if !self.arr[key as usize] {
            return;
        }
        self.arr[key as usize] = false;
    }

    fn contains(&self, key: i32) -> bool {
        self.arr[key as usize]
    }
}

/// p706
struct MyHashMap {
    nums: Vec<i32>,
}

impl MyHashMap {
    fn new() -> Self {
        Self {
            nums: vec![-1; 1000001],
        }
    }

    fn put(&mut self, key: i32, value: i32) {
        self.nums[key as usize] = value;
    }

    fn get(&self, key: i32) -> i32 {
        self.nums[key as usize]
    }

    fn remove(&mut self, key: i32) {
        self.nums[key as usize] = -1;
    }
}

/// p707
struct MyListNode {
    prev: Option<Rc<RefCell<MyListNode>>>,
    next: Option<Rc<RefCell<MyListNode>>>,
    val: i32,
}

impl MyListNode {
    fn new(
        val: i32,
        prev: Option<Rc<RefCell<MyListNode>>>,
        next: Option<Rc<RefCell<MyListNode>>>,
    ) -> MyListNode {
        MyListNode { prev, next, val }
    }
}

struct MyLinkedList {
    head: Option<Rc<RefCell<MyListNode>>>,
    size: i32,
}

impl MyLinkedList {
    fn new() -> Self {
        let head = Rc::new(RefCell::new(MyListNode::new(-1, None, None)));
        head.borrow_mut().prev = Some(head.clone());
        head.borrow_mut().next = Some(head.clone());
        MyLinkedList {
            head: Some(head),
            size: 0,
        }
    }

    fn get(&self, index: i32) -> i32 {
        if index < 0 || index >= self.size {
            return -1;
        }
        self.find_node(index).as_ref().unwrap().borrow().val
    }

    fn add_at_head(&mut self, val: i32) {
        let next = self.head.as_ref().unwrap().borrow().next.clone();
        let curr = Rc::new(RefCell::new(MyListNode::new(
            val,
            self.head.clone(),
            next.clone(),
        )));
        next.clone().as_ref().unwrap().borrow_mut().prev = Some(curr.clone());
        self.head.as_ref().unwrap().borrow_mut().next = Some(curr.clone());
        self.size += 1;
    }

    fn add_at_tail(&mut self, val: i32) {
        let prev = self.head.as_ref().unwrap().borrow().prev.clone();
        let curr = Rc::new(RefCell::new(MyListNode::new(
            val,
            prev.clone(),
            self.head.clone(),
        )));
        prev.clone().as_ref().unwrap().borrow_mut().next = Some(curr.clone());
        self.head.as_ref().unwrap().borrow_mut().prev = Some(curr.clone());
        self.size += 1;
    }

    fn add_at_index(&mut self, index: i32, val: i32) {
        if index > self.size {
            return;
        }
        if index < 0 {
            self.add_at_head(val);
            return;
        }
        if index == self.size {
            self.add_at_tail(val);
            return;
        }
        let curr = self.find_node(index);
        let new_node = Rc::new(RefCell::new(MyListNode::new(
            val,
            curr.as_ref().unwrap().borrow().prev.clone(),
            curr.clone(),
        )));
        curr.as_ref()
            .unwrap()
            .borrow_mut()
            .prev
            .as_ref()
            .unwrap()
            .borrow_mut()
            .next = Some(new_node.clone());
        curr.as_ref().unwrap().borrow_mut().prev = Some(new_node.clone());
        self.size += 1;
    }

    fn delete_at_index(&mut self, index: i32) {
        if self.size <= 0 || index < 0 || index >= self.size {
            return;
        }
        if self.size == 1 {
            self.head.as_ref().unwrap().borrow_mut().prev = self.head.clone();
            self.head.as_ref().unwrap().borrow_mut().next = self.head.clone();
            self.size -= 1;
        }
        let curr = self.find_node(index);
        let (prev, next) = (
            curr.as_ref().unwrap().borrow().prev.clone(),
            curr.as_ref().unwrap().borrow().next.clone(),
        );
        prev.clone().as_ref().unwrap().borrow_mut().next = next.clone();
        next.clone().as_ref().unwrap().borrow_mut().prev = prev.clone();
        self.size -= 1;
    }

    fn find_node(&self, index: i32) -> Option<Rc<RefCell<MyListNode>>> {
        if self.size == 0 {
            return self.head.as_ref().unwrap().borrow().next.clone();
        }
        if self.size == index {
            return self.head.as_ref().unwrap().borrow().prev.clone();
        }
        let mut curr = self.head.as_ref().unwrap().borrow().next.clone();
        for _ in 0..index {
            let node = curr.as_ref().unwrap().borrow().next.clone();
            curr = node;
        }
        return curr;
    }
}

/// p709
pub fn to_lower_case(s: String) -> String {
    s.to_lowercase()
}

/// p712
pub fn minimum_delete_sum(s1: String, s2: String) -> i32 {
    let s1 = s1.bytes().collect::<Vec<_>>();
    let s2 = s2.bytes().collect::<Vec<_>>();
    let mut dp = vec![0; s2.len() + 1];
    s1.iter().for_each(|&c| {
        let mut pre = 0;
        for j in 1..=s2.len() {
            let tmp = dp[j];
            if c == s2[j - 1] {
                dp[j] = pre + c as i32;
            } else {
                dp[j] = dp[j].max(dp[j - 1]);
            }
            pre = tmp;
        }
    });
    let sum1: i32 = s1.iter().map(|&x| x as i32).sum();
    let sum2: i32 = s2.iter().map(|&x| x as i32).sum();
    sum1 + sum2 - 2 * dp[s2.len()]
}

/// p713
pub fn num_subarray_product_less_than_k(nums: Vec<i32>, k: i32) -> i32 {
    let mut count = 0;
    let mut product = 1;
    let mut l = 0;

    for r in 0..nums.len() {
        product = product * nums[r];
        while l <= r && product >= k {
            product = product / nums[l];
            l = l + 1;
        }
        count = count + r - l + 1;
    }

    count as i32
}

/// p714
pub fn max_profit(prices: Vec<i32>, fee: i32) -> i32 {
    let (mut sell, mut buy) = (0, -prices[0]);
    prices.iter().skip(1).for_each(|x| {
        sell = sell.max(buy + x - fee);
        buy = buy.max(sell - x);
    });
    sell
}

/// p715
struct RangeModule {
    range_tab: Vec<Range>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Range(i32, i32);

impl RangeModule {
    fn new() -> Self {
        Self { range_tab: vec![] }
    }

    fn add_range(&mut self, left: i32, right: i32) {
        if self.range_tab.is_empty() {
            self.range_tab.push(Range(left, right));
        } else {
            let idx_start = self.range_tab.partition_point(|Range(_, r)| *r < left);
            let i = idx_start;

            let mut new_left = left;
            let mut new_right = right;
            while i < self.range_tab.len() && right >= self.range_tab[i].0 {
                let l = self.range_tab[i].0;
                let r = self.range_tab[i].1;
                if l < new_left {
                    new_left = l;
                }
                if r > new_right {
                    new_right = r;
                }
                self.range_tab.remove(i);
            }

            self.range_tab.insert(i, Range(new_left, new_right));
        }
    }

    fn query_range(&self, left: i32, right: i32) -> bool {
        if self.range_tab.is_empty() {
            return false;
        }

        let idx_start = self.range_tab.partition_point(|Range(_, r)| *r < left);
        if idx_start == self.range_tab.len() || right <= self.range_tab[idx_start].0 {
            return false;
        }

        let mut i = idx_start;
        while i < self.range_tab.len() && right >= self.range_tab[i].0 {
            let l = self.range_tab[i].0;
            let r = self.range_tab[i].1;
            if l <= left && r >= right {
                return true;
            }
            i += 1;
        }

        false
    }

    fn remove_range(&mut self, left: i32, right: i32) {
        if self.range_tab.is_empty() {
            return;
        }
        let idx_start = self.range_tab.partition_point(|Range(_, r)| *r < left);
        let mut i = idx_start;

        while i < self.range_tab.len() && right >= self.range_tab[i].0 {
            let l = self.range_tab[i].0;
            let r = self.range_tab[i].1;
            //如[1,6)中移除[2,3)
            if l < left && r > right {
                self.range_tab[i].1 = left;
                self.range_tab.insert(i + 1, Range(right, r));
                return;
            }
            //如[1,4), [6, 10)中移除[2,8)
            //移除后要变成[1,2),[8,10)
            if l < left {
                self.range_tab[i].1 = left;
            } else if r > right {
                self.range_tab[i].0 = right;
            } else {
                self.range_tab.remove(i);
                continue;
            }
            i += 1;
        }
    }
}

/// p717
pub fn is_one_bit_character(bits: Vec<i32>) -> bool {
    let (mut i, length) = (0, bits.len());
    while i < length - 1 {
        i = i + bits[i] as usize + 1;
    }
    return i == length - 1;
}

/// p718
pub fn find_length(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let mut ans = 0;
    let mut f = vec![vec![0; nums2.len() + 1]; nums1.len() + 1];
    for (i, x) in nums1.into_iter().enumerate() {
        for (j, &y) in nums2.iter().enumerate() {
            if x == y {
                f[i + 1][j + 1] = f[i][j] + 1;
                ans = ans.max(f[i + 1][j + 1]);
            }
        }
    }
    ans
}

/// p719
pub fn smallest_distance_pair(mut nums: Vec<i32>, k: i32) -> i32 {
    nums.sort();
    let n = nums.len();
    let (mut l, mut r) = (0, nums[n - 1] - nums[0]);
    while l < r {
        let (mut cnt, mut j, mid) = (0, 0, l + ((r - l) >> 1));
        for i in 0..n {
            while j < n && nums[j] - nums[i] <= mid {
                j += 1;
            }
            cnt += (j - i - 1) as i32;
        }
        if cnt >= k { r = mid } else { l = mid + 1 }
    }
    l
}

/// p720
pub fn longest_word(mut words: Vec<String>) -> String {
    words.sort();
    words
        .iter()
        .fold(("", HashSet::new()), |(ret, mut cache), word| {
            if word.len() == 1 || cache.contains(&word[..word.len() - 1]) {
                cache.insert(word.as_str());
                (if ret.len() < word.len() { word } else { ret }, cache)
            } else {
                (ret, cache)
            }
        })
        .0
        .to_string()
}

/// p721
pub fn accounts_merge(accounts: Vec<Vec<String>>) -> Vec<Vec<String>> {
    let mut email_to_idx = HashMap::new();
    for (i, account) in accounts.iter().enumerate() {
        for email in account.iter().skip(1) {
            email_to_idx
                .entry(email.clone())
                .or_insert_with(Vec::new)
                .push(i);
        }
    }

    fn dfs(
        i: usize,
        accounts: &Vec<Vec<String>>,
        email_to_idx: &HashMap<String, Vec<usize>>,
        vis: &mut Vec<bool>,
        email_set: &mut HashSet<String>,
    ) {
        vis[i] = true;
        for email in accounts[i].iter().skip(1) {
            if email_set.contains(email) {
                continue;
            }
            email_set.insert(email.clone());
            for &j in email_to_idx.get(email).unwrap() {
                if !vis[j] {
                    dfs(j, accounts, email_to_idx, vis, email_set);
                }
            }
        }
    }

    let mut ans = vec![];
    let mut vis = vec![false; accounts.len()];
    for (i, account) in accounts.iter().enumerate() {
        if vis[i] {
            continue;
        }
        let mut email_set = HashSet::new();
        dfs(i, &accounts, &email_to_idx, &mut vis, &mut email_set);

        let mut res = email_set.into_iter().collect::<Vec<_>>();
        res.sort_unstable();
        res.insert(0, account[0].clone());

        ans.push(res);
    }
    ans
}

/// p722
pub fn remove_comments(source: Vec<String>) -> Vec<String> {
    let mut ans: Vec<String> = Vec::new();
    let mut t: Vec<String> = Vec::new();
    let mut block_comment = false;

    for s in &source {
        let m = s.len();
        let mut i = 0;
        while i < m {
            if block_comment {
                if i + 1 < m && &s[i..i + 2] == "*/" {
                    block_comment = false;
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                if i + 1 < m && &s[i..i + 2] == "/*" {
                    block_comment = true;
                    i += 2;
                } else if i + 1 < m && &s[i..i + 2] == "//" {
                    break;
                } else {
                    t.push(s.chars().nth(i).unwrap().to_string());
                    i += 1;
                }
            }
        }
        if !block_comment && !t.is_empty() {
            ans.push(t.join(""));
            t.clear();
        }
    }
    ans
}

/// p724
pub fn pivot_index(nums: Vec<i32>) -> i32 {
    let mut sum: Vec<i32> = vec![];
    nums.iter().enumerate().for_each(|(i, &num)| {
        if i == 0 {
            sum.push(num);
        } else {
            sum.push(num + sum[i - 1]);
        }
    });
    let length = nums.len();
    if sum[length - 1] - nums[0] == 0 {
        return 0;
    }

    for i in 1..length - 1 {
        if sum[i - 1] as f32 == (sum[length - 1] - nums[i]) as f32 / 2.0 {
            return i as i32;
        }
    }

    if sum[length - 1] - nums[length - 1] == 0 {
        return (length - 1) as i32;
    }

    return -1;
}

/// p725
pub fn split_list_to_parts(root: Option<Box<ListNode>>, k: i32) -> Vec<Option<Box<ListNode>>> {
    let k = k as usize;
    let mut cnt = 0usize;
    let mut ptr = root.as_ref();

    while let Some(node) = ptr {
        cnt += 1;
        ptr = node.next.as_ref();
    }

    let mut vec = vec![];
    let (a, b) = (cnt / k, cnt % k);
    let mut node = root;

    for i in 0..k {
        let mut num = a;
        if i < b {
            num += 1;
        }
        if num == 0 {
            vec.push(None);
            continue;
        }
        vec.push(node);
        let mut ptr = vec[i].as_mut().unwrap();
        for _ in 0..num - 1 {
            ptr = ptr.next.as_mut().unwrap();
        }
        node = ptr.next.take();
    }
    vec
}

/// p726
type Atom<'a> = (&'a [u8], usize);
type Atoms<'a> = std::collections::BTreeMap<&'a [u8], usize>;

pub struct Parser<'a> {
    src: &'a [u8],
    idx: usize,
}

impl<'a> Parser<'a> {
    pub fn new(src: &str) -> Parser<'_> {
        Parser {
            src: src.as_bytes(),
            idx: 0,
        }
    }

    pub fn r#try<T>(&mut self, f: impl FnOnce(&mut Self) -> Option<T>) -> Option<T> {
        let mut s = Self {
            idx: self.idx,
            src: self.src,
        };
        match f(&mut s) {
            Some(t) => {
                self.idx = s.idx;
                Some(t)
            }
            None => None,
        }
    }
}

fn get_number(p: &mut Parser) -> Option<usize> {
    let nums = p.src[p.idx..]
        .iter()
        .take_while(|c| c.is_ascii_digit())
        .copied()
        .collect::<Vec<_>>();
    p.idx += nums.len();
    unsafe { String::from_utf8_unchecked(nums) }
        .parse::<usize>()
        .ok()
}

fn get_symbol<'a>(p: &mut Parser<'a>) -> Option<&'a [u8]> {
    let start_idx = p.idx;
    let mut iter = p.src[p.idx..].iter();
    if !iter.next()?.is_ascii_uppercase() {
        return None;
    }
    let len = iter.take_while(|c| c.is_ascii_lowercase()).count();
    p.idx += len + 1;
    Some(&p.src[start_idx..p.idx])
}

fn get_single<'a>(p: &mut Parser<'a>) -> Option<Atom<'a>> {
    let symbol = p.r#try(get_symbol)?;
    let number = p.r#try(get_number).unwrap_or(1);
    Some((symbol, number))
}

fn get_bracket<'a>(p: &mut Parser<'a>) -> Option<Atoms<'a>> {
    if !p.src.get(p.idx)? == b'(' {
        return None;
    }
    p.idx += 1;

    let mut map = get_multiple(p)?;
    if !p.src.get(p.idx)? == b')' {
        return None;
    }

    p.idx += 1;
    let scal = p.r#try(get_number).unwrap_or(1);
    for v in map.values_mut() {
        *v *= scal;
    }
    Some(map)
}

fn get_single_or_bracket<'a>(p: &mut Parser<'a>) -> Option<Atoms<'a>> {
    let start = *p.src.get(p.idx)?;
    if start == b'(' {
        p.r#try(get_bracket)
    } else {
        p.r#try(get_single).map(|kv| std::iter::once(kv).collect())
    }
}

fn get_multiple<'a>(p: &mut Parser<'a>) -> Option<Atoms<'a>> {
    let mut map = Atoms::new();
    while let Some(sub_map) = p.r#try(get_single_or_bracket) {
        for (k, v) in sub_map {
            *map.entry(k).or_default() += v;
        }
    }
    Some(map)
}

pub fn count_of_atoms(formula: String) -> String {
    let mut parser = Parser::new(&formula);
    get_multiple(&mut parser)
        .unwrap()
        .into_iter()
        .fold(String::new(), |buffer, (k, v)| {
            buffer
                + &format!(
                    "{}{}",
                    String::from_utf8_lossy(k),
                    if v == 1 { "".to_owned() } else { v.to_string() }
                )
        })
}

/// p728
pub fn self_dividing_numbers(left: i32, right: i32) -> Vec<i32> {
    (left..=right)
        .into_iter()
        .filter(|&x| {
            let mut t = x;
            while t > 0 {
                let m = t % 10;
                if m == 0 || x % m != 0 {
                    return false;
                }
                t = t / 10;
            }
            true
        })
        .collect()
}

/// p729
#[derive(Default)]
struct MyCalendar {
    map: std::collections::BTreeMap<i32, i32>,
}

impl MyCalendar {
    fn new() -> Self {
        Default::default()
    }

    fn book(&mut self, start_time: i32, end_time: i32) -> bool {
        if let Some((_, &prev_end_time)) = self.map.range(..end_time).next_back() {
            if prev_end_time > start_time {
                return false;
            }
        }
        self.map.insert(start_time, end_time);
        true
    }
}

/// p730
pub fn count_palindromic_subsequences(s: String) -> i32 {
    let n = s.len();
    let p = 1000000007;
    let mut dp = vec![vec![vec![0; 4]; n]; n];
    let sv = s.as_bytes();

    for i in (0..n).rev() {
        for j in i..n {
            for k in 0..4 {
                let c = b'a' + k as u8;
                if i == j {
                    if sv[i] == c {
                        dp[i][j][k] = 1
                    }
                } else {
                    if sv[i] != c {
                        dp[i][j][k] = dp[i + 1][j][k];
                    } else if sv[j] != c {
                        dp[i][j][k] = dp[i][j - 1][k];
                    } else {
                        if j == i + 1 {
                            dp[i][j][k] = 2;
                        } else {
                            dp[i][j][k] = 2;
                            for m in 0..4 {
                                dp[i][j][k] += dp[i + 1][j - 1][m];
                                dp[i][j][k] %= p;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut ans = 0;

    for k in 0..4 {
        ans += dp[0][n - 1][k];
        ans %= p;
    }

    ans
}

/// p731
const MAX_TIME: i32 = 1_0000_0000_0;

struct SegmentTree {
    start: i32,
    end: i32,
    max_count: i32,
    lazy_flag: i32,
    left_node: Option<Rc<RefCell<SegmentTree>>>,
    right_node: Option<Rc<RefCell<SegmentTree>>>,
}

impl SegmentTree {
    fn new(start: i32, end: i32) -> Self {
        Self {
            start,
            end,
            max_count: 0,
            lazy_flag: 0,
            left_node: None,
            right_node: None,
        }
    }

    fn update_cur_node(&mut self, count: i32) {
        self.max_count += count;
        self.lazy_flag += count;
    }

    fn update(&mut self, left: i32, right: i32) {
        if left >= self.end || right <= self.start {
            return;
        }
        if left <= self.start && right >= self.end {
            self.update_cur_node(1);
            return;
        }

        self.update_cross_range(left, right);
    }

    fn update_cross_range(&mut self, left: i32, right: i32) {
        let mid = self.start + (self.end - self.start) / 2;
        let left_node = self
            .left_node
            .get_or_insert(Rc::new(RefCell::new(SegmentTree::new(self.start, mid))));
        let right_node = self
            .right_node
            .get_or_insert(Rc::new(RefCell::new(SegmentTree::new(mid, self.end))));

        if self.lazy_flag > 0 {
            left_node.borrow_mut().update_cur_node(self.lazy_flag);
            right_node.borrow_mut().update_cur_node(self.lazy_flag);
            self.lazy_flag = 0;
        }

        left_node.borrow_mut().update(left, right);
        right_node.borrow_mut().update(left, right);

        self.max_count = i32::max(left_node.borrow().max_count, right_node.borrow().max_count);
    }

    fn query(&mut self, left: i32, right: i32) -> i32 {
        if left >= self.end || right <= self.start {
            return 0;
        }
        if left <= self.start && right >= self.end {
            return self.max_count;
        }

        self.query_cross_range(left, right)
    }

    fn query_cross_range(&mut self, left: i32, right: i32) -> i32 {
        let mid = self.start + (self.end - self.start) / 2;
        let left_node = self
            .left_node
            .get_or_insert(Rc::new(RefCell::new(SegmentTree::new(self.start, mid))));
        let right_node = self
            .right_node
            .get_or_insert(Rc::new(RefCell::new(SegmentTree::new(mid, self.end))));

        if self.lazy_flag > 0 {
            left_node.borrow_mut().update_cur_node(self.lazy_flag);
            right_node.borrow_mut().update_cur_node(self.lazy_flag);
            self.lazy_flag = 0;
        }

        i32::max(
            left_node.borrow_mut().query(left, right),
            right_node.borrow_mut().query(left, right),
        )
    }
}

struct MyCalendarTwo {
    root: SegmentTree,
}

impl MyCalendarTwo {
    fn new() -> Self {
        Self {
            root: SegmentTree::new(0, MAX_TIME),
        }
    }

    fn book(&mut self, start: i32, end: i32) -> bool {
        if self.root.query(start, end) >= 2 {
            return false;
        }
        self.root.update(start, end);
        true
    }
}

/// p732
struct MyCalendarThree {
    map: BTreeMap<i32, i32>,
}

impl MyCalendarThree {
    fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    fn book(&mut self, start_time: i32, end_time: i32) -> i32 {
        *self.map.entry(start_time).or_insert(0) += 1;
        *self.map.entry(end_time).or_insert(0) -= 1;

        let (mut ans, mut step) = (0, 0);
        for &count in self.map.values() {
            step += count;
            ans = ans.max(step);
        }

        ans
    }
}

/// p733
pub fn flood_fill(mut image: Vec<Vec<i32>>, sr: i32, sc: i32, color: i32) -> Vec<Vec<i32>> {
    let (r, c) = (sr as usize, sc as usize);
    if image[r][c] == color {
        return image;
    }
    let mut stack: Vec<(usize, usize)> = vec![];
    let old_color = image[r][c];
    image[r][c] = color;
    stack.push((r, c));
    while !stack.is_empty() {
        let (r, c) = stack.pop().unwrap();
        for (i, j) in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)] {
            if i < image.len() && j < image[0].len() && image[i][j] == old_color {
                image[i][j] = color;
                stack.push((i, j));
            }
        }
    }
    image
}

/// p735
pub fn asteroid_collision(mut asteroids: Vec<i32>) -> Vec<i32> {
    let (mut index, mut i): (i32, usize) = (-1, 0);
    while i < asteroids.len() {
        if asteroids[i] > 0 || index == -1 || asteroids[index as usize] < 0 {
            index += 1;
            asteroids[index as usize] = asteroids[i];
        } else if asteroids[index as usize] <= -asteroids[i] {
            if asteroids[index as usize] < -asteroids[i] {
                i -= 1;
            }
            index -= 1;
        }
        i += 1;
    }
    asteroids[0..(index + 1).max(0) as usize].to_vec()
}

/// p736
pub fn evaluate(expression: String) -> i32 {
    let mut tokens = Tokens::new(expression.as_str());
    let mut env = Environment::new();
    eval(&mut tokens, &mut env)
}

fn eval<'a>(tokens: &mut Tokens<'a>, env: &mut Environment<'a>) -> i32 {
    match tokens.next().unwrap() {
        Token::Num(n) => n,
        Token::Var(v) => env.get(v),
        Token::Start => match tokens.next().unwrap() {
            Token::Add => {
                let left = eval(tokens, env);
                let right = eval(tokens, env);
                let res = left + right;
                assert_eq!(Token::End, tokens.next().unwrap());
                res
            }
            Token::Mul => {
                let left = eval(tokens, env);
                let right = eval(tokens, env);
                let res = left * right;
                assert_eq!(Token::End, tokens.next().unwrap());
                res
            }
            Token::Let => {
                let mut vars_to_pop = vec![];
                loop {
                    match tokens.peek().unwrap() {
                        Token::Var(v) => {
                            tokens.next().unwrap();
                            if tokens.peek().unwrap() == Token::End {
                                let res = env.get(v);
                                for var in vars_to_pop {
                                    env.pop(var);
                                }
                                assert_eq!(Token::End, tokens.next().unwrap());
                                return res;
                            } else {
                                let val = eval(tokens, env);
                                env.put(v, val);
                                vars_to_pop.push(v);
                            }
                        }
                        _ => {
                            let res = eval(tokens, env);
                            for var in vars_to_pop {
                                env.pop(var);
                            }
                            assert_eq!(Token::End, tokens.next().unwrap());
                            return res;
                        }
                    }
                }
            }
            _ => panic!("Invalid expr after '('"),
        },
        _ => panic!("Invalid start of expression"),
    }
}

struct Tokens<'a> {
    expr: &'a [u8],
    start: usize,
    end: usize,
}

#[derive(Debug, PartialEq, Eq)]
enum Token<'a> {
    Start,
    End,
    Add,
    Mul,
    Let,
    Num(i32),
    Var(&'a [u8]),
}

impl<'a> Tokens<'a> {
    fn new(expr: &'a str) -> Self {
        Tokens {
            expr: expr.as_bytes(),
            start: 0,
            end: 0,
        }
    }

    fn peek(&mut self) -> Option<Token<'a>> {
        let start = self.start;
        let end = self.end;
        let res = self.next();
        self.start = start;
        self.end = end;
        res
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.end == self.expr.len() {
            return None;
        }
        while self.expr[self.end] == b' ' {
            self.end += 1;
            if self.end == self.expr.len() {
                return None;
            }
        }
        self.start = self.end;
        self.end += 1;
        match self.expr[self.start] {
            b'(' => return Some(Token::Start),
            b')' => return Some(Token::End),
            _ => {}
        }
        while !(self.expr[self.end] == b' ' || self.expr[self.end] == b')') {
            self.end += 1;
            if self.end == self.expr.len() {
                return None;
            }
        }
        if self.expr[self.start] == b'-'
            || (self.expr[self.start] >= b'0' && self.expr[self.start] <= b'9')
        {
            let as_str = unsafe { std::str::from_utf8_unchecked(&self.expr[self.start..self.end]) };
            return Some(Token::Num(as_str.parse::<i32>().unwrap()));
        }
        let token = match &self.expr[self.start..self.end] {
            b"mult" => Token::Mul,
            b"add" => Token::Add,
            b"let" => Token::Let,
            var => Token::Var(var),
        };
        Some(token)
    }
}

struct Environment<'a> {
    vars: HashMap<&'a [u8], Vec<i32>>,
}

impl<'a> Environment<'a> {
    fn new() -> Self {
        Environment {
            vars: HashMap::new(),
        }
    }

    fn get(&self, var: &[u8]) -> i32 {
        self.vars.get(var).map(|v| v[v.len() - 1]).unwrap()
    }

    fn put(&mut self, var: &'a [u8], val: i32) {
        let entry = self.vars.entry(var).or_insert(vec![]);
        (*entry).push(val);
    }

    fn pop(&mut self, var: &[u8]) {
        self.vars.get_mut(var).map(|v| v.pop().unwrap()).unwrap();
    }
}

/// p738
pub fn monotone_increasing_digits(n: i32) -> i32 {
    let mut ans = 0;
    let mut ones = 111111111;
    (0..9).for_each(|_| {
        while ans + ones > n {
            ones /= 10;
        }
        ans += ones;
    });
    ans
}

/// p739
pub fn daily_temperatures(temperatures: Vec<i32>) -> Vec<i32> {
    let mut ans = vec![0; temperatures.len()];
    let mut stack: Vec<(usize, i32)> = vec![];
    for (i, &t) in temperatures.iter().enumerate() {
        while !stack.is_empty() && stack.last().unwrap().1 < t {
            let (ii, _) = stack.pop().unwrap();
            ans[ii] = (i - ii) as i32;
        }
        stack.push((i, t));
    }
    ans
}

/// p740
pub fn delete_and_earn(nums: Vec<i32>) -> i32 {
    let mut sum = vec![0; *nums.iter().max().unwrap() as usize + 1];
    nums.iter().for_each(|&n| sum[n as usize] += n);
    let (mut first, mut second) = (sum[0], sum[0].max(sum[1]));
    sum.iter().skip(2).for_each(|&val| {
        let cur = second.max(first + val);
        first = second;
        second = cur;
    });
    first.max(second)
}

/// p741
pub fn cherry_pickup(grid: Vec<Vec<i32>>) -> i32 {
    let n = grid.len();
    let mut dp = vec![vec![vec![i32::MIN; n]; n]; 2 * n - 1];

    dp[0][0][0] = grid[0][0];

    for k in 1..(2 * n - 1) {
        for r1 in std::cmp::max(0, k as i32 - (n as i32 - 1)) as usize..=std::cmp::min(n - 1, k) {
            let c1 = k - r1;
            if c1 >= n || grid[r1][c1] == -1 {
                continue;
            }

            for r2 in std::cmp::max(0, k as i32 - (n as i32 - 1)) as usize..=std::cmp::min(n - 1, k)
            {
                let c2 = k - r2;
                if c2 >= n || grid[r2][c2] == -1 {
                    continue;
                }

                let mut max_cherries = i32::MIN;

                if r1 > 0 {
                    max_cherries = std::cmp::max(max_cherries, dp[k - 1][r1 - 1][r2]);
                }
                if r2 > 0 {
                    max_cherries = std::cmp::max(max_cherries, dp[k - 1][r1][r2 - 1]);
                }
                if r1 > 0 && r2 > 0 {
                    max_cherries = std::cmp::max(max_cherries, dp[k - 1][r1 - 1][r2 - 1]);
                }
                max_cherries = std::cmp::max(max_cherries, dp[k - 1][r1][r2]);

                if max_cherries != i32::MIN {
                    max_cherries += if r1 == r2 {
                        grid[r1][c1]
                    } else {
                        grid[r1][c1] + grid[r2][c2]
                    };
                    dp[k][r1][r2] = max_cherries;
                }
            }
        }
    }

    std::cmp::max(0, dp[2 * n - 2][n - 1][n - 1])
}

/// p743
pub fn network_delay_time(times: Vec<Vec<i32>>, n: i32, k: i32) -> i32 {
    const INF: i32 = i32::MAX / 2;
    let n = n as usize;
    let mut g = vec![vec![INF; n]; n];
    for t in times {
        g[t[0] as usize - 1][t[1] as usize - 1] = t[2];
    }

    let mut dis = vec![INF; n];
    dis[k as usize - 1] = 0;
    let mut done = vec![false; n];
    loop {
        let mut x = n;
        for (i, &ok) in done.iter().enumerate() {
            if !ok && (x == n || dis[i] < dis[x]) {
                x = i;
            }
        }
        if x == n {
            return *dis.iter().max().unwrap();
        }
        if dis[x] == INF {
            return -1;
        }
        done[x] = true;
        for (y, &d) in g[x].iter().enumerate() {
            dis[y] = dis[y].min(dis[x] + d);
        }
    }
}

/// p744
pub fn next_greatest_letter(letters: Vec<char>, target: char) -> char {
    letters[letters
        .binary_search(&((u32::from(target) + 1) as u8).into())
        .unwrap_or_else(std::convert::identity)
        % letters.len()]
}

/// p745
#[derive(Default)]
struct Trie {
    index: i32,
    next: [Option<Box<Trie>>; 27],
}

struct WordFilter {
    trie: Trie,
}

impl WordFilter {
    fn new(words: Vec<String>) -> Self {
        let mut trie = Trie::default();
        for (i, word) in words.iter().enumerate() {
            let s = String::new() + &word + "{" + &word;
            for j in 0..word.len() {
                let mut curr = &mut trie;
                for &b in &s.as_bytes()[j..] {
                    curr = curr.next[(b - b'a') as usize].get_or_insert_with(Default::default);
                    curr.index = i as i32;
                }
            }
        }
        Self { trie }
    }

    fn f(&self, prefix: String, suffix: String) -> i32 {
        let mut curr = &self.trie;
        let s = String::new() + &suffix + "{" + &prefix;
        for &b in s.as_bytes() {
            if let Some(n) = &curr.next[(b - b'a') as usize] {
                curr = n.as_ref();
            } else {
                return -1;
            }
        }
        curr.index
    }
}

/// p746
pub fn min_cost_climbing_stairs(cost: Vec<i32>) -> i32 {
    let length = cost.len();
    let mut dp = vec![0; length];
    for i in 2..length {
        dp[i] = (dp[i - 2] + cost[i - 2]).min(dp[i - 1] + cost[i - 1]);
    }
    (dp[length - 1] + cost[length - 1]).min(dp[length - 2] + cost[length - 2])
}

/// p747
pub fn dominant_index(nums: Vec<i32>) -> i32 {
    let (mut max1, mut max2, mut idx) = (0, 0, 0);
    for (i, &n) in nums.iter().enumerate() {
        if n > max1 {
            idx = i;
            max2 = max1;
            max1 = n;
        } else if n > max2 {
            max2 = n;
        }
    }
    return if max1 >= max2 * 2 { idx as i32 } else { -1 };
}

/// p748
pub fn shortest_completing_word(license_plate: String, words: Vec<String>) -> String {
    let cnt = license_plate
        .to_lowercase()
        .chars()
        .fold(vec![0; 26], |mut acc, c| {
            if c.is_ascii_alphabetic() {
                acc[c as u8 as usize - 'a' as usize] += 1;
            }

            acc
        });

    let mut ans = "".to_string();

    for word in words {
        let cnt1 = word.to_lowercase().chars().fold(vec![0; 26], |mut acc, c| {
            if c.is_ascii_alphabetic() {
                acc[c as u8 as usize - 'a' as usize] += 1;
            }

            acc
        });

        let b = (0..26).all(|i| cnt[i] <= cnt1[i]);

        if b && (ans.is_empty() || ans.len() > word.len()) {
            ans = word;
        }
    }

    ans
}

/// p749
const DIRS: &'static [isize] = &[0, -1, 0, 1, 0];
const INFECTED: i32 = 1;
const UNINFECTED: i32 = 0;
const BLOCKED: i32 = -1;

pub fn contain_virus(mut grid: Vec<Vec<i32>>) -> i32 {
    let _len_rs: usize = grid.len();
    let len_cs: usize = grid[0].len();
    let mut cnt: i32 = 0;
    let mut phase: i32 = 1;
    let mut heap: BinaryHeap<Region> = {
        let mut heap: BinaryHeap<Region> = BinaryHeap::new();
        add(phase, &mut grid, &mut heap);
        heap
    };
    while let Some(Region {
        infected,
        uninfected_neighbors: _,
        walls_needed,
    }) = heap.pop()
    {
        cnt += walls_needed;
        for hash in infected {
            grid[hash / len_cs][hash % len_cs] = BLOCKED;
        }
        phase += 1;
        while let Some(Region {
            infected: _,
            uninfected_neighbors,
            walls_needed: _,
        }) = heap.pop()
        {
            for hash in uninfected_neighbors {
                grid[hash / len_cs][hash % len_cs] = phase;
            }
        }
        add(phase, &mut grid, &mut heap);
    }
    cnt
}
fn add(phase: i32, grid: &mut Vec<Vec<i32>>, heap: &mut BinaryHeap<Region>) {
    let len_rs: usize = grid.len();
    let len_cs: usize = grid[0].len();
    for r in 0..len_rs {
        for c in 0..len_cs {
            if grid[r][c] == phase {
                let mut region = Region::new();
                dfs((r as isize, c as isize), phase, grid, &mut region);
                if !region.uninfected_neighbors.is_empty() {
                    heap.push(region);
                }
            }
        }
    }
}
fn dfs(coord: (isize, isize), phase: i32, grid: &mut Vec<Vec<i32>>, region: &mut Region) {
    let len_rs: isize = grid.len() as isize;
    let len_cs: isize = grid[0].len() as isize;
    let (r, c) = coord;
    if r < 0
        || c < 0
        || r >= len_rs
        || c >= len_cs
        || grid[r as usize][c as usize] == -1
        || grid[r as usize][c as usize] > phase
    {
        return;
    }
    if grid[r as usize][c as usize] == UNINFECTED {
        region.uninfected_neighbors.insert(hash(r, c, len_cs));
        region.walls_needed += 1;
        return;
    }
    grid[r as usize][c as usize] += 1;
    region.infected.insert(hash(r, c, len_cs));
    for d in 0..4 {
        let r_nxt: isize = r + DIRS[d];
        let c_nxt: isize = c + DIRS[d + 1];
        dfs((r_nxt, c_nxt), phase, grid, region);
    }
}
fn hash(r: isize, c: isize, len_cs: isize) -> usize {
    (r * len_cs + c) as usize
}

struct Region {
    infected: HashSet<usize>,
    uninfected_neighbors: HashSet<usize>,
    walls_needed: i32,
}

impl Region {
    pub fn new() -> Self {
        Region {
            infected: HashSet::new(),
            uninfected_neighbors: HashSet::new(),
            walls_needed: 0,
        }
    }
}

impl Ord for Region {
    fn cmp(&self, other: &Self) -> Ordering {
        self.uninfected_neighbors
            .len()
            .cmp(&other.uninfected_neighbors.len())
    }
}

impl PartialOrd for Region {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        self.uninfected_neighbors.len() == other.uninfected_neighbors.len()
    }
}

impl Eq for Region {}

/// p752
pub fn open_lock(deadends: Vec<String>, target: String) -> i32 {
    if target == "0000" {
        return 0;
    }
    let target: u16 = target.parse().unwrap();
    let dead: HashSet<u16> = deadends.iter().map(|s| s.parse::<u16>().unwrap()).collect();
    if dead.contains(&0) || dead.contains(&target) {
        return -1;
    }

    fn get_next(num: u16) -> Vec<u16> {
        let mut ans = vec![];
        for i in 0..4 {
            let exp = 10_u16.pow(i);
            let n = num / exp % 10;
            ans.push(if n == 9 { num - 9 * exp } else { num + exp });
            ans.push(if n == 0 { num + 9 * exp } else { num - exp });
        }
        ans
    }

    let (mut top_q, mut bottom_q) = (VecDeque::new(), VecDeque::new());
    top_q.push_back(0);
    bottom_q.push_back(target);
    let mut step = 0;
    let (mut top_set, mut bottom_set) = (HashSet::new(), HashSet::new());
    top_set.insert(0);
    bottom_set.insert(target);
    while !top_q.is_empty() && !bottom_q.is_empty() {
        step += 1;
        let (q, this, other) = if top_q.len() > bottom_q.len() {
            (&mut bottom_q, &mut bottom_set, &mut top_set)
        } else {
            (&mut top_q, &mut top_set, &mut bottom_set)
        };
        for _ in 0..q.len() {
            let cur = q.pop_front().unwrap();
            for &next in get_next(cur).iter() {
                if other.contains(&next) {
                    return step;
                }
                if !dead.contains(&next) && !this.contains(&next) {
                    this.insert(next);
                    q.push_back(next);
                }
            }
        }
    }
    -1
}

/// p753
pub fn crack_safe(n: i32, k: i32) -> String {
    let mut selected = vec![0; n as usize];
    let mut vis = HashSet::new();
    vis.insert(selected.clone());

    fn dfs(
        n: usize,
        k: usize,
        total: usize,
        selected: &mut Vec<usize>,
        vis: &mut HashSet<Vec<usize>>,
        cur: usize,
    ) -> bool {
        if vis.len() == total {
            return true;
        }

        for i in 0..k {
            let mut sub = selected[cur..].to_vec();
            sub.push(i);

            if vis.insert(sub.clone()) {
                selected.push(i);

                if dfs(n, k, total, selected, vis, cur + 1) {
                    return true;
                }

                selected.pop();
                vis.remove(&sub);
            }
        }

        false
    }

    dfs(
        n as usize,
        k as usize,
        k.pow(n as u32) as usize,
        &mut selected,
        &mut vis,
        1,
    );

    selected
        .into_iter()
        .map(|s| (s as u8 + b'0') as char)
        .collect()
}

/// p754
pub fn reach_number(target: i32) -> i32 {
    let mut target = target.abs();
    let mut k = 0;
    while target > 0 {
        k = k + 1;
        target = target - k;
    }
    return if target % 2 == 0 { k } else { k + 1 + k % 2 };
}

/// p756
pub fn pyramid_transition(bottom: String, allowed: Vec<String>) -> bool {
    let n = bottom.len();
    let mut matrix = vec![vec![0_u8; n]; n];
    for (i, b) in bottom.bytes().enumerate() {
        matrix[0][i] = b - b'A';
    }

    let mut allowed_map: HashMap<u8, Vec<u8>> = HashMap::new();
    for bytes in allowed.iter().map(|s| s.as_bytes()) {
        let key = (bytes[0] - b'A') * 10 + (bytes[1] - b'A');
        let val = bytes[2] - b'A';
        allowed_map.entry(key).or_insert(vec![]).push(val);
    }
    fn backtracing(
        matrix: &mut Vec<Vec<u8>>,
        row: usize,
        col: usize,
        allowed: &HashMap<u8, Vec<u8>>,
    ) -> bool {
        let n = matrix.len();
        if row >= n {
            return true;
        }
        if row + col == n {
            return backtracing(matrix, row + 1, 0, allowed);
        }

        let key = matrix[row - 1][col] * 10 + matrix[row - 1][col + 1];
        if let Some(next) = allowed.get(&key) {
            for &b in next.iter() {
                matrix[row][col] = b;
                if backtracing(matrix, row, col + 1, allowed) {
                    return true;
                }
            }
        }

        false
    }

    backtracing(&mut matrix, 1, 0, &allowed_map)
}

/// p757
pub fn intersection_size_two(mut intervals: Vec<Vec<i32>>) -> i32 {
    intervals.sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap());
    intervals
        .iter()
        .fold((0, -1, -1), |(cnt, second_end, first_end), interval| {
            if interval[0] > first_end {
                (cnt + 2, interval[1] - 1, interval[1])
            } else if interval[0] > second_end {
                (
                    cnt + 1,
                    if first_end == interval[1] {
                        first_end - 1
                    } else {
                        first_end
                    },
                    interval[1],
                )
            } else {
                (cnt, second_end, first_end)
            }
        })
        .0
}

/// p761
pub fn make_largest_special(s: String) -> String {
    let mut cnt = 0;
    let mut arr = s
        .split_inclusive(|ch| {
            cnt += if ch == '1' { 1 } else { -1 };
            cnt == 0
        })
        .map(|curr| {
            format!(
                "1{}0",
                make_largest_special(curr[1..curr.len() - 1].to_string())
            )
        })
        .collect::<Vec<_>>();
    arr.sort_by(|a, b| b.cmp(a));
    arr.concat()
}

/// p762
pub fn count_prime_set_bits(left: i32, right: i32) -> i32 {
    (left..=right).fold(0, |mut ret, i| {
        let (mut cnt, mut j) = (0, i);
        while j > 0 {
            cnt += &j & 1;
            j >>= 1;
        }
        if [2, 3, 5, 7, 11, 13, 17, 19].contains(&cnt) {
            ret += 1;
        }
        ret
    })
}

/// p763
pub fn partition_labels(s: String) -> Vec<i32> {
    let mut pos = [0; 26];
    s.chars()
        .enumerate()
        .for_each(|(i, c)| pos[c as usize - 'a' as usize] = i);
    let (mut start, mut end) = (0, 0);
    let mut ans = vec![];
    s.chars().enumerate().for_each(|(i, c)| {
        let idx = c as usize - 'a' as usize;
        end = end.max(pos[idx]);
        if i == end {
            ans.push((i - start) as i32 + 1);
            start = i + 1;
        }
    });
    ans
}

/// p764
pub fn order_of_largest_plus_sign(n: i32, mines: Vec<Vec<i32>>) -> i32 {
    let (mut grid, ret) = (vec![vec![n; n as usize]; n as usize], 0);
    for mine in mines {
        grid[mine[0] as usize][mine[1] as usize] = 0;
    }
    for i in 0..n as usize {
        let (mut j, mut k, mut u, mut d, mut l, mut r) = (0, n as usize - 1, 0, 0, 0, 0);
        while j < n as usize {
            r = if grid[i][j] == 0 { 0 } else { r + 1 };
            d = if grid[j][i] == 0 { 0 } else { d + 1 };
            l = if grid[i][k] == 0 { 0 } else { l + 1 };
            u = if grid[k][i] == 0 { 0 } else { u + 1 };
            grid[i][j] = grid[i][j].min(r);
            grid[j][i] = grid[j][i].min(d);
            grid[i][k] = grid[i][k].min(l);
            grid[k][i] = grid[k][i].min(u);
            k -= 1;
            j += 1;
        }
    }
    ret.max(*grid.iter().map(|v| v.iter().max().unwrap()).max().unwrap())
}

/// p765
pub fn min_swaps_couples(row: Vec<i32>) -> i32 {
    let mut parent = (0..60).collect::<Vec<usize>>();
    for i in 0..30 {
        parent[i * 2 + 1] = parent[i * 2];
    }
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
    let mut ans = row.len() / 2;
    for i in 0..row.len() / 2 {
        if find(&mut parent, row[2 * i] as usize) != find(&mut parent, row[2 * i + 1] as usize) {
            union(&mut parent, row[2 * i] as usize, row[2 * i + 1] as usize);
            ans -= 1;
        }
    }
    (row.len() / 2 - ans) as i32
}

/// p766
pub fn is_toeplitz_matrix(matrix: Vec<Vec<i32>>) -> bool {
    let (width, height) = (matrix.len(), matrix[0].len());
    for r in 0..width {
        for c in 0..height {
            if r - 1 < width && c - 1 < height && matrix[r][c] != matrix[r - 1][c - 1] {
                return false;
            }
        }
    }
    true
}

/// p767
pub fn reorganize_string(s: String) -> String {
    let n = s.len();
    let mut count = HashMap::new();
    for c in s.bytes() {
        *count.entry(c).or_insert(0) += 1;
    }

    let mut a = count.into_iter().collect::<Vec<_>>();
    a.sort_unstable_by(|p, q| q.1.cmp(&p.1));
    let m = a[0].1;
    if m > n - m + 1 {
        return "".to_string();
    }

    let mut ans = vec![b'\0'; n];
    let mut i = 0;
    for (ch, cnt) in a {
        for _ in 0..cnt {
            ans[i] = ch;
            i += 2;
            if i >= n {
                i = 1;
            }
        }
    }
    unsafe { String::from_utf8_unchecked(ans) }
}

/// p768
pub fn max_chunks_to_sorted(arr: Vec<i32>) -> i32 {
    let mut arr_copy = arr.clone();
    arr_copy.sort();
    arr.iter()
        .zip(arr_copy.iter())
        .fold((0, 0, 0), |(ret, sum1, sum2), (x, y)| {
            (ret + if sum1 == sum2 { 1 } else { 0 }, sum1 + x, sum2 + y)
        })
        .0
}

/// p769
pub fn max_chunks_to_sorted_2(arr: Vec<i32>) -> i32 {
    arr.iter()
        .enumerate()
        .fold((0, i32::MIN), |(cnt, mut maximum), (i, &num)| {
            maximum = maximum.max(num as i32);
            if maximum == i as i32 {
                (cnt + 1, maximum)
            } else {
                (cnt, maximum)
            }
        })
        .0
}

/// p771
pub fn num_jewels_in_stones(jewels: String, stones: String) -> i32 {
    let jewel = jewels.chars().collect::<HashSet<char>>();
    stones
        .chars()
        .fold(0, |acc, c| if jewel.contains(&c) { acc + 1 } else { acc })
}

/// p773
pub fn sliding_puzzle(board: Vec<Vec<i32>>) -> i32 {
    const TARGET: i32 = 123450;

    fn vec_to_i32(cur_board: &Vec<Vec<i32>>) -> i32 {
        let mut val = 0;
        for i in 0..2 {
            for j in 0..3 {
                val = val * 10 + cur_board[i][j];
            }
        }
        val
    }

    fn get_next_state(
        mut cur_state: i32,
        q: &mut VecDeque<i32>,
        records: &mut HashSet<i32>,
    ) -> bool {
        let mut cur_board = vec![vec![0; 3]; 2];
        let (mut zero_x, mut zero_y) = (0, 0);

        for i in (0..2).rev() {
            for j in (0..3).rev() {
                cur_board[i][j] = cur_state % 10;
                if cur_board[i][j] == 0 {
                    zero_x = i;
                    zero_y = j;
                }
                cur_state /= 10;
            }
        }

        if zero_x > 0 {
            cur_board[zero_x][zero_y] = cur_board[zero_x - 1][zero_y];
            cur_board[zero_x - 1][zero_y] = 0;

            let next_state = vec_to_i32(&cur_board);
            if next_state == TARGET {
                return true;
            }
            if !records.contains(&next_state) {
                records.insert(next_state);
                q.push_back(next_state);
            }

            cur_board[zero_x - 1][zero_y] = cur_board[zero_x][zero_y];
            cur_board[zero_x][zero_y] = 0;
        }
        if zero_x + 1 < 2 {
            cur_board[zero_x][zero_y] = cur_board[zero_x + 1][zero_y];
            cur_board[zero_x + 1][zero_y] = 0;

            let next_state = vec_to_i32(&cur_board);
            if next_state == TARGET {
                return true;
            }
            if !records.contains(&next_state) {
                records.insert(next_state);
                q.push_back(next_state);
            }

            cur_board[zero_x + 1][zero_y] = cur_board[zero_x][zero_y];
            cur_board[zero_x][zero_y] = 0;
        }
        if zero_y > 0 {
            cur_board[zero_x][zero_y] = cur_board[zero_x][zero_y - 1];
            cur_board[zero_x][zero_y - 1] = 0;

            let next_state = vec_to_i32(&cur_board);
            if next_state == TARGET {
                return true;
            }
            if !records.contains(&next_state) {
                records.insert(next_state);
                q.push_back(next_state);
            }

            cur_board[zero_x][zero_y - 1] = cur_board[zero_x][zero_y];
            cur_board[zero_x][zero_y] = 0;
        }
        if zero_y + 1 < 3 {
            cur_board[zero_x][zero_y] = cur_board[zero_x][zero_y + 1];
            cur_board[zero_x][zero_y + 1] = 0;

            let next_state = vec_to_i32(&cur_board);
            if next_state == TARGET {
                return true;
            }
            if !records.contains(&next_state) {
                records.insert(next_state);
                q.push_back(next_state);
            }

            cur_board[zero_x][zero_y + 1] = cur_board[zero_x][zero_y];
            cur_board[zero_x][zero_y] = 0;
        }

        false
    }

    let init = vec_to_i32(&board);
    if init == TARGET {
        return 0;
    }

    let mut q = VecDeque::new();
    let mut records = HashSet::new();
    q.push_back(init);
    records.insert(init);

    let mut count = 1;
    while !q.is_empty() {
        let size = q.len();
        for _ in 0..size {
            let cur_state = q.pop_front().unwrap();
            if get_next_state(cur_state, &mut q, &mut records) {
                return count;
            }
        }
        count += 1;
    }

    -1
}

/// p775
pub fn is_ideal_permutation(nums: Vec<i32>) -> bool {
    for i in 0..nums.len() {
        if (nums[i] - i as i32).abs() > 1 {
            return false;
        }
    }
    true
}

/// p777
pub fn can_transform(start: String, end: String) -> bool {
    let (mut i, mut j, m, n, start_arr, end_arr) = (
        0,
        0,
        start.len(),
        end.len(),
        start.as_bytes(),
        end.as_bytes(),
    );
    while i < m || j < n {
        while i < m && start_arr[i] == b'X' {
            i += 1;
        }
        while j < n && end_arr[j] == b'X' {
            j += 1;
        }
        if i >= m || j >= n {
            break;
        }
        if start_arr[i] != end_arr[j]
            || start_arr[i] == b'L' && i < j
            || start_arr[i] == b'R' && i > j
        {
            return false;
        }
        i += 1;
        j += 1;
    }
    i == j
}

/// p778
pub fn swim_in_water(grid: Vec<Vec<i32>>) -> i32 {
    let n = grid.len();
    let m = n * n;
    let mut p: Vec<usize> = (0..m).collect();
    let mut hi = vec![0usize; m];

    for i in 0..n {
        for j in 0..n {
            hi[grid[i][j] as usize] = i * n + j;
        }
    }

    fn find(x: usize, p: &mut Vec<usize>) -> usize {
        if p[x] != x {
            p[x] = find(p[x], p);
        }
        p[x]
    }

    let dirs = [-1isize, 0, 1, 0, -1];

    for t in 0..m {
        let id = hi[t];
        let x = id / n;
        let y = id % n;

        for k in 0..4 {
            let nx = x as isize + dirs[k];
            let ny = y as isize + dirs[k + 1];
            if nx >= 0 && nx < n as isize && ny >= 0 && ny < n as isize {
                let nx = nx as usize;
                let ny = ny as usize;
                if grid[nx][ny] as usize <= t {
                    let a = find(x * n + y, &mut p);
                    let b = find(nx * n + ny, &mut p);
                    p[a] = b;
                }
            }
        }

        if find(0, &mut p) == find(m - 1, &mut p) {
            return t as i32;
        }
    }

    0
}

/// p779
pub fn kth_grammar(_: i32, k: i32) -> i32 {
    (k - 1).count_ones() as i32 & 1
}

/// p780
pub fn reaching_points(sx: i32, sy: i32, mut tx: i32, mut ty: i32) -> bool {
    while tx > sx && ty > sy {
        if tx > ty { tx -= ty } else { ty -= tx }
    }
    tx == sx && ty >= sy && (ty - sy) % tx == 0 || tx >= sx && ty == sy && (tx - sx) % ty == 0
}

/// p781
pub fn num_rabbits(answers: Vec<i32>) -> i32 {
    let mut count: HashMap<i32, i32> = HashMap::new();
    answers.iter().for_each(|ans| {
        let count = count.entry(*ans).or_insert(0);
        *count = *count + 1;
    });
    count
        .iter()
        .fold(0, |total, (k, v)| total + ((v + k) / (k + 1)) * (k + 1))
}

/// p782
pub fn moves_to_chessboard(board: Vec<Vec<i32>>) -> i32 {
    let n = board.len();

    let mut row_mask = 0;
    let mut col_mask = 0;
    for i in 0..n {
        row_mask |= board[0][i] << i;
        col_mask |= board[i][0] << i;
    }

    let reverse_row_mask = ((1 << n) - 1) ^ row_mask;
    let reverse_col_mask = ((1 << n) - 1) ^ col_mask;
    let mut row_cnt = 0;
    let mut col_cnt = 0;

    for i in 0..n {
        let mut curr_row_mask = 0;
        let mut curr_col_mask = 0;
        for j in 0..n {
            curr_row_mask |= board[i][j] << j;
            curr_col_mask |= board[j][i] << j;
        }

        if (curr_row_mask != row_mask && curr_row_mask != reverse_row_mask)
            || (curr_col_mask != col_mask && curr_col_mask != reverse_col_mask)
        {
            return -1;
        }

        if curr_row_mask == row_mask {
            row_cnt += 1;
        }
        if curr_col_mask == col_mask {
            col_cnt += 1;
        }
    }

    fn get_moves(mask: usize, count: i32, n: usize) -> i32 {
        let ones = mask.count_ones() as i32;
        if n % 2 == 1 {
            if (n as i32 - 2 * ones).abs() != 1 || (n as i32 - 2 * count).abs() != 1 {
                return -1;
            }
            if ones == (n / 2) as i32 {
                return (n as i32 / 2) - (mask & 0xAAAAAAAA).count_ones() as i32;
            } else {
                return ((n as i32 + 1) / 2) - (mask & 0x55555555).count_ones() as i32;
            }
        } else {
            if ones != (n / 2) as i32 || count != (n / 2) as i32 {
                return -1;
            }
            let count0 = (n as i32 / 2) - (mask & 0xAAAAAAAA).count_ones() as i32;
            let count1 = (n as i32 / 2) - (mask & 0x55555555).count_ones() as i32;
            return std::cmp::min(count0, count1);
        }
    }

    let row_moves = get_moves(row_mask as usize, row_cnt, n);
    let col_moves = get_moves(col_mask as usize, col_cnt, n);
    if row_moves == -1 || col_moves == -1 {
        return -1;
    } else {
        return row_moves + col_moves;
    }
}

/// p783
pub fn min_diff_in_bst(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut vals: Vec<i32> = vec![];
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, vals: &mut Vec<i32>) {
        if let Some(node) = root.as_ref() {
            dfs(node.borrow().left.clone(), vals);
            vals.push(node.borrow().val);
            dfs(node.borrow().right.clone(), vals);
        }
    }
    dfs(root, &mut vals);
    vals.windows(2)
        .fold(i32::MAX, |min, vals| (vals[1] - vals[0]).min(min))
}

/// p784
pub fn letter_case_permutation(s: String) -> Vec<String> {
    let chars = s.chars().collect::<Vec<char>>();
    let mut ans: Vec<String> = vec!["".to_string()];
    chars.iter().for_each(|c| {
        let mut new_ans: Vec<String> = vec![];
        ans.iter_mut().for_each(|s| {
            if c.is_ascii_alphabetic() {
                s.push(c.to_ascii_lowercase());
                new_ans.push(s.clone());
                s.pop();
                s.push(c.to_ascii_uppercase());
                new_ans.push(s.clone());
            } else {
                s.push(*c);
                new_ans.push(s.clone());
            }
        });
        ans = new_ans;
    });
    ans
}

/// p785
pub fn is_bipartite(graph: Vec<Vec<i32>>) -> bool {
    fn dfs(x: usize, c: i8, graph: &[Vec<i32>], colors: &mut [i8]) -> bool {
        colors[x] = c;
        for &y in &graph[x] {
            let y = y as usize;
            if colors[y] == c || colors[y] == 0 && !dfs(y, -c, graph, colors) {
                return false;
            }
        }
        true
    }

    let n = graph.len();
    let mut colors = vec![0; n];

    for i in 0..n {
        if colors[i] == 0 && !dfs(i, 1, &graph, &mut colors) {
            return false;
        }
    }
    true
}

/// p786
pub fn kth_smallest_prime_fraction(arr: Vec<i32>, k: i32) -> Vec<i32> {
    let length = arr.len();
    let mut fractions: Vec<Vec<i32>> = vec![];
    for i in 0..length - 1 {
        for j in i + 1..length {
            fractions.push(vec![arr[i], arr[j]])
        }
    }
    fractions.sort_by(|first, second| (first[0] * second[1]).cmp(&(first[1] * second[0])));
    fractions[(k - 1) as usize].clone()
}

/// p787
pub fn find_cheapest_price(n: i32, flights: Vec<Vec<i32>>, src: i32, dst: i32, k: i32) -> i32 {
    let mut graph = vec![vec![]; n as usize];
    for flight in flights {
        let from = flight[0] as usize;
        let to = flight[1] as usize;
        let price = flight[2];
        graph[from].push((to, price));
    }
    let mut dist = vec![vec![i32::MAX; k as usize + 2]; n as usize];
    for i in 0..k as usize + 2 {
        dist[src as usize][i] = 0;
    }
    let mut binary_heap = std::collections::BinaryHeap::new();
    binary_heap.push((0, src as usize, 0));
    while let Some((price, v, count)) = binary_heap.pop() {
        if v == dst as usize {
            return -price;
        }
        if count > k {
            continue;
        }

        let price = -price;
        for (nv, nprice) in &graph[v] {
            let tmp = nprice + price;
            let count = count + 1;
            if tmp < dist[*nv][count as usize] {
                dist[*nv][count as usize] = tmp;
                binary_heap.push((-tmp, *nv, count));
            }
        }
    }
    -1
}

/// p788
pub fn rotated_digits(n: i32) -> i32 {
    let (mut cnt, mut dp) = (0, vec![0; n as usize + 1]);
    dp[0] = 1;
    for i in 1..=n {
        let (a, b) = (i / 10, i % 10);
        if [3, 4, 7].contains(&b) {
            continue;
        }
        if [0, 1, 8].contains(&b) {
            dp[i as usize] = dp[a as usize];
        } else {
            dp[i as usize] = if dp[a as usize] == 0 { 0 } else { 2 };
        }
        cnt += if dp[i as usize] == 2 { 1 } else { 0 };
    }
    cnt
}

/// p789
pub fn escape_ghosts(ghosts: Vec<Vec<i32>>, target: Vec<i32>) -> bool {
    let target_distance = target[0].abs() + target[1].abs();
    for ghost in ghosts {
        if (target[0] - ghost[0]).abs() + (target[1] - ghost[1]).abs() <= target_distance {
            return false;
        }
    }
    true
}

/// p790
pub fn num_tilings(n: i32) -> i32 {
    (1..n)
        .fold((0, 1, 1, 1e9 as i32 + 7), |(a, b, c, m), _| {
            (b, c, (2 * c % m + a) % m, m)
        })
        .2
}

/// p791
pub fn custom_sort_string(order: String, s: String) -> String {
    let mut weights: HashMap<char, i32> = HashMap::new();
    order.chars().enumerate().for_each(|(i, c)| {
        weights.insert(c, i as i32);
    });
    let mut s_chars = s.chars().collect::<Vec<char>>();
    s_chars.sort_by(|l, r| {
        weights
            .get(l)
            .unwrap_or(&-1)
            .cmp(weights.get(r).unwrap_or(&-1))
    });
    s_chars.into_iter().collect()
}
