use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
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
