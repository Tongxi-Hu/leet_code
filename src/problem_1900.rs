use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque},
    i32,
};

/// 01
pub fn get_number_of_backlog_orders(orders: Vec<Vec<i32>>) -> i32 {
    #[derive(Eq, Debug, Clone, Default)]
    struct Order {
        price: i32,
        amount: i32,
        order_type: i32,
    }

    impl Order {
        fn new(price: i32, amount: i32, order_type: i32) -> Self {
            Order {
                price,
                amount,
                order_type,
            }
        }
    }

    impl PartialEq for Order {
        fn eq(&self, other: &Self) -> bool {
            self.price == other.price
        }
    }

    impl PartialOrd for Order {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Order {
        fn cmp(&self, other: &Self) -> Ordering {
            self.price.cmp(&other.price)
        }
    }

    const MOD: i32 = 1000000007;
    let (buy, sell) = orders
        .iter()
        .fold((BinaryHeap::new(), BinaryHeap::new()), |mut acc, cur| {
            if cur[2] == 0 {
                acc.0.push(Order::new(cur[0], cur[1], cur[2]));
            } else {
                acc.1.push(Reverse(Order::new(cur[0], cur[1], cur[2])));
            }
            while !acc.0.is_empty()
                && !acc.1.is_empty()
                && acc.0.peek().unwrap().price >= acc.1.peek().unwrap().0.price
            {
                let amount = acc
                    .0
                    .peek()
                    .unwrap()
                    .amount
                    .min(acc.1.peek().unwrap().0.amount);
                acc.0.peek_mut().unwrap().amount -= amount;
                acc.1.peek_mut().unwrap().0.amount -= amount;
                if acc.0.peek().unwrap().amount == 0 {
                    acc.0.pop();
                }
                if acc.1.peek().unwrap().0.amount == 0 {
                    acc.1.pop();
                }
            }
            (acc.0, acc.1)
        });
    let cnt = buy.iter().fold(0, |cnt, o| (cnt + o.amount) % MOD);
    sell.iter().fold(cnt, |cnt, o| (cnt + o.0.amount) % MOD)
}

/// 02
pub fn max_value(n: i32, index: i32, max_sum: i32) -> i32 {
    let mut lo = 0;
    let mut hi = max_sum;
    let rest = max_sum as i64 - n as i64;

    while lo <= hi {
        let mid = lo + (hi - lo) / 2;

        let left_cnt = (mid + 1.max(mid - index)) as i64 * (mid.min(index + 1)) as i64 / 2; // [0..=index]的总高度
        let right_cnt =
            (mid + 1.max(mid - (n - 1 - index))) as i64 * (mid.min(n - index)) as i64 / 2; // [index..n]的总高度

        if left_cnt + right_cnt - mid as i64 > rest {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    lo
}

/// 03
pub fn count_pairs(nums: Vec<i32>, low: i32, high: i32) -> i32 {
    fn f(nums: &Vec<i32>, x: i32) -> i32 {
        let mut root = Trie::new();
        let mut ans = 0;

        for i in 1..nums.len() {
            root.insert(nums[i - 1]);
            ans += root.search(nums[i], x);
        }

        ans
    }

    f(&nums, high) - f(&nums, low - 1)
}

#[derive(Default)]
struct Trie {
    children: [Option<Box<Trie>>; 2],
    sum: i32,
}

impl Trie {
    fn new() -> Self {
        Default::default()
    }

    fn insert(&mut self, num: i32) {
        let mut p = self;

        for k in (0..=14).rev() {
            let b = ((num >> k) & 1) as usize;
            p = p.children[b].get_or_insert_with(|| Box::new(Trie::new()));
            p.sum += 1
        }
    }

    fn search(&self, num: i32, x: i32) -> i32 {
        let mut p = self;
        let mut sum = 0;

        for k in (0..=14).rev() {
            let r = ((num >> k) & 1) as usize;

            if ((x >> k) & 1) != 0 {
                if let Some(ref child) = p.children[r] {
                    sum += child.sum;
                }
                if let Some(ref child) = p.children[r ^ 1] {
                    p = child;
                } else {
                    return sum;
                }
            } else {
                if let Some(ref child) = p.children[r] {
                    p = child;
                } else {
                    return sum;
                }
            }
        }

        sum += p.sum;

        sum
    }
}

/// 05
pub fn num_different_integers(word: String) -> i32 {
    word.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_start_matches('0'))
        .collect::<HashSet<_>>()
        .len() as i32
}

/// 06
pub fn reinitialize_permutation(n: i32) -> i32 {
    let (mut cnt, mut idx) = (1, n / 2);
    while idx != 1 {
        idx = if idx & 1 == 1 {
            n / 2 + (idx - 1) / 2
        } else {
            idx / 2
        };
        cnt += 1;
    }
    cnt
}

/// 07
pub fn evaluate(s: String, knowledge: Vec<Vec<String>>) -> String {
    let map: HashMap<&str, &str> = knowledge
        .iter()
        .map(|v| (v[0].as_str(), v[1].as_str()))
        .collect();
    let mut s = s;
    s.push('(');
    let (mut l, mut r) = (0, 0);
    let mut ans = String::with_capacity(s.len());
    for c in s.bytes() {
        match c {
            b'(' => {
                ans.push_str(&s[l..r]);
                r += 1;
                l = r;
            }
            b')' => {
                if let Some(x) = map.get(&s[l..r]) {
                    ans.push_str(x);
                } else {
                    ans.push_str("?");
                }
                r += 1;
                l = r;
            }
            _ => r += 1,
        }
    }
    ans
}

/// 12
pub fn square_is_white(coordinates: String) -> bool {
    let chars = coordinates.chars().collect::<Vec<char>>();
    let num = chars[1].to_digit(10).unwrap();
    match (num % 2, (chars[0] as u8 - b'a') % 2) {
        (0, 0) => true,
        (1, 0) => false,
        (0, 1) => false,
        (1, 1) => true,
        _ => false,
    }
}

/// 13
pub fn are_sentences_similar(sentence1: String, sentence2: String) -> bool {
    let (words_1, words_2) = (
        sentence1.split(" ").collect::<Vec<&str>>(),
        sentence2.split(" ").collect::<Vec<&str>>(),
    );
    let (mut i, mut j) = (0, 0);
    while i < words_1.len() && i < words_2.len() && words_1[i] == words_2[i] {
        i += 1;
    }
    while j < words_1.len() - i
        && j < words_2.len() - i
        && words_1[words_1.len() - j - 1] == words_2[words_2.len() - j - 1]
    {
        j += 1;
    }
    i + j == words_1.len().min(words_2.len())
}

/// 14
pub fn count_nice_pairs(nums: Vec<i32>) -> i32 {
    fn reverse(mut n: i32) -> i32 {
        let mut cur = 0;
        while n > 0 {
            cur = cur * 10 + n % 10;
            n = n / 10;
        }
        cur
    }
    nums.iter()
        .fold((HashMap::new(), 0), |mut a, c| {
            let v = c - reverse(*c);
            let cnt = *a.0.entry(v).or_insert(0);
            a.0.insert(v, cnt + 1);
            a.1 = (a.1 + cnt) % 1000000007;
            a
        })
        .1
}

/// 15
pub fn max_happy_groups(batch_size: i32, groups: Vec<i32>) -> i32 {
    const KWIDTH: i64 = 5;
    const KWIDTH_MASK: i64 = (1 << KWIDTH) - 1;
    fn dfs(tab: &mut HashMap<i64, i32>, batch_size: i64, mask: i64) -> i32 {
        if let Some(x) = tab.get(&mask) {
            return *x;
        }
        let total = (1..batch_size).fold(0, |total, i| {
            let amount = mask.overflowing_shr(((i - 1) * KWIDTH) as u32).0 & KWIDTH_MASK;
            total + i * amount
        });
        let best = (1..batch_size).fold(0, |best, i| {
            let amount = mask.overflowing_shr(((i - 1) * KWIDTH) as u32).0 & KWIDTH_MASK;
            if amount > 0 {
                let mut result = dfs(
                    tab,
                    batch_size,
                    mask - 1i64.overflowing_shl(((i - 1) * KWIDTH) as u32).0,
                );
                if (total - i) % batch_size == 0 {
                    result += 1;
                }
                best.max(result)
            } else {
                best
            }
        });
        tab.entry(mask).or_insert(best);
        best
    }
    let mut tab = HashMap::with_capacity(1 << 16);
    let mut cnt = vec![0; batch_size as usize];
    groups
        .into_iter()
        .for_each(|i| cnt[(i % batch_size) as usize] += 1);
    let start = (1..batch_size)
        .rev()
        .fold(0, |start, i| (start << KWIDTH) | cnt[i as usize]);
    tab.entry(0).or_insert(0);
    dfs(&mut tab, batch_size as i64, start) + cnt[0] as i32
}

/// 16
pub fn truncate_sentence(s: String, k: i32) -> String {
    let (words, mut ans) = (s.split(" ").collect::<Vec<&str>>(), "".to_string());
    for i in 1..=k as usize {
        if i != 1 {
            ans = ans + &" ";
        }
        ans = ans + words[i - 1];
    }
    ans
}

/// 17
pub fn finding_users_active_minutes(logs: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
    let (mut ret, mut cnt) = (vec![0; k as usize], HashMap::new());
    for log in logs {
        cnt.entry(log[0]).or_insert(HashSet::new()).insert(log[1]);
    }
    for curr in cnt.values() {
        ret[curr.len() - 1] += 1;
    }
    ret
}

/// 18
pub fn min_absolute_sum_diff(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let n = nums1.len();
    let mut cloned = nums1.clone();
    cloned.sort();
    let mut sum = 0;
    let mut max = 0;
    for i in 0..n {
        let diff = (nums1[i] - nums2[i]).abs();
        sum += diff as i64;
        let j = cloned.binary_search(&nums2[i]).unwrap_or_else(|x| x);
        if j > 0 {
            max = max.max(diff - (nums2[i] - cloned[j - 1]));
        }
        if j < n {
            max = max.max(diff - (cloned[j] - nums2[i]));
        }
    }
    sum -= max as i64;
    (sum % 1000000007) as i32
}

/// 19
pub fn count_different_subsequence_gc_ds(nums: Vec<i32>) -> i32 {
    let max_num = *nums.iter().max().unwrap();
    let mut vec = vec![false; max_num as usize + 1];
    for num in nums {
        vec[num as usize] = true;
    }

    fn gcd(a: i32, b: i32) -> i32 {
        if b == 0 {
            return a;
        }
        gcd(b, a % b)
    }

    let mut ans = 0;
    for i in 1..=max_num {
        let mut v = 0;
        for j in (i..=max_num).step_by(i as usize) {
            if vec[j as usize] {
                v = gcd(v, j);
            }
        }
        if v == i {
            ans += 1;
        }
    }

    ans
}

/// 22
pub fn array_sign(nums: Vec<i32>) -> i32 {
    let mut res = 1;
    for n in nums {
        if n > 0 {
            continue;
        } else if n == 0 {
            return 0;
        } else {
            res = -res;
        }
    }
    res
}

/// 23
pub fn find_the_winner(n: i32, k: i32) -> i32 {
    let mut queue = VecDeque::from_iter(1..=n as usize);
    while queue.len() > 1 {
        for _ in 1..=k - 1 {
            let v = queue.pop_front().unwrap();
            queue.push_back(v);
        }
        queue.pop_front();
    }
    *queue.front().unwrap() as i32
}

/// 24
pub fn min_side_jumps(obstacles: Vec<i32>) -> i32 {
    let mut dp = [1, 0, 1];
    for ob in obstacles {
        if ob > 0 {
            dp[ob as usize - 1] = i32::MAX;
        }
        for i in 0..3_usize {
            if ob == i as i32 + 1 {
                continue;
            }
            dp[i] = dp[i].min(dp[(i + 1) % 3].min(dp[(i + 2) % 3]) + 1);
        }
    }
    *dp.iter().min().unwrap()
}

/// 25
struct OrderedSet<T> {
    tree: BTreeMap<T, i32>,
}

impl<T> OrderedSet<T>
where
    T: PartialOrd + Ord + Copy,
{
    fn new() -> Self {
        Self {
            tree: BTreeMap::new(),
        }
    }

    fn insert(&mut self, x: T) {
        *self.tree.entry(x).or_insert(0) += 1;
    }

    fn remove(&mut self, x: &T) -> bool {
        let need_remove = {
            if let Some(t) = self.tree.get_mut(x) {
                *t -= 1;
                *t == 0
            } else {
                return false;
            }
        };
        if need_remove {
            self.tree.remove(x);
        }
        true
    }

    fn peek_first(&mut self) -> &T {
        self.tree.iter().next().unwrap().0
    }

    fn peek_last(&mut self) -> &T {
        self.tree.iter().rev().next().unwrap().0
    }

    fn pop_first(&mut self) -> T {
        let t = *self.peek_first();
        self.remove(&t);
        t
    }

    fn pop_last(&mut self) -> T {
        let t = *self.peek_last();
        self.remove(&t);
        t
    }
}

struct MKAverage {
    m: i32,
    k: i32,
    fifo: VecDeque<i32>,
    s_min: OrderedSet<i32>,
    s_mid: OrderedSet<i32>,
    s_max: OrderedSet<i32>,
    sum: i64,
}

impl MKAverage {
    fn new(m: i32, k: i32) -> Self {
        Self {
            m,
            k,
            fifo: VecDeque::with_capacity(m as usize),
            s_min: OrderedSet::new(),
            s_mid: OrderedSet::new(),
            s_max: OrderedSet::new(),
            sum: 0,
        }
    }

    fn add_element(&mut self, num: i32) {
        if self.fifo.len() < self.m as usize {
            self.sum += num as i64;
            self.s_mid.insert(num);
            self.fifo.push_back(num);
            if self.fifo.len() == self.m as usize {
                for _ in 0..self.k {
                    let t = self.s_mid.pop_first();
                    self.sum -= t as i64;
                    self.s_min.insert(t);
                }
                for _ in 0..self.k {
                    let t = self.s_mid.pop_last();
                    self.sum -= t as i64;
                    self.s_max.insert(t);
                }
            }
            return;
        }
        let out = self.fifo.pop_front().unwrap();
        self.fifo.push_back(num);
        if num < *self.s_min.peek_last() {
            let t = self.s_min.pop_last();
            self.s_min.insert(num);
            self.s_mid.insert(t);
            self.sum += t as i64;
        } else if num > *self.s_max.peek_first() {
            let t = self.s_max.pop_first();
            self.s_max.insert(num);
            self.s_mid.insert(t);
            self.sum += t as i64;
        } else {
            self.s_mid.insert(num);
            self.sum += num as i64;
        }
        if self.s_mid.remove(&out) {
            self.sum -= out as i64;
        } else if self.s_min.remove(&out) {
            let t = self.s_mid.pop_first();
            self.sum -= t as i64;
            self.s_min.insert(t);
        } else {
            let t = self.s_mid.pop_last();
            self.sum -= t as i64;
            self.s_max.insert(t);
        }
    }

    fn calculate_mk_average(&self) -> i32 {
        if self.fifo.len() < self.m as usize {
            -1
        } else {
            (self.sum / (self.m - self.k * 2) as i64) as i32
        }
    }
}

/// 27
pub fn min_operations(mut nums: Vec<i32>) -> i32 {
    let mut step = 0;
    for i in 0..nums.len() - 1 {
        if nums[i + 1] > nums[i] {
            continue;
        } else {
            step += nums[i] + 1 - nums[i + 1];
            nums[i + 1] = nums[i] + 1;
        }
    }
    step
}

/// 28
pub fn count_points(points: Vec<Vec<i32>>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    fn dis_sq(a: &Vec<i32>, b: &Vec<i32>) -> i32 {
        (a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2)
    }
    queries
        .iter()
        .map(|q| {
            let mut ans = 0;
            for p in points.iter() {
                if dis_sq(p, &vec![q[0], q[1]]) <= q[2].pow(2) {
                    ans += 1;
                }
            }
            ans
        })
        .collect()
}

/// 29
pub fn get_maximum_xor(nums: Vec<i32>, maximum_bit: i32) -> Vec<i32> {
    let mask = (1 << maximum_bit) - 1;
    let mut acc = 0;
    let mut ans = Vec::with_capacity(nums.len());
    for val in nums {
        acc ^= val;
        ans.push(acc ^ mask);
    }
    ans.reverse();
    ans
}

/// 32
pub fn check_if_pangram(sentence: String) -> bool {
    sentence
        .chars()
        .fold(HashSet::new(), |mut acc, cur| {
            acc.insert(cur);
            acc
        })
        .len()
        == 26
}

/// 33
pub fn max_ice_cream(mut costs: Vec<i32>, mut coins: i32) -> i32 {
    let mut cnt = 0;
    costs.sort();
    for c in costs {
        if coins >= c {
            coins -= c;
            cnt += 1;
        } else {
            break;
        }
    }
    cnt
}

/// 34
pub fn get_order(tasks: Vec<Vec<i32>>) -> Vec<i32> {
    let (mut i, mut s, n) = (0, 0, tasks.len());
    let mut id: Vec<usize> = (0..n).collect();
    id.sort_by(|&a, &b| tasks[a][0].cmp(&tasks[b][0]));
    let mut h = BinaryHeap::new();
    let mut ans = vec![];
    loop {
        if h.is_empty() && i < n && s < tasks[id[i]][0] {
            s = tasks[id[i]][0];
        }
        while i < n && tasks[id[i]][0] <= s {
            let j = id[i];
            h.push((-tasks[j][1], -(j as i32)));
            i += 1;
        }
        if let Some((t, j)) = h.pop() {
            ans.push(-j);
            s += -t;
        } else {
            break ans;
        }
    }
}

/// 35
pub fn get_xor_sum(arr1: Vec<i32>, arr2: Vec<i32>) -> i32 {
    let xor1 = arr1.iter().fold(0, |a, &b| a ^ b);
    let xor2 = arr2.iter().fold(0, |a, &b| a ^ b);
    xor1 & xor2
}

/// 37
pub fn sum_base(mut n: i32, k: i32) -> i32 {
    let mut ans = 0;
    while n > 0 {
        ans += n % k;
        n = n / k;
    }
    ans
}
