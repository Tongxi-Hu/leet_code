use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
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
