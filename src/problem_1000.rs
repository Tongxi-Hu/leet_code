use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
};

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
    let (max, min) = (nums.iter().max().unwrap(), nums.iter().min().unwrap());
    return if max - min > 2 * k {
        max - min - 2 * k
    } else {
        0
    };
}
