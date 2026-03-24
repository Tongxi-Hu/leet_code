use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

/// 01
pub fn average_waiting_time(customers: Vec<Vec<i32>>) -> f64 {
    let (wait, _) = customers
        .iter()
        .fold((0.0, customers[0][0] as f64), |mut acc, cur| {
            let (arr, cost) = (cur[0] as f64, cur[1] as f64);
            let time = if acc.1 < arr { arr } else { acc.1 } + cost;
            let wait = time - arr;
            acc.0 += wait;
            (acc.0, time)
        });
    wait / customers.len() as f64
}

/// 02
pub fn maximum_binary_string(binary: String) -> String {
    let n = binary.len();
    let mut s: Vec<char> = binary.chars().collect();
    let mut j = 0;
    for i in 0..n {
        if s[i] == '0' {
            while j <= i || (j < n && s[j] == '1') {
                j += 1;
            }
            if j < n {
                s[j] = '1';
                s[i] = '1';
                s[i + 1] = '0';
            }
        }
    }
    s.iter().collect()
}

/// 03
pub fn min_moves(mut nums: Vec<i32>, k: i32) -> i32 {
    let mut start = 0;
    for curr in 0..nums.len() {
        if nums[curr] == 1 {
            nums[start] = curr as i32;
            start += 1
        }
    }
    nums.truncate(start as usize);
    let mut iter = nums.windows(k as usize);
    let tmp = iter.next().unwrap();
    let currsum: i32 = tmp.iter().map(|x| (x - tmp[(k / 2) as usize]).abs()).sum();
    let c = iter.fold(currsum, |s, tmp| {
        s.min(tmp.iter().map(|x| (x - tmp[(k / 2) as usize]).abs()).sum())
    });
    c - (k as u64 * k as u64 / 4) as i32
}

/// 04
pub fn halves_are_alike(s: String) -> bool {
    let chars = s.chars().collect::<Vec<char>>();
    let vows = HashSet::from(['A', 'a', 'E', 'e', 'I', 'i', 'O', 'o', 'U', 'u']);
    let (mut l, mut r, mut cnt) = (0, chars.len() - 1, 0);
    while l < r {
        if vows.contains(&chars[l]) {
            cnt += 1;
        }
        if vows.contains(&chars[r]) {
            cnt -= 1;
        }
        l += 1;
        r -= 1;
    }
    cnt == 0
}

/// 05
pub fn eaten_apples(apples: Vec<i32>, days: Vec<i32>) -> i32 {
    let mut ans = 0;
    let mut pq = BinaryHeap::new();
    let n = apples.len();
    let mut i = 0;

    while i < n {
        while let Some(Reverse((rotten_day, _))) = pq.peek() {
            if *rotten_day <= i as i32 {
                pq.pop();
            } else {
                break;
            }
        }
        let rotten_day = i as i32 + days[i];
        let count = apples[i];
        if count > 0 {
            pq.push(Reverse((rotten_day, count)));
        }
        if let Some(Reverse((rotten_day, mut count))) = pq.pop() {
            count -= 1;
            if count > 0 {
                pq.push(Reverse((rotten_day, count)));
            }
            ans += 1;
        }
        i += 1;
    }

    while let Some(Reverse((rotten_day, count))) = pq.pop() {
        if rotten_day <= i as i32 {
            continue;
        }
        let num = std::cmp::min(rotten_day - i as i32, count);
        ans += num;
        i += num as usize;
    }
    ans
}
