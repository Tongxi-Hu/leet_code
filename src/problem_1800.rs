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

/// 06
pub fn find_ball(grid: Vec<Vec<i32>>) -> Vec<i32> {
    let n = grid[0].len();
    let mut ans = vec![0; n];
    for j in 0..n {
        let mut cur_col = j as i32;
        for row in &grid {
            let d = row[cur_col as usize];
            cur_col += d;
            if cur_col < 0 || cur_col as usize == n || row[cur_col as usize] != d {
                cur_col = -1;
                break;
            }
        }
        ans[j] = cur_col;
    }
    ans
}

/// 07
pub fn maximize_xor(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    struct TrieNode {
        next: Vec<Option<TrieNode>>,
    }

    let mut nums = nums;
    nums.sort_unstable();
    let mut queries = queries;
    queries
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| v.push(i as i32));
    queries.sort_unstable_by(|a, b| a[1].cmp(&b[1]));

    let mut trie = TrieNode {
        next: vec![None, None],
    };

    let mut nums_i = 0;
    for i in 0..queries.len() {
        if queries[i][1] < nums[0] {
            queries[i].push(-1);
            continue;
        }
        while nums_i < nums.len() && queries[i][1] >= nums[nums_i] {
            let mut trie_ptr = &mut trie;
            for nums_j in (0..32).rev() {
                let bit = ((nums[nums_i] >> nums_j) & 1) as usize;
                if trie_ptr.next[bit].is_none() {
                    trie_ptr.next[bit] = Some(TrieNode {
                        next: vec![None, None],
                    });
                }
                trie_ptr = trie_ptr.next[bit].as_mut().unwrap();
            }
            nums_i += 1;
        }

        let x = queries[i][0];
        let mut trie_ptr = &trie;
        let mut y = 0;
        for j in (0..32).rev() {
            let best = 1 - ((x >> j) & 1) as usize;
            if trie_ptr.next[best].is_none() {
                y = y | ((1 - best) << j) as i32;
                trie_ptr = trie_ptr.next[1 - best].as_ref().unwrap();
            } else {
                y = y | (best << j) as i32;
                trie_ptr = trie_ptr.next[best].as_ref().unwrap();
            }
        }
        queries[i].push(x ^ y);
    }
    queries.sort_unstable_by(|a, b| a[2].cmp(&b[2]));
    queries.into_iter().map(|x| x[3]).collect()
}

/// 10
pub fn maximum_units(mut box_types: Vec<Vec<i32>>, mut truck_size: i32) -> i32 {
    box_types.sort_by(|a, b| b[1].cmp(&a[1]));
    let mut total = 0;
    for i in 0..box_types.len() {
        if truck_size >= box_types[i][0] {
            total += box_types[i][0] * box_types[i][1];
            truck_size -= box_types[i][0];
        } else {
            total += truck_size * box_types[i][1];
            break;
        }
    }
    total
}
