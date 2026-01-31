use std::{
    cell::RefCell,
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    i32,
    rc::Rc,
};

use crate::common::{ListNode, TreeNode};

/// 1201
pub fn nth_ugly_number(n: i32, a: i32, b: i32, c: i32) -> i32 {
    fn lcm(a: i64, b: i64) -> i64 {
        a * b / gcd(a, b)
    }

    fn gcd(a: i64, b: i64) -> i64 {
        if a == 0 {
            return b;
        }
        gcd(b % a, a)
    }
    let a = a as i64;
    let b = b as i64;
    let c = c as i64;
    let mut lo = 0;
    let mut hi = 2_000_000_000;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let cnt = mid / a + mid / b + mid / c - mid / lcm(a, b) - mid / lcm(b, c) - mid / lcm(a, c)
            + mid / lcm(a, lcm(b, c));

        if cnt < n as i64 {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    lo as i32
}

/// 1202
pub fn smallest_string_with_swaps(s: String, pairs: Vec<Vec<i32>>) -> String {
    if pairs.is_empty() {
        return s;
    }

    let len = s.len();

    let mut parent = (0..len).collect::<Vec<usize>>();
    let mut rank = vec![1usize; len];

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if x != parent[x] {
            parent[x] = find(parent, parent[x]);
        }
        return parent[x];
    }

    fn union(parent: &mut Vec<usize>, rank: &mut Vec<usize>, x: usize, y: usize) {
        let root_x = find(parent, x);
        let root_y = find(parent, y);
        if root_x == root_y {
            return;
        }
        if rank[root_x] == rank[root_y] {
            parent[root_x] = root_y;
            rank[root_y] += 1;
        } else if rank[root_x] < rank[root_y] {
            parent[root_x] = parent[root_y];
        } else {
            parent[root_y] = parent[root_x];
        }
    }

    for pair in pairs {
        union(&mut parent, &mut rank, pair[0] as usize, pair[1] as usize);
    }

    let s = s.into_bytes();
    let mut map = std::collections::HashMap::<
        usize,
        std::collections::BinaryHeap<std::cmp::Reverse<u8>>,
    >::new();
    for i in 0..len {
        let root = find(&mut parent, i);
        if map.contains_key(&root) {
            map.get_mut(&root).unwrap().push(std::cmp::Reverse(s[i]));
        } else {
            let mut min_heap = std::collections::BinaryHeap::<std::cmp::Reverse<u8>>::new();
            min_heap.push(std::cmp::Reverse(s[i]));
            map.insert(root, min_heap);
        }
    }

    let mut ret = Vec::<u8>::with_capacity(len);

    for i in 0..len {
        let root = find(&mut parent, i);
        let c = map.get_mut(&root).unwrap().pop().unwrap().0;
        ret.push(c);
    }
    String::from_utf8(ret).unwrap()
}

/// 1203
pub fn sort_items(
    n: i32,
    mut m: i32,
    mut group: Vec<i32>,
    before_items: Vec<Vec<i32>>,
) -> Vec<i32> {
    fn topo_sort(in_degree: &mut Vec<i32>, graph: &Vec<Vec<usize>>) -> Vec<usize> {
        let mut queue = VecDeque::new();
        let mut result = vec![];

        for (i, &val) in in_degree.iter().enumerate() {
            if val == 0 {
                queue.push_back(i);
                result.push(i);
            }
        }

        while let Some(id) = queue.pop_front() {
            for &next_id in graph[id].iter() {
                in_degree[next_id] -= 1;
                if in_degree[next_id] == 0 {
                    queue.push_back(next_id);
                    result.push(next_id);
                }
            }
        }

        if result.len() != in_degree.len() {
            result.clear();
        }
        result
    }

    let mut group_counts = vec![0; m as usize];
    for g in group.iter_mut() {
        if *g == -1 {
            *g = m;
            m += 1;
            group_counts.push(1);
        } else {
            group_counts[*g as usize] += 1;
        }
    }

    let (n, m) = (n as usize, m as usize);
    let mut group_in_degree = vec![0; m];
    let mut group_graph = vec![vec![]; m];

    for (i, item) in before_items.iter().enumerate() {
        let cur_group_id = group[i] as usize;
        for &before in item {
            let before_group_id = group[before as usize] as usize;
            if cur_group_id != before_group_id {
                group_in_degree[cur_group_id] += 1;
                group_graph[before_group_id].push(cur_group_id);
            }
        }
    }

    let group_topo_sort = topo_sort(&mut group_in_degree, &group_graph);
    if group_topo_sort.is_empty() {
        return vec![];
    }

    let mut item_in_degree = vec![0; n];
    let mut item_graph = vec![vec![]; n];
    for (i, items) in before_items.iter().enumerate() {
        item_in_degree[i] = items.len() as i32;
        for &before in items {
            item_graph[before as usize].push(i);
        }
    }

    let mut queue = vec![VecDeque::new(); m];
    for (i, &val) in item_in_degree.iter().enumerate() {
        if val == 0 {
            queue[group[i] as usize].push_back(i);
        }
    }

    let mut result = vec![];

    for cur_group in group_topo_sort.into_iter() {
        let mut cur_count = 0;

        while let Some(cur_item) = queue[cur_group].pop_front() {
            cur_count += 1;
            result.push(cur_item as i32);

            for &next_item in item_graph[cur_item].iter() {
                item_in_degree[next_item] -= 1;
                if item_in_degree[next_item] == 0 {
                    queue[group[next_item] as usize].push_back(next_item);
                }
            }
        }

        if cur_count != group_counts[cur_group] {
            return vec![];
        }
    }

    result
}

/// 1206
struct Skiplist {
    d: Vec<i32>,
}

impl Skiplist {
    fn new() -> Self {
        Self { d: vec![0; 20001] }
    }

    fn search(&self, target: i32) -> bool {
        self.d[target as usize] > 0
    }

    fn add(&mut self, num: i32) {
        self.d[num as usize] += 1;
    }

    fn erase(&mut self, num: i32) -> bool {
        match self.d[num as usize] {
            0 => false,
            _ => {
                self.d[num as usize] -= 1;
                true
            }
        }
    }
}

/// 1207
pub fn unique_occurrences(arr: Vec<i32>) -> bool {
    let total = arr.iter().fold(HashMap::new(), |mut acc, cur| {
        let cnt = acc.entry(cur).or_insert(0);
        *cnt = *cnt + 1;
        acc
    });
    total.values().collect::<HashSet<&i32>>().len() == total.len()
}

/// 1208
pub fn equal_substring(s: String, t: String, max_cost: i32) -> i32 {
    fn cost(a: char, b: char) -> i32 {
        let (a, b) = (a as u8, b as u8);
        if a >= b {
            (a - b) as i32
        } else {
            (b - a) as i32
        }
    }
    let (length, s_char, t_char) = (
        s.len(),
        s.chars().collect::<Vec<char>>(),
        t.chars().collect::<Vec<char>>(),
    );
    let (mut l, mut diff, mut max_len) = (0, 0, 0);
    for r in 0..length {
        diff = diff + cost(s_char[r], t_char[r]);
        while diff > max_cost {
            diff = diff - cost(s_char[l], t_char[l]);
            l = l + 1;
        }
        max_len = max_len.max(r - l + 1);
    }

    max_len as i32
}

/// 1209
pub fn remove_duplicates(s: String, k: i32) -> String {
    let chars = s.chars().collect::<Vec<char>>();
    let cnt = chars.iter().fold(VecDeque::new(), |mut acc, &c| {
        if acc.back().is_some() {
            let last: (char, i32) = acc.pop_back().unwrap();
            if last.0 == c {
                if last.1 != k - 1 {
                    acc.push_back((c, last.1 + 1));
                }
            } else {
                acc.push_back(last);
                acc.push_back((c, 1));
            }
        } else {
            acc.push_back((c, 1));
        }
        acc
    });
    cnt.iter().fold("".to_string(), |mut acc, cur| {
        (0..cur.1).into_iter().for_each(|_| acc.push(cur.0));
        acc
    })
}

/// 1210
pub fn minimum_moves(grid: Vec<Vec<i32>>) -> i32 {
    #[derive(Clone)]
    struct Point {
        x: i32,
        y: i32,
        s: i32,
    }

    let dirs = [
        Point { x: 1, y: 0, s: 0 },
        Point { x: 0, y: 1, s: 0 },
        Point { x: 0, y: 0, s: 1 },
    ];
    let n: i32 = grid.len() as i32;
    let mut visit = vec![vec![vec![false; 2]; n as usize]; n as usize];
    visit[0][0][0] = true;
    let mut q = vec![Some(Point { x: 0, y: 0, s: 0 })];
    let mut step = 1;
    while !q.is_empty() {
        let mut tmp: Vec<Option<Point>> = vec![];
        //tmp.clone_from_slice(&q);
        tmp = q.clone();
        q.clear();
        while let Some(t) = tmp.pop() {
            let t1 = t.unwrap();
            for d in &dirs {
                let (x, y, s) = (t1.x + d.x, t1.y + d.y, t1.s ^ d.s);
                let (x2, y2) = (x + s, y + (s ^ 1));
                if x2 < n
                    && y2 < n
                    && !visit
                        .get(x as usize)
                        .unwrap()
                        .get(y as usize)
                        .unwrap()
                        .get(s as usize)
                        .unwrap()
                    && grid[x as usize][y as usize] == 0
                    && grid[x2 as usize][y2 as usize] == 0
                    && (d.s == 0 || grid[(x + 1) as usize][(y + 1) as usize] == 0)
                {
                    if x == n - 1 && y == n - 2 {
                        return step;
                    }
                    visit[x as usize][y as usize][s as usize] = true;
                    q.push(Some(Point { x: x, y: y, s: s }));
                }
            }
        }
        step += 1;
    }
    -1
}

/// 1217
pub fn min_cost_to_move_chips(position: Vec<i32>) -> i32 {
    let cnt = position.iter().fold((0, 0), |mut acc, cur| {
        if cur % 2 == 0 {
            acc.0 = acc.0 + 1;
        } else {
            acc.1 = acc.1 + 1
        }
        acc
    });
    return cnt.1.min(cnt.0);
}

/// 1218
pub fn longest_subsequence(arr: Vec<i32>, difference: i32) -> i32 {
    let mut record = HashMap::new();
    arr.iter().for_each(|i| {
        let last = i - difference;
        if record.contains_key(&last) {
            record.insert(i, record.get(&last).unwrap() + 1);
        } else {
            record.insert(i, 1);
        }
    });
    *record.values().max().unwrap()
}

/// 1219
pub fn get_maximum_gold(mut grid: Vec<Vec<i32>>) -> i32 {
    let (mut gold, mut max) = (0, 0);
    fn dfs(grid: &mut Vec<Vec<i32>>, location: (usize, usize), gold: &mut i32, max: &mut i32) {
        let (i, j) = location;
        let direction = vec![(-1, 0), (1, 0), (0, 1), (0, -1)];
        let current = grid[i][j];
        grid[i][j] = 0;
        *gold = *gold + current;
        let is_boundary = direction.iter().fold(true, |mut is_boundary, step| {
            let (new_x, new_y) = ((i as i32 + step.0) as usize, (j as i32 + step.1) as usize);
            if new_x < grid.len() && new_y < grid[0].len() && grid[new_x][new_y] != 0 {
                dfs(grid, (new_x, new_y), gold, max);
                is_boundary = false;
                is_boundary
            } else {
                is_boundary
            }
        });
        if is_boundary {
            *max = *max.max(gold);
        }
        grid[i][j] = current;
        *gold = *gold - current;
    }
    for i in 0..grid.len() {
        for j in 0..grid[0].len() {
            if grid[i][j] != 0 {
                dfs(&mut grid, (i, j), &mut gold, &mut max);
            }
        }
    }
    max
}

/// 1220
pub fn count_vowel_permutation(n: i32) -> i32 {
    let m: i64 = 1000000007;
    ((1..n)
        .into_iter()
        .fold((1, 1, 1, 1, 1, 5), |(a, e, i, o, u, _), _| {
            let (_a, _e, _i, _o, _u) = ((e + i + u) % m, (a + i) % m, (e + o) % m, i, (i + o) % m);
            (_a, _e, _i, _o, _u, (_a + _e + _i + _o + _u))
        })
        .5
        % m) as i32
}

/// 1221
pub fn balanced_string_split(s: String) -> i32 {
    s.chars()
        .fold((0, 0), |(mut diff, cut), cur| {
            if cur == 'R' {
                diff = diff + 1;
            } else {
                diff = diff - 1;
            }
            if diff == 0 {
                (diff, cut + 1)
            } else {
                (diff, cut)
            }
        })
        .1
}

/// 1222
pub fn queens_attackthe_king(queens: Vec<Vec<i32>>, king: Vec<i32>) -> Vec<Vec<i32>> {
    let dir = vec![
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
    ];

    dir.iter().fold(vec![], |mut acc, d| {
        for i in 0..8 {
            let (r, c) = ((king[0] as i32) + i * d.0, (king[1] as i32) + i * d.1);
            if r < 8 && c < 8 {
                if queens.iter().find(|q| **q == vec![r, c]).is_some() {
                    acc.push(vec![r, c]);
                    break;
                }
            }
        }
        acc
    })
}

/// 1223
pub fn die_simulator(n: i32, roll_max: Vec<i32>) -> i32 {
    let mut dp = vec![vec![1_i64; 6]; n as usize];

    for i in 1..n as usize {
        for j in 0..6 {
            dp[i][j] = 0;

            for k in 0..6 {
                dp[i][j] += dp[i - 1][k];
            }

            if (roll_max[j] as usize) < i {
                for k in 0..6 {
                    if k != j {
                        dp[i][j] -= dp[i - 1 - roll_max[j] as usize][k]
                    }
                }
            }

            if roll_max[j] as usize == i {
                dp[i][j] -= 1;
            }

            dp[i][j] %= 1_000_000_007;

            if dp[i][j] < 0 {
                dp[i][j] += 1_000_000_007;
            }
        }
    }

    (dp[n as usize - 1].iter().sum::<i64>() % 1_000_000_007) as i32
}

/// 1224
pub fn max_equal_freq(nums: Vec<i32>) -> i32 {
    let (mut cnt, mut freq) = (vec![0; 100001], vec![0; 100001]);
    nums.iter().for_each(|num| {
        cnt[*num as usize] += 1;
        freq[cnt[*num as usize] as usize] += 1;
    });
    for i in (1..nums.len()).rev() {
        if freq[cnt[nums[i] as usize] as usize] * cnt[nums[i] as usize] == i {
            return i as i32 + 1;
        }
        freq[cnt[nums[i] as usize] as usize] -= 1;
        cnt[nums[i] as usize] -= 1;
        if freq[cnt[nums[i - 1] as usize] as usize] * cnt[nums[i - 1] as usize] == i {
            return i as i32 + 1;
        }
    }
    1
}

/// 1227
pub fn nth_person_gets_nth_seat(n: i32) -> f64 {
    if n == 1 { 1.0 } else { 0.5 }
}

/// 1232
pub fn check_straight_line(coordinates: Vec<Vec<i32>>) -> bool {
    if coordinates.len() == 2 {
        true
    } else {
        if coordinates[1][1] == coordinates[0][1] {
            return coordinates.iter().all(|c| c[1] == coordinates[0][1]);
        } else if coordinates[1][0] == coordinates[0][0] {
            return coordinates.iter().all(|c| c[0] == coordinates[0][0]);
        } else {
            let slop = ((coordinates[1][1] - coordinates[0][1]) as f64)
                / ((coordinates[1][0] - coordinates[0][0]) as f64);
            return coordinates.iter().skip(1).all(|c| {
                ((c[1] - coordinates[0][1]) as f64) / ((c[0] - coordinates[0][0]) as f64) == slop
            });
        };
    }
}

/// 1233
pub fn remove_subfolders(folder: Vec<String>) -> Vec<String> {
    if folder.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<String> = Vec::new();
    folder.iter().for_each(|f| {
        if let Some(prev) = result.last() {
            let prefix = format!("{}/", prev);
            if !f.starts_with(&prefix) {
                result.push(f.clone());
            }
        }
    });
    result
}

/// 1234
pub fn balanced_string(s: String) -> i32 {
    let (mut cnt_q, mut cnt_w, mut cnt_e, mut cnt_r) = (0_i32, 0_i32, 0_i32, 0_i32);
    let (n, mut left, mut right): (usize, usize, usize) = (s.len(), 0_usize, 0_usize);
    let mut s_char: Vec<char> = vec![' '; n];
    for c in s.chars() {
        match c {
            'Q' => cnt_q += 1,
            'W' => cnt_w += 1,
            'E' => cnt_e += 1,
            _ => cnt_r += 1,
        }
        s_char[left] = c;
        left += 1;
    }
    let target: i32 = s.len() as i32 / 4;
    if cnt_q == target && cnt_w == target && cnt_e == target && cnt_r == target {
        return 0;
    }
    let mut res: usize = 2147483647_usize;
    left = 0;
    while right < n {
        while (cnt_q > target || cnt_w > target || cnt_e > target || cnt_r > target) && right < n {
            match s_char[right] {
                'Q' => cnt_q -= 1,
                'W' => cnt_w -= 1,
                'E' => cnt_e -= 1,
                _ => cnt_r -= 1,
            }
            right += 1;
        }
        if cnt_q > target || cnt_w > target || cnt_e > target || cnt_r > target {
            break;
        }
        while left < right
            && cnt_q <= target
            && cnt_w <= target
            && cnt_e <= target
            && cnt_r <= target
        {
            match s_char[left] {
                'Q' => cnt_q += 1,
                'W' => cnt_w += 1,
                'E' => cnt_e += 1,
                _ => cnt_r += 1,
            }
            left += 1;
        }
        res = std::cmp::min(res, right - left + 1);
    }
    res as i32
}

/// 1235
pub fn job_scheduling(start_time: Vec<i32>, end_time: Vec<i32>, profit: Vec<i32>) -> i32 {
    let mut jobs = start_time
        .iter()
        .zip(end_time.iter())
        .zip(profit.iter())
        .map(|((&s, &e), &p)| (s, e, p))
        .collect::<Vec<_>>();
    jobs.sort_unstable_by(|a, b| a.1.cmp(&b.1));

    let n = jobs.len();
    let mut f = vec![0; n + 1];
    for (i, &(st, _, p)) in jobs.iter().enumerate() {
        let j = jobs[..i].partition_point(|job| job.1 <= st);
        f[i + 1] = f[i].max(f[j] + p);
    }
    f[n]
}

/// 1238
pub fn circular_permutation(n: i32, start: i32) -> Vec<i32> {
    let mut res: Vec<i32> = vec![0, 1];
    for bit in 1..n {
        let next: Vec<i32> = res.iter().rev().map(|x| (*x) | (1 << bit)).collect();
        res.extend(next.iter());
    }
    let (mut first, mut second) = (vec![], vec![]);
    let mut flag: bool = false;
    for num in res {
        if num == start {
            flag = true
        }
        if flag {
            second.push(num);
        } else {
            first.push(num);
        }
    }
    second.extend(first.iter());
    second
}

/// 1239
pub fn max_length(arr: Vec<String>) -> i32 {
    let mut ans: i32 = 0;
    let mut masks: Vec<i32> = vec![0];
    for s in arr {
        let mut mask = 0;
        for ch in s.chars() {
            let mut ch = ch as usize;
            ch = ch as usize - 97;
            if (mask >> ch) & 1 != 0 {
                mask = 0;
                break;
            }
            mask |= 1 << ch;
        }
        if mask == 0 {
            continue;
        }
        let n = masks.len();
        for i in 0..n {
            let m = masks[i];
            if m & mask == 0 {
                masks.push(m | mask);
                ans = ans.max((((m | mask) as usize).count_ones()) as i32);
            }
        }
    }
    return ans;
}

/// 1240
pub fn tiling_rectangle(n: i32, m: i32) -> i32 {
    if n > m {
        return tiling_rectangle(m, n);
    }
    fn cal(n: i32, m: i32, cache: &mut Vec<Vec<i32>>) -> i32 {
        if n > m {
            return cal(m, n, cache);
        }
        if n == 0 {
            return 0;
        }
        if n == m {
            return 1;
        }
        if n == 1 {
            return m;
        }
        if cache[n as usize][m as usize] > 0 {
            return cache[n as usize][m as usize];
        }
        let (mut count, max_size) = (i32::MAX, m.min(n));
        for i in 1..=max_size {
            count = count.min(1 + cal(n - i, m, cache) + cal(i, m - i, cache));
            count = count.min(1 + cal(n, m - i, cache) + cal(n - i, i, cache));
            let mut j = n - i + 1;
            while j < m - i && j < n {
                count = count.min(
                    2 + cal(n - i, m - j, cache)
                        + cal(n - j, m - i, cache)
                        + cal(i + j - n, m - i - j, cache),
                );
                j += 1;
            }
        }
        cache[n as usize][m as usize] = count;
        count
    }
    cal(n, m, &mut vec![vec![0; m as usize + 1]; n as usize + 1])
}

/// 1247
pub fn minimum_swap(s1: String, s2: String) -> i32 {
    let cnt = s1.chars().zip(s2.chars()).fold((0, 0), |acc, cur| {
        if cur.0 == cur.1 {
            return acc;
        } else if cur.0 == 'x' && cur.1 == 'y' {
            return (acc.0 + 1, acc.1);
        } else {
            return (acc.0, acc.1 + 1);
        }
    });
    if (cnt.0 + cnt.1) % 2 == 1 {
        return -1;
    } else {
        cnt.0 / 2 + cnt.0 % 2 + cnt.1 / 2 + cnt.1 % 2
    }
}

/// 1248
pub fn number_of_subarrays(nums: Vec<i32>, k: i32) -> i32 {
    let k = k as usize;
    let mut cnt = vec![0; nums.len() + 1];
    cnt[0] = 1;
    nums.iter()
        .fold((0, 0, cnt), |mut acc, cur| {
            if cur % 2 == 1 {
                acc.1 = acc.1 + 1;
            }
            acc.2[acc.1] = acc.2[acc.1] + 1;
            if acc.1 >= k {
                acc.0 = acc.0 + acc.2[acc.1 - k]
            };
            (acc.0, acc.1, acc.2)
        })
        .0
}

/// 1249
pub fn min_remove_to_make_valid(s: String) -> String {
    let (mut stack, mut invalid) = (VecDeque::new(), Vec::new());
    s.chars()
        .enumerate()
        .filter(|(_, c)| *c == '(' || *c == ')')
        .for_each(|(i, c)| {
            if c == '(' {
                stack.push_back(i);
            } else {
                if stack.is_empty() {
                    invalid.push(i)
                } else {
                    stack.pop_back();
                }
            }
        });
    s.chars()
        .enumerate()
        .fold("".to_string(), |mut acc, (i, cur)| {
            if !stack.contains(&i) && !invalid.contains(&i) {
                acc.push(cur);
            }
            acc
        })
}

///1250
pub fn is_good_array(nums: Vec<i32>) -> bool {
    let gcd = |mut first: i32, mut second: i32| -> i32 {
        let mut temp: i32;
        if first < second {
            temp = second;
            second = first;
            first = temp;
        }
        while second != 0 {
            temp = first % second;
            first = second;
            second = temp;
        }
        first
    };
    let mut res: i32 = nums[0];
    for index in 1..nums.len() {
        res = gcd(res, nums[index]);
    }
    res == 1
}

/// 1252
pub fn odd_cells(m: i32, n: i32, indices: Vec<Vec<i32>>) -> i32 {
    let (m, n) = (m as usize, n as usize);
    let mut matrix = vec![vec![0; n]; m];
    indices.iter().for_each(|v| {
        matrix[v[0] as usize].iter_mut().for_each(|n| *n = *n + 1);
        matrix.iter_mut().for_each(|r| {
            r[v[1] as usize] = r[v[1] as usize] + 1;
        });
    });
    let mut odd = 0;
    for i in 0..m {
        for j in 0..n {
            if matrix[i][j] % 2 == 1 {
                odd = odd + 1;
            }
        }
    }
    odd
}

/// 1253
pub fn reconstruct_matrix(upper: i32, lower: i32, colsum: Vec<i32>) -> Vec<Vec<i32>> {
    let (row, col) = (2, colsum.len());
    let (mut ans, mut upper, mut lower) = (vec![vec![0; col]; row], upper, lower);
    colsum.iter().enumerate().for_each(|(i, s)| match s {
        2 => {
            ans[0][i] = 1;
            ans[1][i] = 1;
            upper = upper - 1;
            lower = lower - 1;
        }
        1 => {
            if upper >= lower {
                ans[0][i] = 1;
                ans[1][i] = 0;
                upper = upper - 1;
            } else {
                ans[0][i] = 0;
                ans[1][i] = 1;
                lower = lower - 1;
            }
        }
        _ => (),
    });
    if upper != 0 || lower != 0 {
        return vec![];
    }
    ans
}

/// 1254
pub fn closed_island(mut grid: Vec<Vec<i32>>) -> i32 {
    let (height, width) = (grid.len(), grid[0].len());
    let mut islands = vec![];
    fn dfs(location: (usize, usize), island: &mut Vec<(usize, usize)>, grid: &mut Vec<Vec<i32>>) {
        let (height, width) = (grid.len(), grid[0].len());
        if grid[location.0][location.1] == 0 {
            island.push(location);
            grid[location.0][location.1] = 2;
            for d in vec![(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let new_location = (location.0 + d.0 as usize, location.1 + d.1 as usize);
                if new_location.0 < height && new_location.1 < width {
                    dfs(new_location, island, grid);
                }
            }
        }
    }
    for r in 0..height {
        for c in 0..width {
            if grid[r][c] == 0 {
                let mut island = vec![];
                dfs((r, c), &mut island, &mut grid);
                islands.push(island)
            }
        }
    }
    islands
        .iter()
        .filter(|island| {
            island.iter().all(|location| {
                location.0 != 0
                    && location.0 != height - 1
                    && location.1 != 0
                    && location.1 != width - 1
            })
        })
        .collect::<Vec<_>>()
        .len() as i32
}

/// 1255
pub fn max_score_words(words: Vec<String>, letters: Vec<char>, score: Vec<i32>) -> i32 {
    let letters = letters.into_iter().fold([0; 26], |mut t, c| {
        t[c as usize - b'a' as usize] += 1;
        t
    });
    let words: Vec<_> = words
        .into_iter()
        .map(|w| {
            let mut w = w.into_bytes();
            w.iter_mut().for_each(|c| *c -= b'a');
            w
        })
        .collect();
    let mut ans = 0;
    for i in 1..1 << words.len() {
        let mut tab = [0; 26];
        let mut sum = 0;
        for j in 0..words.len() {
            if i & (1 << j) == 0 {
                continue;
            }
            for &c in words[j].iter() {
                let c = c as usize;
                tab[c] += 1;
                if tab[c] > letters[c] {
                    sum = 0;
                    break;
                }
                sum += score[c];
            }
            ans = ans.max(sum);
        }
    }
    ans
}

/// 1260
pub fn shift_grid(grid: Vec<Vec<i32>>, mut k: i32) -> Vec<Vec<i32>> {
    let (m, n) = (grid.len(), grid[0].len());
    let total = m * n;
    k %= total as i32;
    let mut ret = vec![vec![0; n]; m];
    let (mut i, mut j) = (0, total - k as usize);
    while i < total {
        j = if j == total { 0 } else { j };
        ret[i / n][i % n] = grid[j / n][j % n];
        j += 1;
        i += 1;
    }
    ret
}

/// 1261
struct FindElements {
    root: Option<Rc<RefCell<TreeNode>>>,
}

impl FindElements {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, v: i32) {
        if let Some(node) = root {
            node.borrow_mut().val = v;
            Self::dfs(node.borrow().left.clone(), v * 2 + 1);
            Self::dfs(node.borrow().right.clone(), v * 2 + 2);
        }
    }
    fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        Self::dfs(root.clone(), 0);
        Self { root }
    }

    fn find_node(root: Option<Rc<RefCell<TreeNode>>>, target: i32) -> bool {
        if let Some(node) = root {
            match node.borrow().val.cmp(&target) {
                Ordering::Equal => {
                    return true;
                }
                Ordering::Less => {
                    return Self::find_node(node.borrow().left.clone(), target)
                        || Self::find_node(node.borrow().right.clone(), target);
                }
                Ordering::Greater => {
                    return false;
                }
            }
        }
        false
    }

    fn find(&self, target: i32) -> bool {
        let root = self.root.clone();
        Self::find_node(root, target)
    }
}

/// 1262
pub fn min_push_box(grid: Vec<Vec<char>>) -> i32 {
    fn get_pos(grid: &Vec<Vec<char>>, n: usize, m: usize, target: char) -> (usize, usize) {
        for i in 0..n {
            for j in 0..m {
                if grid[i][j] == target {
                    return (i, j);
                }
            }
        }

        unreachable!()
    }

    let n = grid.len();
    let m = grid[0].len();
    let dirs = [(-1, 0), (0, -1), (0, 1), (1, 0)];
    let (pi, pj) = get_pos(&grid, n, m, 'S');
    let (bi, bj) = get_pos(&grid, n, m, 'B');
    let (ti, tj) = get_pos(&grid, n, m, 'T');
    let mut vis = HashSet::new();
    let mut pq = BinaryHeap::new();
    pq.push((Reverse(0), bi, bj, pi, pj));
    vis.insert((bi, bj, pi, pj));

    while let Some((Reverse(cost), bi, bj, pi, pj)) = pq.pop() {
        if (ti, tj) == (bi, bj) {
            return cost;
        }

        for &(di, dj) in dirs.iter() {
            let px = (pi as i32 + di) as usize;
            let py = (pj as i32 + dj) as usize;

            if px >= n || py >= m {
                continue;
            }

            if grid[px][py] == '#' {
                continue;
            }

            let mut c = cost;
            let mut bx = bi;
            let mut by = bj;

            if (px, py) == (bx, by) {
                bx = (bx as i32 + di) as usize;
                by = (by as i32 + dj) as usize;

                if bx >= n || by >= m {
                    continue;
                }

                if grid[bx][by] == '#' {
                    continue;
                }

                c += 1;
            }

            if !vis.insert((bx, by, px, py)) {
                continue;
            }

            pq.push((Reverse(c), bx, by, px, py));
        }
    }

    -1
}

/// 1266
pub fn min_time_to_visit_all_points(points: Vec<Vec<i32>>) -> i32 {
    points.windows(2).fold(0, |acc, cur| {
        let (pre, cur) = (&cur[0], &cur[1]);
        acc + (pre[0] - cur[0]).abs().max((pre[1] - cur[1]).abs())
    })
}

/// 1267
pub fn count_servers(grid: Vec<Vec<i32>>) -> i32 {
    let (height, width) = (grid.len(), grid[0].len());
    let (mut r_cnt, mut c_cnt) = (HashMap::new(), HashMap::new());
    for i in 0..height {
        for j in 0..width {
            if grid[i][j] == 1 {
                let r_cnt = r_cnt.entry(i).or_insert(0);
                *r_cnt = *r_cnt + 1;
                let c_cnt = c_cnt.entry(j).or_insert(0);
                *c_cnt = *c_cnt + 1;
            }
        }
    }
    let mut total = 0;
    for i in 0..height {
        for j in 0..width {
            if grid[i][j] == 1 {
                let r_cnt = r_cnt.get(&i).unwrap();
                let c_cnt = c_cnt.get(&j).unwrap();
                if *r_cnt > 1 || *c_cnt > 1 {
                    total = total + 1;
                }
            }
        }
    }
    total
}

/// 1268
struct KTrie {
    is_end: bool,
    next: HashMap<char, Box<KTrie>>,

    k: usize,
    pub recommendations: BinaryHeap<String>,
}

impl KTrie {
    pub fn new(k: usize) -> Self {
        Self {
            is_end: false,
            next: HashMap::new(),
            k,
            recommendations: BinaryHeap::new(),
        }
    }

    fn add_recommendation(&mut self, s: &str) {
        self.recommendations.push(s.to_owned());
        if self.recommendations.len() > self.k {
            self.recommendations.pop();
        }
    }

    pub fn insert(&mut self, s: &str) {
        let k = self.k;
        let mut cur = self;
        cur.add_recommendation(s);
        for c in s.chars() {
            cur = &mut **cur.next.entry(c).or_insert_with(|| Box::new(KTrie::new(k)));
            cur.add_recommendation(s);
        }

        cur.is_end = true;
    }

    pub fn walk<'a>(&'a self, s: &'a str) -> impl Iterator<Item = Option<&'a KTrie>> {
        let mut cur = Some(self);
        let path = s.chars();

        path.map(move |c| {
            if let Some(node) = cur {
                cur = node.next.get(&c).map(|b| &**b);
                cur
            } else {
                None
            }
        })
    }
}

pub fn suggested_products(products: Vec<String>, search_word: String) -> Vec<Vec<String>> {
    let mut root = KTrie::new(3);
    for product in products {
        root.insert(&product);
    }

    root.walk(&search_word)
        .map(|node| {
            node.map(|node| {
                let mut v: Vec<_> = node.recommendations.iter().cloned().collect();
                v.sort_unstable();
                v
            })
            .unwrap_or_else(|| vec![])
        })
        .collect()
}

/// 1269
pub fn num_ways(steps: i32, arr_len: i32) -> i32 {
    use std::collections::HashMap;
    fn dfs(pos: usize, steps: i32, arr_len: usize, memo: &mut HashMap<(usize, i32), i32>) -> i32 {
        if steps == 0 {
            if pos == 0 {
                return 1;
            }
            return 0;
        }

        if let Some(&sum) = memo.get(&(pos, steps)) {
            return sum;
        }
        let mut sum = dfs(pos, steps - 1, arr_len, memo);
        if pos < arr_len - 1 {
            sum += dfs(pos + 1, steps - 1, arr_len, memo);
            sum = sum % 1000000007;
        }
        if pos > 0 {
            sum += dfs(pos - 1, steps - 1, arr_len, memo);
            sum = sum % 1000000007;
        }
        memo.insert((pos, steps), sum);
        sum
    }

    dfs(0, steps, arr_len as usize, &mut HashMap::new())
}

/// 1275
pub fn tictactoe(moves: Vec<Vec<i32>>) -> String {
    let mut a = 0;
    let mut b = 0;
    let n = moves.len();

    for i in 0..n {
        if i % 2 == 0 {
            a ^= 1 << (3 * moves[i][0] + moves[i][1]);
        } else {
            b ^= 1 << (3 * moves[i][0] + moves[i][1]);
        }
    }

    let wins = [7, 56, 448, 73, 146, 292, 273, 84];

    for win in wins {
        if a & win == win {
            return "A".into();
        }
        if b & win == win {
            return "B".into();
        }
    }

    if n == 9 {
        "Draw".into()
    } else {
        "Pending".into()
    }
}

/// 1276
pub fn num_of_burgers(tomato_slices: i32, cheese_slices: i32) -> Vec<i32> {
    if tomato_slices > cheese_slices * 4
        || tomato_slices < cheese_slices * 2
        || tomato_slices % 2 != 0
    {
        vec![]
    } else {
        let jumbo = (tomato_slices - cheese_slices * 2) / 4;
        vec![jumbo, cheese_slices - jumbo]
    }
}

/// 1277
pub fn count_squares(matrix: Vec<Vec<i32>>) -> i32 {
    let (height, width) = (matrix.len(), matrix[0].len());
    let (mut dp, mut total) = (vec![vec![0; width]; height], 0);
    for r in 0..height {
        for c in 0..width {
            dp[r][c] = if r == 0 || c == 0 {
                matrix[r][c]
            } else if matrix[r][c] == 0 {
                0
            } else {
                dp[r - 1][c].min(dp[r][c - 1]).min(dp[r - 1][c - 1]) + 1
            };
            total = total + dp[r][c];
        }
    }
    total
}

/// 1278
pub fn palindrome_partition(s: String, k: i32) -> i32 {
    fn cost(mut l: usize, mut r: usize, s: &String) -> i32 {
        let mut ret = 0;
        while l < r {
            if s.as_bytes()[l] != s.as_bytes()[r] {
                ret += 1;
            }
            l += 1;
            r -= 1;
        }
        ret
    }
    let (length, k) = (s.len(), k as usize);
    let mut f = vec![vec![i32::MAX; k + 1]; length + 1];
    f[0][0] = 0;
    for i in 1..=length {
        for j in 1..=k.min(i) {
            if j == 1 {
                f[i][j] = cost(0, i - 1, &s);
            } else {
                for l in (j - 1)..i {
                    f[i][j] = f[i][j].min(f[l][j - 1] + cost(l, i - 1, &s));
                }
            }
        }
    }
    f[length][k as usize]
}

/// 1281
pub fn subtract_product_and_sum(mut n: i32) -> i32 {
    let (mut sum, mut product) = (0, 1);
    while n > 0 {
        let remain = n % 10;
        sum = sum + remain;
        product = product * remain;
        n = n / 10;
    }
    product - sum
}

/// 1282
pub fn group_the_people(group_sizes: Vec<i32>) -> Vec<Vec<i32>> {
    group_sizes
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut acc, (i, size)| {
            let index = acc.entry(size).or_insert(Vec::new());
            index.push(i as i32);
            acc
        })
        .iter()
        .fold(Vec::new(), |mut acc, (&&size, index)| {
            index
                .chunks(size as usize)
                .for_each(|i| acc.push(i.to_vec()));
            acc
        })
}

/// 1283
pub fn smallest_divisor(nums: Vec<i32>, threshold: i32) -> i32 {
    fn up_divde(target: i32, divisor: i32) -> i32 {
        if target % divisor == 0 {
            target / divisor
        } else {
            target / divisor + 1
        }
    }
    let (mut l, mut r, mut ans) = (1, nums.iter().max().unwrap().clone(), -1);
    while l <= r {
        let mid = (l + r) / 2;
        let total = nums.iter().fold(0, |acc, &cur| acc + up_divde(cur, mid));
        if total <= threshold {
            ans = mid;
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    }
    ans
}

/// 1284
pub fn min_flips(mat: Vec<Vec<i32>>) -> i32 {
    fn convert(mat: &mut Vec<Vec<i32>>, m: usize, n: usize, i: usize, j: usize) {
        let dirs: [i32; 6] = [0, 1, 0, -1, 0, 0];
        for k in 0..5 {
            let x = i as i32 + dirs[k];
            let y = j as i32 + dirs[k + 1];
            if x >= 0 && x < (m as i32) && y >= 0 && y < (n as i32) {
                mat[x as usize][y as usize] ^= 1;
            }
        }
    }
    let m = mat.len();
    let n = mat[0].len();
    let mut ans = i32::MAX;
    for bin in 0..(1 << n) {
        let mut mat_copy = mat.clone();
        let mut filp_cnt = 0;
        for j in 0..n {
            if bin & (1 << j) != 0 {
                filp_cnt += 1;
                convert(&mut mat_copy, m, n, 0, j);
            }
        }

        for i in 1..m {
            for j in 0..n {
                if mat_copy[i - 1][j] == 1 {
                    filp_cnt += 1;
                    convert(&mut mat_copy, m, n, i, j);
                }
            }
        }
        let mut flag = true;
        for j in 0..n {
            if mat_copy[m - 1][j] != 0 {
                flag = false;
                break;
            }
        }
        if flag {
            ans = std::cmp::min(ans, filp_cnt);
        }
    }
    if ans != i32::MAX { ans } else { -1 }
}

/// 1286
struct CombinationIterator {
    total: Vec<String>,
}

impl CombinationIterator {
    fn new(characters: String, combination_length: i32) -> Self {
        fn dfs(
            chars: &Vec<char>,
            total: &mut Vec<String>,
            pointer: usize,
            cur: &mut Vec<char>,
            size: usize,
        ) {
            if cur.len() == size {
                total.push(cur.iter().collect::<String>());
                return;
            }
            for i in pointer..chars.len() {
                cur.push(chars[i]);
                dfs(&chars, total, i + 1, cur, size as usize);
                cur.pop();
            }
        }
        let chars = characters.chars().collect::<Vec<char>>();
        let (mut cur, mut total) = (vec![], vec![]);

        dfs(&chars, &mut total, 0, &mut cur, combination_length as usize);
        Self { total }
    }

    fn next(&mut self) -> String {
        if self.has_next() {
            self.total.remove(0)
        } else {
            "".to_string()
        }
    }

    fn has_next(&self) -> bool {
        self.total.len() > 0
    }
}

/// 1287
pub fn find_special_integer(arr: Vec<i32>) -> i32 {
    let n = arr.len();
    let mut cur = arr[0];
    let mut cnt = 0;
    for &item in arr.iter() {
        if item == cur {
            cnt += 1;
            if cnt * 4 > n {
                return cur;
            }
        } else {
            cur = item;
            cnt = 1;
        }
    }
    -1
}

/// 1288
pub fn remove_covered_intervals(intervals: Vec<Vec<i32>>) -> i32 {
    let mut vec_temp: Vec<Vec<i32>> = intervals.clone();

    if intervals.len() == 0 {
        return 0;
    }

    vec_temp.sort_by(|a, b| a[0].cmp(&b[0]).then_with(|| b[1].cmp(&a[1])));

    let mut result = 1;
    let mut right = vec_temp[0][1];

    for i in 1..vec_temp.len() {
        if vec_temp[i][1] > right {
            result += 1;
            right = vec_temp[i][1];
        }
    }

    result
}

/// 1289
pub fn min_falling_path_sum(grid: Vec<Vec<i32>>) -> i32 {
    let mut state = grid[0].clone();
    for i in 1..grid.len() {
        let (min_idx, min) = match state.iter().enumerate().min_by_key(|(_, x)| **x) {
            Some((min_idx, &min)) => (min_idx, min),
            None => unreachable!(),
        };
        let (_, sub_min) = match state
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != min_idx)
            .min_by_key(|(_, x)| **x)
        {
            Some((sub_min_idx, &sub_min)) => (sub_min_idx, sub_min),
            None => unreachable!(),
        };
        grid[i]
            .iter()
            .enumerate()
            .for_each(|(j, x)| match j == min_idx {
                false => state[j] = min + *x,
                true => state[j] = sub_min + *x,
            });
    }
    *state.iter().min().unwrap()
}

/// 1290
pub fn get_decimal_value(head: Option<Box<ListNode>>) -> i32 {
    let (mut cur, mut res) = (&head, 0);
    while let Some(node) = cur {
        res = res * 2 + node.val;
        cur = &node.next
    }
    res
}
