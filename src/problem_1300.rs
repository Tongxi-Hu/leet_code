use std::collections::{HashMap, HashSet, VecDeque};

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
