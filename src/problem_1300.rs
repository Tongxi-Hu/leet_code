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
// pub fn get_maximum_gold(grid: Vec<Vec<i32>>) -> i32 {}

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
