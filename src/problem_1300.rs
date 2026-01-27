use std::collections::VecDeque;

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
