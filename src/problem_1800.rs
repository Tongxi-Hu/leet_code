use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
};

use crate::common::ListNode;

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

/// 11
pub fn count_pairs(deliciousness: Vec<i32>) -> i32 {
    let m = 10_i32.pow(9) + 7;
    let mut hashmap = HashMap::new();
    let pows: Vec<i32> = (0..22).map(|i| 2_i32.pow(i)).collect();
    let mut answer = 0;
    for x in deliciousness {
        for i in 0..22 {
            if pows[i] >= x {
                let y = pows[i] - x;
                if let Some(&n) = hashmap.get(&y) {
                    answer += n;
                    answer = answer % m;
                }
            }
        }
        *hashmap.entry(x).or_insert(0) += 1;
    }
    answer
}

/// 12
pub fn ways_to_split(mut nums: Vec<i32>) -> i32 {
    let mut ret = 0;

    for i in 1..nums.len() {
        nums[i] += nums[i - 1];
    }

    let sum = *nums.last().unwrap();

    for i in 0..nums.len() - 2 {
        let j = match nums[i + 1..].binary_search(&(2 * nums[i] - 1)) {
            Ok(a) => a + 1,
            Err(b) => b,
        };
        let k = match nums[i + 1..].binary_search(&((sum - nums[i]) / 2 + nums[i])) {
            Ok(a) if a == nums.len() - i - 2 => a,
            Ok(a) => a + 1,
            Err(b) if b == nums.len() - i - 1 => b - 1,
            Err(b) => b,
        };

        ret = (ret + k.saturating_sub(j) as i32) % 1_000_000_007;
    }

    ret
}

/// 13
pub fn min_operations(target: Vec<i32>, arr: Vec<i32>) -> i32 {
    let mp = target
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut acc, (i, v)| {
            acc.insert(v, i);
            acc
        });

    let mut vec = vec![];

    for val in arr.iter() {
        if let Some(&i) = mp.get(val) {
            let j = vec.binary_search(&i).unwrap_or_else(|x| x);
            if j == vec.len() {
                vec.push(i);
            } else {
                vec[j] = i;
            }
        }
    }

    (target.len() - vec.len()) as i32
}

/// 16
pub fn total_money(n: i32) -> i32 {
    const D: i32 = 7;
    let w = n / D;
    let r = n % D;
    (w * D * (w + D) + r * (w * 2 + r + 1)) / 2
}

/// 17
pub fn maximum_gain(s: String, x: i32, y: i32) -> i32 {
    let (a, b, a_s, b_s) = if x > y {
        (b'a', b'b', x, y)
    } else {
        (b'b', b'a', y, x)
    };
    let mut res = 0;
    let mut s = s.as_bytes().to_vec();
    fn remove_s(s: &mut Vec<u8>, a: u8, b: u8, sc: i32) -> i32 {
        let mut i = 0;
        let mut score = 0;
        while i + 1 < s.len() {
            if s[i] == a && s[i + 1] == b {
                score += sc;
                s.remove(i);
                s.remove(i);
                i = if i == 0 { 0 } else { i - 1 };
            } else {
                i += 1;
            }
        }
        score
    }
    res += remove_s(&mut s, a, b, a_s);
    res += remove_s(&mut s, b, a, b_s);
    res
}

/// 18
pub fn construct_distanced_sequence(n: i32) -> Vec<i32> {
    fn backtrace(n: i32, idx: usize, path: &mut Vec<i32>, vis: &mut HashSet<i32>) -> bool {
        if path.len() <= idx {
            true
        } else if path[idx] != 0 {
            backtrace(n, idx + 1, path, vis)
        } else {
            for x in (1..=n).rev() {
                if vis.insert(x) {
                    if x == 1 {
                        path[idx] = x;
                        if backtrace(n, idx + 1, path, vis) {
                            return true;
                        }
                        path[idx] = 0;
                    } else if matches!(path.get(idx + (x as usize)), Some(v) if *v == 0) {
                        path[idx] = x;
                        path[idx + (x as usize)] = x;
                        if backtrace(n, idx + 1, path, vis) {
                            return true;
                        }
                        path[idx] = 0;
                        path[idx + (x as usize)] = 0;
                    }
                    vis.remove(&x);
                }
            }
            false
        }
    }

    let mut vis = HashSet::new();
    let mut path = vec![0; n as usize * 2 - 1];
    backtrace(n, 0, &mut path, &mut vis);
    path
}

/// 19
pub fn check_ways(pairs: Vec<Vec<i32>>) -> i32 {
    let mut graph: HashMap<i32, HashSet<i32>> = HashMap::new();
    let mut result = 1;
    for pair in pairs {
        (*graph.entry(pair[0]).or_insert(HashSet::new())).insert(pair[1]);
        (*graph.entry(pair[1]).or_insert(HashSet::new())).insert(pair[0]);
    }

    let mut nodes: Vec<i32> = graph.keys().map(|x| *x).collect();
    nodes.sort_by(|a, b| {
        graph
            .get(a)
            .unwrap()
            .len()
            .partial_cmp(&(graph.get(b).unwrap().len()))
            .unwrap()
    });

    let mut tree: HashMap<i32, Vec<i32>> = HashMap::new();
    let mut root = -1;
    for i in 0..nodes.len() {
        let mut p = i + 1;
        let leaf = nodes.get(i).unwrap();
        while p < nodes.len() && !graph.get(nodes.get(p).unwrap()).unwrap().contains(leaf) {
            p += 1;
        }
        if p < nodes.len() {
            (*tree.entry(*nodes.get(p).unwrap()).or_insert(Vec::new())).push(*leaf);
            if graph.get(nodes.get(p).unwrap()).unwrap().len() == graph.get(leaf).unwrap().len() {
                result = 2;
            }
        } else {
            if root == -1 {
                root = *leaf;
            } else {
                return 0;
            }
        }
    }

    fn solve_by_dfs(
        root: i32,
        depth: i32,
        result: i32,
        tree: RefCell<HashMap<i32, Vec<i32>>>,
        graph: &HashMap<i32, HashSet<i32>>,
        visited: &mut HashSet<i32>,
    ) -> (i32, i32) {
        if result == 0 {
            return (-1, 0);
        }
        if visited.contains(&root) {
            return (-1, 0);
        }

        visited.insert(root);

        let mut descendants_num = 0;
        tree.borrow_mut().entry(root).or_insert(Vec::new());
        if let Some(nodes) = tree.borrow().get(&root) {
            nodes.iter().for_each(|node| {
                descendants_num += solve_by_dfs(
                    *node,
                    depth + 1,
                    result,
                    RefCell::clone(&tree),
                    graph,
                    visited,
                )
                .0
            });
        }

        if descendants_num + depth != graph.get(&root).unwrap().len() as i32 {
            return (-1, 0);
        }

        return (descendants_num + 1, result);
    }

    solve_by_dfs(
        root,
        0,
        result,
        RefCell::new(tree),
        &graph,
        &mut HashSet::new(),
    )
    .1
}

/// 20
pub fn decode(encoded: Vec<i32>, first: i32) -> Vec<i32> {
    let (mut ret, mut p) = (Vec::new(), first);
    ret.push(first);
    for e in encoded {
        let n = e ^ p;
        ret.push(n);
        p = n;
    }
    ret
}

/// 21
pub fn swap_nodes(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let mut n = 0;
    let mut ptr = head.as_ref();
    while let Some(node) = ptr {
        n += 1;
        ptr = node.next.as_ref();
    }

    let swap1 = k.min(n + 1 - k);
    let swap2 = k.max(n + 1 - k);
    if swap1 == swap2 {
        return head;
    }

    let mut head = head;
    let mut ptr = head.as_mut();
    let mut swap1_val = 0;
    let mut swap2_val = 0;

    let mut i = 1;
    while let Some(node) = ptr {
        if i == swap1 {
            swap1_val = node.val;
        }
        if i == swap2 {
            swap2_val = node.val;
        }
        i += 1;
        ptr = node.next.as_mut();
    }

    let mut ptr = head.as_mut();
    let mut i = 1;
    while let Some(node) = ptr {
        if i == swap1 {
            node.val = swap2_val;
        }
        if i == swap2 {
            node.val = swap1_val;
        }
        i += 1;
        ptr = node.next.as_mut();
    }

    head
}

/// 22
pub fn minimum_hamming_distance(
    source: Vec<i32>,
    target: Vec<i32>,
    allowed_swaps: Vec<Vec<i32>>,
) -> i32 {
    use std::collections::HashMap;
    let n = source.len();
    let mut unfd = UnionFind::with_capacity(n);
    allowed_swaps
        .into_iter()
        .for_each(|v| unfd.union(v[0] as usize, v[1] as usize));
    let mut indice = HashMap::new();

    source
        .iter()
        .enumerate()
        .for_each(|(i, &v)| indice.entry(v).or_insert(Vec::new()).push(i));
    let mut visit = vec![true; n];
    let mut ans = 0;

    'outer: for i in 0..n {
        if let Some(other) = indice.get(&target[i]) {
            for &j in other.iter() {
                if visit[j] && unfd.is_connected(i, j) {
                    visit[j] = false;
                    continue 'outer;
                }
            }
            ans += 1;
        } else {
            ans += 1;
        }
    }
    ans
}

struct UnionFind {
    parents: Vec<usize>,
    size: Vec<usize>,
    count: usize,
}
impl UnionFind {
    fn with_capacity(n: usize) -> Self {
        Self {
            parents: (0..n).collect(),
            size: vec![1; n],
            count: n,
        }
    }
    fn find(&mut self, node: usize) -> usize {
        while self.parents[node] != self.parents[self.parents[node]] {
            self.parents[node] = self.parents[self.parents[node]];
        }
        self.parents[node]
    }
    fn union(&mut self, node_a: usize, node_b: usize) {
        let p_a = self.find(node_a);
        let p_b = self.find(node_b);
        if p_a != p_b {
            self.count -= 1;
            if self.size[p_a] >= self.size[p_b] {
                self.parents[p_b] = p_a;
                self.size[p_a] += self.size[p_b];
            } else {
                self.parents[p_a] = p_b;
                self.size[p_b] += self.size[p_a];
            }
        }
    }
    fn is_connected(&mut self, node_a: usize, node_b: usize) -> bool {
        self.find(node_a) == self.find(node_b)
    }
}

/// 23
pub fn minimum_time_required(mut jobs: Vec<i32>, k: i32) -> i32 {
    fn check(jobs: &Vec<i32>, limit: i32, k: usize) -> bool {
        let mut workloads = vec![0; k];

        return backtrack(jobs, &mut workloads, k, limit, 0);
    }

    fn backtrack(
        jobs: &Vec<i32>,
        workloads: &mut Vec<i32>,
        k: usize,
        limit: i32,
        index: usize,
    ) -> bool {
        if index >= jobs.len() {
            return true;
        }

        let job = jobs[index];

        for i in 0..k {
            if workloads[i] + job <= limit {
                workloads[i] += job;

                if backtrack(jobs, workloads, k, limit, index + 1) {
                    return true;
                }

                workloads[i] -= job;

                if workloads[i] == 0 {
                    break;
                }
            }
        }

        false
    }
    jobs.sort_by(|a, b| b.cmp(a));

    let mut l = jobs[0];
    let mut r = jobs.iter().sum();
    let k = k as usize;

    while l < r {
        let m = (l + r) >> 1;

        if check(&jobs, m, k) {
            r = m;
        } else {
            l = m + 1;
        }
    }

    l
}

/// 25
pub fn count_good_rectangles(rectangles: Vec<Vec<i32>>) -> i32 {
    rectangles
        .iter()
        .fold((0, 0), |a, c| {
            let cur = c[0].min(c[1]);
            if cur > a.0 {
                (cur, 1)
            } else if cur == a.0 {
                (a.0, a.1 + 1)
            } else {
                (a.0, a.1)
            }
        })
        .1
}

/// 26
pub fn tuple_same_product(nums: Vec<i32>) -> i32 {
    let mut cnt = HashMap::new();
    for i in 0..nums.len() {
        for j in i + 1..nums.len() {
            *cnt.entry(nums[i] * nums[j]).or_insert(0) += 1;
        }
    }
    cnt.values().into_iter().fold(0, |a, c| a + c * (c - 1) * 4)
}

/// 27
pub fn largest_submatrix(matrix: Vec<Vec<i32>>) -> i32 {
    let m = matrix.len();
    let n = matrix[0].len();
    let mut matrix = matrix;
    let mut max_area = 0;

    for i in 1..m {
        for j in 0..n {
            if matrix[i][j] == 1 {
                matrix[i][j] += matrix[i - 1][j];
            }
        }
    }

    for i in 0..m {
        matrix[i].sort_by(|a, b| b.cmp(a));
        for j in 0..n {
            let area = (j as i32 + 1) * matrix[i][j];
            if area > max_area {
                max_area = area;
            }
        }
    }

    max_area
}

/// 28
pub fn can_mouse_win(grid: Vec<String>, cat_jump: i32, mouse_jump: i32) -> bool {
    static MOUSE_TURN: usize = 0;
    static CAT_TURN: usize = 1;

    fn build_graph(jump: i32, grid_arr: &Vec<Vec<char>>) -> Vec<Vec<i32>> {
        let (m, n) = (grid_arr.len(), grid_arr[0].len());
        let mut graph = vec![Vec::new(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut list = Vec::new();
                if grid_arr[i][j] == '#' {
                    continue;
                }
                list.push((i * n + j) as i32);
                for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]] {
                    for k in 1..=jump {
                        let (x, y) = (i as i32 + dir[0] * k, j as i32 + dir[1] * k);
                        if x < 0
                            || x as usize >= m
                            || y < 0
                            || y as usize >= n
                            || grid_arr[x as usize][y as usize] == '#'
                        {
                            break;
                        }
                        list.push(x * n as i32 + y);
                    }
                }
                graph[i * n + j] = list;
            }
        }
        graph
    }

    fn dfs<'a>(
        p1: usize,
        p2: usize,
        food_pos: i32,
        mut turn: usize,
        memo: &'a RefCell<Vec<Vec<Vec<i32>>>>,
        graph: &Vec<Vec<Vec<i32>>>,
    ) -> &'a RefCell<Vec<Vec<Vec<i32>>>> {
        if p1 == p2 {
            return &memo;
        }
        let dst = if turn == 0 { p2 } else { p1 };
        if dst == food_pos as usize {
            return &memo;
        }
        if memo.borrow()[p1][p2][turn as usize] < 0 {
            return &memo;
        }
        memo.borrow_mut()[p1][p2][turn as usize] = -1;
        turn ^= 1;
        for w in &graph[turn as usize][p2] {
            if turn == MOUSE_TURN {
                dfs(*w as usize, p1, food_pos, turn, memo, graph);
            } else {
                memo.borrow_mut()[*w as usize][p1][turn as usize] += 1;
                if memo.borrow()[*w as usize][p1][turn as usize] as usize
                    == graph[turn as usize][*w as usize].len()
                {
                    dfs(*w as usize, p1, food_pos, turn, memo, graph);
                }
            }
        }
        &memo
    }

    let grid_arr = grid
        .iter()
        .map(|g| g.chars().collect::<Vec<char>>())
        .collect::<Vec<_>>();
    let (m, n) = (grid.len(), grid[0].len());
    let (mut mouse_pos, mut cat_pos, mut food_pos) = (0, 0, 0);

    for i in 0..m {
        for j in 0..n {
            match grid_arr[i][j] {
                'F' => food_pos = i * n + j,
                'C' => cat_pos = i * n + j,
                'M' => mouse_pos = i * n + j,
                _ => {}
            }
        }
    }
    let mut graph: Vec<Vec<Vec<i32>>> = vec![Vec::new(); 2];
    graph[0] = build_graph(mouse_jump, &grid_arr);
    graph[1] = build_graph(cat_jump, &grid_arr);
    let memo = RefCell::new(vec![vec![vec![0; 2]; m * n]; m * n]);
    for i in 0..m {
        for j in 0..n {
            let ch = grid_arr[i][j];
            if ch == '#' || ch == 'F' {
                continue;
            }
            let ret = dfs(
                i * n + j,
                food_pos,
                food_pos as i32,
                CAT_TURN as usize,
                &memo,
                &graph,
            )
            .borrow()[mouse_pos][cat_pos][MOUSE_TURN]
                < 0;
            if ret {
                return true;
            }
        }
    }
    false
}

/// 32
pub fn largest_altitude(gain: Vec<i32>) -> i32 {
    gain.iter()
        .fold((0, 0), |a, c| (a.0 + c, a.1.max(a.0 + c)))
        .1
}

/// 33
pub fn minimum_teachings(_: i32, languages: Vec<Vec<i32>>, friendships: Vec<Vec<i32>>) -> i32 {
    let mut cncon = HashSet::new();
    for friendship in friendships {
        let mut mp = HashSet::new();
        let mut conm = false;
        for &lan in &languages[friendship[0] as usize - 1] {
            mp.insert(lan);
        }
        for &lan in &languages[friendship[1] as usize - 1] {
            if mp.contains(&lan) {
                conm = true;
                break;
            }
        }

        if !conm {
            cncon.insert(friendship[0] - 1);
            cncon.insert(friendship[1] - 1);
        }
    }

    let mut max_cnt = 0;
    let mut cnt = HashMap::new();
    for &person in &cncon {
        for &lan in &languages[person as usize] {
            *cnt.entry(lan).or_insert(0) += 1;
            max_cnt = max_cnt.max(*cnt.get(&lan).unwrap());
        }
    }

    cncon.len() as i32 - max_cnt
}

/// 34
pub fn decode_ii(mut encoded: Vec<i32>) -> Vec<i32> {
    let n = encoded.len() + 1;
    let mut s = encoded[1..]
        .chunks(2)
        .fold((2..=n as i32).fold(1, |s, x| s ^ x), |s, x| s ^ x[0]);
    encoded.push(0);
    encoded.iter_mut().for_each(|x| {
        let t = *x;
        *x = s;
        s ^= t
    });
    encoded
}

/// 35
pub fn ways_to_fill_array(queries: Vec<Vec<i32>>) -> Vec<i32> {
    const MOD: i64 = 1_000_000_007;
    const MAXN: usize = 10_014;
    const MAXM: usize = 14;

    let mut comb = vec![vec![0; MAXM]; MAXN];
    let mut ans = Vec::new();

    comb[0][0] = 1;
    for i in 1..MAXN {
        comb[i][0] = 1;
        for j in 1..=i.min(MAXM - 1) {
            comb[i][j] = (comb[i - 1][j - 1] + comb[i - 1][j]) % MOD;
        }
    }

    for q in queries {
        let n = q[0] as usize;
        let mut k = q[1] as i64;
        let mut tot = 1;

        let mut i = 2;
        while i * i <= k {
            if k % i == 0 {
                let mut cnt = 0;
                while k % i == 0 {
                    k /= i;
                    cnt += 1;
                }
                tot = (tot * comb[n + cnt - 1][cnt]) % MOD;
            }
            i += 1;
        }
        if k > 1 {
            tot = (tot * n as i64) % MOD;
        }
        ans.push(tot as i32);
    }
    ans
}

/// 36
pub fn maximum_time(time: String) -> String {
    let times = time.as_bytes();
    let mut ans = String::new();

    if times[0] == b'?' {
        ans.push(if b'4' <= times[1] && times[1] <= b'9' {
            '1'
        } else {
            '2'
        });
    } else {
        ans.push(times[0] as char);
    }

    if times[1] == b'?' {
        ans.push(if times[0] == b'2' || times[0] == b'?' {
            '3'
        } else {
            '9'
        });
    } else {
        ans.push(times[1] as char);
    }

    ans.push(':');

    if times[3] == b'?' {
        ans.push('5');
    } else {
        ans.push(times[3] as char);
    }

    if times[4] == b'?' {
        ans.push('9');
    } else {
        ans.push(times[4] as char);
    }

    ans
}

/// 37
pub fn min_characters(a: String, b: String) -> i32 {
    let mut ac = vec![0; 26];
    for c in a.bytes() {
        let c = c as usize - b'a' as usize;
        ac[c] += 1;
    }
    let mut bc = vec![0; 26];
    for c in b.bytes() {
        let c = c as usize - b'a' as usize;
        bc[c] += 1;
    }
    let mut ans = a.len() + b.len() - (0..ac.len()).map(|i| ac[i] + bc[i]).max().unwrap();
    let (sa, sb) = (a.len(), b.len());
    let mut a = 0;
    let mut b = 0;
    for i in 0..25 {
        a += ac[i];
        b += bc[i];
        ans = ans.min(a + sb - b).min(b + sa - a);
    }

    ans as _
}

/// 38
pub fn kth_largest_value(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
    let (height, width) = (matrix.len(), matrix[0].len());
    let (mut prefix, mut all) = (vec![vec![0; width + 1]; height + 1], vec![]);
    for r in 0..height {
        for c in 0..width {
            prefix[r + 1][c + 1] =
                prefix[r][c + 1] ^ prefix[r + 1][c] ^ prefix[r][c] ^ matrix[r][c];
            all.push(prefix[r + 1][c + 1]);
        }
    }
    all.sort();
    all[all.len() - k as usize]
}

/// 39
pub fn minimum_boxes(mut n: i32) -> i32 {
    let mut base = (6_f64 * n as f64).cbrt() as i64;
    if base * (base + 1) * (base + 2) / 6_i64 > n as i64 {
        base -= 1;
    }
    let block_base = base * (base + 1) / 2;
    n -= (base * (base + 1) * (base + 2) / 6) as i32;
    let extra = (2_f64 * n as f64).sqrt() as i64;
    if extra * (extra + 1) / 2 >= n as i64 {
        (block_base + extra) as i32
    } else {
        (block_base + extra) as i32 + 1
    }
}

/// 42
pub fn count_balls(low_limit: i32, high_limit: i32) -> i32 {
    let mut cnt = HashMap::new();
    for mut i in low_limit..=high_limit {
        let mut key = 0;
        while i > 0 {
            key += i % 10;
            i = i / 10;
        }
        *cnt.entry(key).or_insert(0) += 1;
    }
    *cnt.values().into_iter().max().unwrap()
}
