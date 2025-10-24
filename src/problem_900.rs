use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    rc::Rc,
};

use crate::common::{ListNode, TreeNode};

/// p801
pub fn min_swap(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let x = (1..nums1.len()).fold((1, 0), |(a, b), i| {
        if nums1[i - 1] >= nums2[i] || nums2[i - 1] >= nums1[i] {
            (a + 1, b)
        } else if nums1[i - 1] >= nums1[i] || nums2[i - 1] >= nums2[i] {
            (b + 1, a)
        } else {
            (a.min(b) + 1, a.min(b))
        }
    });
    x.0.min(x.1)
}

/// p802
pub fn eventual_safe_nodes(graph: Vec<Vec<i32>>) -> Vec<i32> {
    let n = graph.len();
    let mut out_degree = vec![0; n];
    let mut in_graph = vec![Vec::new(); n];
    let mut queue = VecDeque::new();
    let mut result = vec![];

    for (i, points) in graph.iter().enumerate() {
        points
            .iter()
            .filter(|&&next_node| next_node != i as i32)
            .for_each(|&next_node| {
                in_graph[next_node as usize].push(i);
            });
        out_degree[i] = points.len() as i32;

        if out_degree[i] == 0 {
            result.push(i as i32);
            queue.push_back(i);
        }
    }

    while let Some(node) = queue.pop_front() {
        for &prev_node in in_graph[node].iter() {
            out_degree[prev_node] -= 1;
            if out_degree[prev_node] == 0 {
                result.push(prev_node as i32);
                queue.push_back(prev_node);
            }
        }
    }

    result.sort_unstable();
    result
}

/// p803
pub struct UnionFindSize {
    parent: Vec<usize>,
    size: Vec<i32>,
}

impl UnionFindSize {
    pub fn new(n: usize) -> Self {
        UnionFindSize {
            parent: (0..=n).collect(),
            size: [1].repeat(n + 1),
        }
    }

    pub fn find(&mut self, idx: usize) -> usize {
        if idx != self.parent[idx] {
            self.parent[idx] = self.find(self.parent[idx]);
        }
        self.parent[idx]
    }

    pub fn union(&mut self, idx1: usize, idx2: usize) {
        let (x, y) = (self.find(idx1), self.find(idx2));
        if x == y {
            return;
        }
        if x == self.size.len() - 1 {
            self.parent[y] = x;
            self.size[x] += self.size[y];
        } else {
            self.parent[x] = y;
            self.size[y] += self.size[x];
        }
    }

    pub fn get_size(&self, idx: usize) -> i32 {
        self.size[idx]
    }
}

pub fn hit_bricks(grid: Vec<Vec<i32>>, hits: Vec<Vec<i32>>) -> Vec<i32> {
    let (rows, cols) = (grid.len(), grid[0].len());
    let root = rows * cols;
    let mut flashback = grid.clone();

    hits.iter()
        .for_each(|h| flashback[h[0] as usize][h[1] as usize] = 0);

    let mut uf = UnionFindSize::new(rows * cols);
    for j in 0..cols {
        if flashback[0][j] == 1 {
            uf.union(j, root);
        }
    }
    for i in 1..rows {
        for j in 0..cols {
            if flashback[i][j] == 0 {
                continue;
            }
            if flashback[i - 1][j] == 1 {
                uf.union(i * cols + j, (i - 1) * cols + j);
            }
            if j > 0 && flashback[i][j - 1] == 1 {
                uf.union(i * cols + j, i * cols + j - 1);
            }
        }
    }

    let mut counts = vec![uf.get_size(root)];
    for hit in hits.iter().rev() {
        let (i, j) = (hit[0] as usize, hit[1] as usize);
        if grid[i][j] == 0 {
            counts.push(uf.get_size(root));
            continue;
        }
        flashback[i][j] = 1;
        for (x, y) in [(i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j)] {
            if x < rows && y < cols && flashback[x][y] == 1 {
                uf.union(i * cols + j, x * cols + y);
            }
        }
        if i == 0 {
            uf.union(j, root);
        }
        counts.push(uf.get_size(root));
    }
    let n = hits.len();
    let mut ans = vec![0; n];
    for i in 0..ans.len() {
        let d = counts[n - i] - counts[n - i - 1] - 1;
        ans[i] = if d < 0 { 0 } else { d };
    }
    ans
}

/// p804
pub fn unique_morse_representations(words: Vec<String>) -> i32 {
    let morse_table = [
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
        "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--",
        "--..",
    ];
    words
        .iter()
        .fold(HashSet::new(), |mut count, word| {
            let mut s = String::new();
            word.chars()
                .for_each(|c| s.push_str(morse_table[((c as u8).abs_diff('a' as u8)) as usize]));
            count.insert(s);
            count
        })
        .len() as i32
}

/// p805
pub fn split_array_same_average(nums: Vec<i32>) -> bool {
    let (sum, n) = (nums.iter().sum::<i32>(), nums.len());
    let mut dp = vec![0; sum as usize + 1];
    dp[0] = 1;
    for num in nums {
        for s in (num..=sum).rev() {
            if dp[(s - num) as usize] > 0 {
                dp[s as usize] |= dp[(s - num) as usize] << 1;
            }
        }
    }
    for i in 1..n {
        if sum * i as i32 % n as i32 != 0 {
            continue;
        }
        let s = sum * i as i32 / n as i32;
        if dp[s as usize] > 0 && (dp[s as usize] & (1 << i as i32)) > 0 {
            return true;
        }
    }
    false
}

/// p806
pub fn number_of_lines(widths: Vec<i32>, s: String) -> Vec<i32> {
    if s.len() == 0 {
        return vec![0, 0];
    } else {
        s.chars().fold(vec![1, 0], |mut cur, c| {
            let char_width = widths[((c as u8).abs_diff('a' as u8)) as usize];
            if char_width + cur[1] > 100 {
                cur[0] = cur[0] + 1;
                cur[1] = char_width;
            } else {
                cur[1] = cur[1] + char_width;
            }
            cur
        })
    }
}

/// p807
pub fn max_increase_keeping_skyline(grid: Vec<Vec<i32>>) -> i32 {
    let (height, width) = (grid.len(), grid[0].len());
    let (mut max_by_col, mut max_by_row) = (vec![0; width], vec![0; height]);
    for r in 0..height {
        for c in 0..width {
            max_by_row[r] = max_by_row[r].max(grid[r][c]);
            max_by_col[c] = max_by_col[c].max(grid[r][c]);
        }
    }
    let mut acc = 0;
    for r in 0..height {
        for c in 0..width {
            acc = acc + max_by_row[r].min(max_by_col[c]) - grid[r][c]
        }
    }
    acc
}

/// p808
pub fn soup_servings(n: i32) -> f64 {
    if n > 4800 {
        return 1.0;
    }
    fn dfs(i: i32, j: i32) -> f64 {
        static mut F: [[f64; 200]; 200] = [[0.0; 200]; 200];

        unsafe {
            if i <= 0 && j <= 0 {
                return 0.5;
            }
            if i <= 0 {
                return 1.0;
            }
            if j <= 0 {
                return 0.0;
            }
            if F[i as usize][j as usize] > 0.0 {
                return F[i as usize][j as usize];
            }

            let ans =
                0.25 * (dfs(i - 4, j) + dfs(i - 3, j - 1) + dfs(i - 2, j - 2) + dfs(i - 1, j - 3));
            F[i as usize][j as usize] = ans;
            ans
        }
    }

    dfs((n + 24) / 25, (n + 24) / 25)
}

/// p809
pub fn expressive_words(s: String, words: Vec<String>) -> i32 {
    let expand = |s: &[u8], w: &[u8]| -> bool {
        let (mut i, mut j, m, n) = (0, 0, s.len(), w.len());
        while i < m || j < n {
            if i < m && j < n && s[i] == w[j] {
                i += 1;
                j += 1;
            } else if i > 1 && i < m && s[i] == s[i - 1] && s[i - 1] == s[i - 2]
                || i > 0 && i < m - 1 && s[i] == s[i - 1] && s[i] == s[i + 1]
            {
                i += 1;
            } else {
                return false;
            }
        }
        j == n
    };
    words
        .iter()
        .filter(|w| expand(s.as_bytes(), w.as_bytes()))
        .count() as i32
}

/// p810
pub fn xor_game(nums: Vec<i32>) -> bool {
    nums.len() % 2 == 0 || nums.iter().fold(0, |pre, cur| pre ^ cur) == 0
}

/// p811
pub fn subdomain_visits(cpdomains: Vec<String>) -> Vec<String> {
    let mut count: HashMap<String, usize> = HashMap::new();
    cpdomains.iter().for_each(|visit| {
        let mut stats = visit.split(' ');
        let time = stats.next().unwrap().parse::<usize>().unwrap();
        let domain = stats.next().unwrap();
        domain
            .split('.')
            .rev()
            .fold("".to_string(), |mut acc, cur| {
                if acc == "".to_string() {
                    acc = cur.to_string();
                } else {
                    acc = cur.to_string() + "." + &acc;
                }
                let size = count.entry(acc.clone()).or_insert(0);
                *size = *size + time;
                acc
            });
    });
    count
        .into_iter()
        .map(|(k, v)| v.to_string() + " " + &k)
        .collect::<Vec<String>>()
}

/// p812
pub fn largest_triangle_area(points: Vec<Vec<i32>>) -> f64 {
    let points: Vec<Vec<f64>> = points
        .iter()
        .map(|point| point.iter().map(|coordinate| *coordinate as f64).collect())
        .collect();
    let count = points.len();
    let mut max_area = 0_f64;
    for i in 0..count - 2 {
        for j in i..count - 1 {
            for k in j..count {
                max_area = max_area.max(
                    0.5 * (points[i][0] * points[j][1]
                        + points[j][0] * points[k][1]
                        + points[k][0] * points[i][1]
                        - points[i][0] * points[k][1]
                        - points[j][0] * points[i][1]
                        - points[k][0] * points[j][1])
                        .abs(),
                );
            }
        }
    }
    max_area
}

/// p813
pub fn largest_sum_of_averages(nums: Vec<i32>, k: i32) -> f64 {
    let n = nums.len();
    let mut pre_sum = vec![0; n + 1];

    (0..n).for_each(|i| pre_sum[i + 1] = pre_sum[i] + nums[i]);

    let mut dp = vec![0.0; n];

    (0..n).for_each(|i| dp[i] = (pre_sum[n] - pre_sum[i]) as f64 / (n - i) as f64);

    (0..k - 1).for_each(|_| {
        (0..n).for_each(|i| {
            (i + 1..n).for_each(|j| {
                let v = (pre_sum[j] - pre_sum[i]) as f64 / (j - i) as f64 + dp[j];

                if v > dp[i] {
                    dp[i] = v;
                }
            })
        })
    });

    dp[0]
}

/// p814
pub fn prune_tree(mut root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    fn dfs(root: &mut Option<Rc<RefCell<TreeNode>>>) {
        if let Some(node) = root.as_ref() {
            dfs(&mut node.borrow_mut().left);
            dfs(&mut node.borrow_mut().right);
            if node.borrow().left.is_none()
                && node.borrow().right.is_none()
                && node.borrow().val == 0
            {
                *root = None;
            }
        }
    }
    dfs(&mut root);
    root
}

/// p815
pub fn num_buses_to_destination(mut routes: Vec<Vec<i32>>, source: i32, target: i32) -> i32 {
    let mut stop_to_buses: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, route) in routes.iter().enumerate() {
        for &x in route {
            stop_to_buses.entry(x).or_default().push(i);
        }
    }

    if !stop_to_buses.contains_key(&source) || !stop_to_buses.contains_key(&target) {
        return if source != target { -1 } else { 0 };
    }

    let mut dis = HashMap::new();
    dis.insert(source, 0);
    let mut q = VecDeque::new();
    q.push_back(source);
    while let Some(x) = q.pop_front() {
        let dis_x = dis[&x];
        for &i in &stop_to_buses[&x] {
            for &y in &routes[i] {
                if !dis.contains_key(&y) {
                    dis.insert(y, dis_x + 1);
                    q.push_back(y);
                }
            }
            routes[i].clear();
        }
    }

    dis.get(&target).copied().unwrap_or(-1)
}

/// p816
pub fn ambiguous_coordinates(s: String) -> Vec<String> {
    fn solve(s: &str) -> Vec<String> {
        let (mut ret, arr) = (vec![], s.chars().collect::<Vec<_>>());
        if arr.len() == 0 || arr.len() > 1 && arr[0] == '0' && arr[arr.len() - 1] == '0' {
            return ret;
        }
        if arr.len() > 1 && arr[0] == '0' {
            ret.push(format!("0.{}", &arr[1..].iter().collect::<String>()));
            return ret;
        }
        ret.push(s.to_string());
        if arr.len() == 0 || arr[arr.len() - 1] == '0' {
            return ret;
        }
        for i in 1..arr.len() {
            ret.push(format!(
                "{}.{}",
                &arr[0..i].iter().collect::<String>(),
                &arr[i..].iter().collect::<String>()
            ));
        }
        ret
    }
    let mut ret = vec![];
    for i in 1..s.len() - 2 {
        let (left, right) = (solve(&s[1..i + 1]), solve(&s[i + 1..s.len() - 1]));
        for l in left.iter() {
            for r in right.iter() {
                ret.push(format!("({}, {})", l, r));
            }
        }
    }
    ret
}

/// p817
pub fn num_components(head: Option<Box<ListNode>>, nums: Vec<i32>) -> i32 {
    let nums: HashSet<i32> = nums.into_iter().collect();
    let mut head_ref = &head;
    let (mut is_in_component, mut component) = (false, 0);
    while let Some(node) = head_ref.as_ref() {
        if nums.contains(&node.val) {
            if is_in_component == false {
                is_in_component = true;
            }
        } else {
            if is_in_component == true {
                is_in_component = false;
                component = component + 1;
            }
        }
        head_ref = &node.next;
    }
    if is_in_component == true {
        component = component + 1;
    }
    component
}

/// p818
pub fn racecar(target: i32) -> i32 {
    let mut dp = vec![u32::MAX; target as usize + 3];
    dp[0] = 0;
    dp[1] = 1;
    dp[2] = 4;
    for t in 3..=target {
        let k = 32 - t.leading_zeros();
        if t == (1 << k) - 1 {
            dp[t as usize] = k;
            continue;
        }
        let t = t as usize;
        for j in 0..k - 1 {
            dp[t] = dp[t].min(dp[t - (1 << (k - 1)) + (1 << j)] + k - 1 + j + 2);
        }
        if (1 << k) - 1 - t < t {
            dp[t] = dp[t].min(dp[(1 << k) - 1 - t] + k + 1);
        }
    }
    dp[target as usize] as i32
}

/// p819
pub fn most_common_word(paragraph: String, banned: Vec<String>) -> String {
    use std::iter::FromIterator;
    let banned: HashSet<String> = HashSet::from_iter(banned);
    let mut count: HashMap<String, usize> = HashMap::new();
    paragraph
        .split(|c: char| c.is_ascii_punctuation() || c.is_ascii_whitespace())
        .for_each(|w| {
            let word = w.to_ascii_lowercase();
            if !word.is_empty() && !banned.contains(&word) {
                let accuracy = count.entry(word).or_insert(0);
                *accuracy = *accuracy + 1;
            }
        });
    count.iter().max_by(|a, b| a.1.cmp(b.1)).unwrap().0.clone()
}

/// p820
pub fn minimum_length_encoding(words: Vec<String>) -> i32 {
    let mut s = words.iter().map(|x| x.as_str()).collect::<HashSet<&str>>();
    for word in words.iter() {
        for k in 1..word.len() {
            s.remove(&word[k..]);
        }
    }
    let mut ans = 0;
    for w in s.iter() {
        ans += w.len() + 1;
    }
    ans as i32
}

/// p821
pub fn shortest_to_char(s: String, c: char) -> Vec<i32> {
    let mut locations: Vec<i32> = vec![];
    let chars = s.chars().collect::<Vec<char>>();
    chars.iter().enumerate().for_each(|(i, l)| {
        if *l == c {
            locations.push(i as i32);
        }
    });
    let mut distance: Vec<i32> = vec![];
    chars.iter().enumerate().for_each(|(i, l)| {
        if *l == c {
            distance.push(0)
        } else {
            distance.push(locations.iter().fold(i32::MAX, |mut distance, location| {
                distance = distance.min((location - (i as i32)).abs());
                distance
            }));
        }
    });
    distance
}

/// p822
pub fn flipgame(fronts: Vec<i32>, backs: Vec<i32>) -> i32 {
    let n = fronts.len();
    let mut set = HashSet::new();

    for i in 0..n {
        if fronts[i] == backs[i] {
            set.insert(fronts[i]);
        }
    }

    let mut min_val = i32::MAX;
    for i in 0..n {
        if fronts[i] != backs[i] {
            if !set.contains(&fronts[i]) {
                min_val = i32::min(min_val, fronts[i]);
            }
            if !set.contains(&backs[i]) {
                min_val = i32::min(min_val, backs[i]);
            }
        }
    }

    if min_val == i32::MAX { 0 } else { min_val }
}

/// p823
pub fn num_factored_binary_trees(mut arr: Vec<i32>) -> i32 {
    const MOD: i64 = 1_000_000_007;
    arr.sort_unstable();
    let mut count_map = HashMap::<i32, i64>::new();
    arr.into_iter()
        .map(|n| {
            let count: i64 = count_map
                .iter()
                .map(|(&k, &v)| {
                    if n % k == 0 && count_map.contains_key(&(n / k)) {
                        (v * count_map[&(n / k)]) % MOD
                    } else {
                        0
                    }
                })
                .fold(1, |sum, c| (sum + c) % MOD);
            count_map.insert(n, count);
            count
        })
        .fold(0, |sum, c| (sum + c) % MOD) as i32
}

/// p824
pub fn to_goat_latin(sentence: String) -> String {
    sentence
        .split_ascii_whitespace()
        .enumerate()
        .map(|(i, s)| {
            match s[..1]
                .to_lowercase()
                .starts_with(&['a', 'e', 'i', 'o', 'u'][..])
            {
                true => s.to_string() + "ma" + "a".repeat(i + 1).as_str(),
                false => s[1..].to_string() + &s[0..1] + "ma" + "a".repeat(i + 1).as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// p825
pub fn num_friend_requests(mut ages: Vec<i32>) -> i32 {
    ages.sort();
    let length = ages.len();
    let (mut ans, mut l, mut r) = (0, 0, 0);
    for &age in ages.iter() {
        if age < 15 {
            continue;
        } else {
            while ages[l] as f32 <= age as f32 * 0.5 + 7 as f32 {
                l = l + 1;
            }
            while r + 1 < length && ages[r + 1] <= age {
                r = r + 1;
            }
            ans = ans + r - l;
        }
    }
    ans as i32
}
