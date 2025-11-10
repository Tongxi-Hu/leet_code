use std::{
    cell::RefCell,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque},
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

/// p826
pub fn max_profit_assignment(difficulty: Vec<i32>, profit: Vec<i32>, mut worker: Vec<i32>) -> i32 {
    let mut jobs = difficulty
        .into_iter()
        .zip(profit)
        .collect::<Vec<(i32, i32)>>();
    jobs.sort();
    worker.sort();
    let mut gain = vec![];
    worker.iter().for_each(|ability| {
        let mut profit = 0;
        for job in jobs.iter() {
            if job.0 > *ability {
                break;
            } else {
                profit = profit.max(job.1)
            }
        }
        gain.push(profit);
    });
    gain.iter().fold(0, |acc, cur| acc + cur)
}

/// p827
pub fn largest_island(mut grid: Vec<Vec<i32>>) -> i32 {
    fn dfs(grid: &mut Vec<Vec<i32>>, i: usize, j: usize, id: i32) -> i32 {
        grid[i][j] = id;
        let mut size = 1;
        for (x, y) in [
            (i.saturating_sub(1), j),
            (i + 1, j),
            (i, j.saturating_sub(1)),
            (i, j + 1),
        ] {
            if x < grid.len() && y < grid[0].len() && grid[x][y] == 1 {
                size += dfs(grid, x, y, id);
            }
        }
        size
    }

    let n = grid.len();
    let mut area = vec![];
    for i in 0..n {
        for j in 0..n {
            if grid[i][j] == 1 {
                area.push(dfs(&mut grid, i, j, area.len() as i32 + 2));
            }
        }
    }

    if area.is_empty() {
        return 1;
    }

    let mut ans = 0;
    let mut s = HashSet::new();
    for (i, row) in grid.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            if x != 0 {
                continue;
            }
            s.clear();
            let mut new_area = 1;
            for (x, y) in [
                (i.saturating_sub(1), j),
                (i + 1, j),
                (i, j.saturating_sub(1)),
                (i, j + 1),
            ] {
                if x < n && y < n && grid[x][y] != 0 && s.insert(grid[x][y]) {
                    new_area += area[(grid[x][y] - 2) as usize];
                }
            }
            ans = ans.max(new_area);
        }
    }

    if ans == 0 { (n * n) as _ } else { ans }
}

/// p828
pub fn unique_letter_string(s: String) -> i32 {
    let mut d: Vec<Vec<i32>> = vec![vec![-1; 1]; 26];
    for (i, c) in s.chars().enumerate() {
        d[(c as usize) - ('A' as usize)].push(i as i32);
    }
    let mut ans = 0;
    for v in d.iter_mut() {
        v.push(s.len() as i32);
        for i in 1..v.len() - 1 {
            ans += (v[i] - v[i - 1]) * (v[i + 1] - v[i]);
        }
    }
    ans as i32
}

/// p829
pub fn consecutive_numbers_sum(mut n: i32) -> i32 {
    let (mut cnt, k) = (0, ((2 * n) as f64).sqrt() as i32);
    for i in 1..=k {
        n -= i;
        cnt += if n % i == 0 { 1 } else { 0 }
    }
    cnt
}

/// p830
pub fn large_group_positions(s: String) -> Vec<Vec<i32>> {
    let length = s.len();
    s.chars()
        .into_iter()
        .enumerate()
        .fold((vec![], ' ', 1, -1), |mut acc, (i, c)| {
            if c == acc.1 {
                acc.2 = acc.2 + 1;
                if i == length - 1 && acc.2 >= 3 {
                    acc.0.push(vec![acc.3, acc.3 + acc.2 - 1]);
                }
            } else {
                if acc.2 >= 3 {
                    acc.0.push(vec![acc.3, acc.3 + acc.2 - 1]);
                }
                acc.1 = c;
                acc.2 = 1;
                acc.3 = i as i32;
            }
            return acc;
        })
        .0
}

/// p831
pub fn mask_pii(s: String) -> String {
    const COUNTRY_CODE: [&str; 4] = ["", "+*-", "+**-", "+***-"];
    match s.find('@') {
        Some(idx) => s[0..1].to_lowercase() + "*****" + &s[(idx - 1)..].to_lowercase(),
        None => {
            let numbers = s.matches(char::is_numeric).collect::<String>();
            let n = numbers.len();
            COUNTRY_CODE[n - 10].to_string() + "***-***-" + &numbers[(n - 4)..]
        }
    }
}

/// p832
pub fn flip_and_invert_image(mut image: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    image.iter_mut().for_each(|row| {
        row.reverse();
        row.iter_mut()
            .for_each(|pixel| *pixel = if *pixel == 1 { 0 } else { 1 });
    });
    image
}

/// p833
pub fn find_replace_string(
    s: String,
    indices: Vec<i32>,
    sources: Vec<String>,
    targets: Vec<String>,
) -> String {
    let n = s.len();
    let mut records: BTreeMap<usize, usize> = BTreeMap::new();
    for (i, idx) in indices.iter().map(|&idx| idx as usize).enumerate() {
        let len = sources[i].len();
        if idx + len <= n && &s[idx..idx + len] == sources[i].as_str() {
            records.insert(idx, i);
        }
    }
    let mut replace = String::new();
    let mut expect_idx = 0;
    for (idx, i) in records.into_iter() {
        if expect_idx != idx {
            replace.push_str(&s[expect_idx..idx]);
        }
        replace.push_str(targets[i].as_str());
        expect_idx = idx + sources[i].len();
    }
    if expect_idx != n {
        replace.push_str(&s[expect_idx..]);
    }
    replace
}

/// p834
pub fn sum_of_distances_in_tree(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
    let n = n as usize;

    let mut graph: Vec<Vec<usize>> = vec![vec![]; n];
    for edge in edges.iter() {
        let (node1, node2) = (edge[0] as usize, edge[1] as usize);
        graph[node1].push(node2);
        graph[node2].push(node1);
    }

    fn re_root(
        graph: &Vec<Vec<usize>>,
        result: &mut Vec<i32>,
        counts: &Vec<i32>,
        cur_node: usize,
        parent: usize,
    ) {
        let n = graph.len() as i32;
        for &next_node in graph[cur_node].iter() {
            if next_node != parent {
                result[next_node] = result[cur_node] + n - 2 * counts[next_node];
                re_root(graph, result, counts, next_node, cur_node);
            }
        }
    }

    fn dfs(
        graph: &Vec<Vec<usize>>,
        result: &mut Vec<i32>,
        counts: &mut Vec<i32>,
        cur_node: usize,
        parent: usize,
        depth: i32,
    ) {
        result[0] += depth; // depth是当前节点cur_node到0节点的距离，所以在这个过程中，可以计算出来result[0]

        for &next_node in graph[cur_node].iter() {
            if next_node != parent {
                dfs(graph, result, counts, next_node, cur_node, depth + 1);
                counts[cur_node] += counts[next_node];
            }
        }
    }

    let mut counts: Vec<i32> = vec![1; n];
    let mut result: Vec<i32> = vec![0; n];
    dfs(&graph, &mut result, &mut counts, 0, n, 0);

    re_root(&graph, &mut result, &counts, 0, n);

    result
}

/// p835
pub fn largest_overlap(img1: Vec<Vec<i32>>, img2: Vec<Vec<i32>>) -> i32 {
    let n = img1.len() as i32;
    let mut ans = 0;

    for di in -n + 1..=n - 1 {
        for dj in -n + 1..=n - 1 {
            let mut count = 0;

            for i in 0.max(-di)..n.min(n - di) {
                for j in 0.max(-dj)..n.min(n - dj) {
                    count +=
                        img1[i as usize][j as usize] * img2[(i + di) as usize][(j + dj) as usize];
                }
            }

            ans = ans.max(count);
        }
    }

    ans
}

/// p836
pub fn is_rectangle_overlap(rec1: Vec<i32>, rec2: Vec<i32>) -> bool {
    let (x1, y1, x2, y2) = (rec1[0], rec1[1], rec1[2], rec1[3]);
    let (x3, y3, x4, y4) = (rec2[0], rec2[1], rec2[2], rec2[3]);
    !(x3 >= x2 || x4 <= x1 || y3 >= y2 || y4 <= y1)
}

/// p837
pub fn new21_game(n: i32, k: i32, max_pts: i32) -> f64 {
    let n = n as usize;
    let k = k as usize;
    let max_pts = max_pts as usize;
    let mut f = vec![0.0; n + 1];
    let mut s = 0.0;
    for i in (0..=n).rev() {
        f[i] = if i >= k { 1.0 } else { s / max_pts as f64 };
        s += f[i];
        if i + max_pts <= n {
            s -= f[i + max_pts];
        }
    }
    f[0]
}

/// p838
pub fn push_dominoes(dominoes: String) -> String {
    let mut arr: Vec<char> = dominoes.chars().collect();
    let (mut l, mut r) = (-1, -1);
    (0..=dominoes.len()).for_each(|i| {
        if i == dominoes.len() || arr[i] == 'R' {
            if r > l {
                while r < i as i32 {
                    arr[r as usize] = 'R';
                    r += 1;
                }
            }
            r = i as i32;
        } else if arr[i] == 'L' {
            if l > r || r == -1 {
                l += 1;
                while l < i as i32 {
                    arr[l as usize] = 'L';
                    l += 1;
                }
            } else {
                l = i as i32;
                let (mut lo, mut hi) = (r + 1, l - 1);
                while lo < hi {
                    arr[lo as usize] = 'R';
                    arr[hi as usize] = 'L';
                    lo += 1;
                    hi -= 1;
                }
            }
        }
    });
    arr.iter().collect::<String>()
}

/// p839
pub fn num_similar_groups(strs: Vec<String>) -> i32 {
    let n = strs.len();
    let mut parents = (0..n).collect::<Vec<_>>();

    fn is_similar(s1: &[u8], s2: &[u8]) -> bool {
        let (mut idx1, mut idx2) = (None, None);

        for (i, (&b1, &b2)) in s1.iter().zip(s2.iter()).enumerate() {
            if b1 == b2 {
                continue;
            }

            match (idx1, idx2) {
                (None, None) => {
                    idx1 = Some(i);
                }
                (Some(id), None) if s1[id] == b2 && s2[id] == b1 => {
                    idx2 = Some(i);
                }
                _ => return false,
            }
        }

        !(idx1.is_some() && idx2.is_none())
    }
    fn parent(parents: &mut Vec<usize>, mut v: usize) -> usize {
        while v != parents[v] {
            parents[v] = parents[parents[v]];
            v = parents[v];
        }
        v
    }

    let mut count = n as i32;
    for i in 1..n {
        for j in 0..i {
            if is_similar(strs[i].as_bytes(), strs[j].as_bytes()) {
                let p = parent(&mut parents, j);
                if i != p {
                    parents[p] = i;
                    count -= 1;
                }
            }
        }
    }

    count
}

/// p840
pub fn num_magic_squares_inside(grid: Vec<Vec<i32>>) -> i32 {
    let mut count = 0;
    let (rows, cols) = (grid.len(), grid[0].len());
    if rows < 3 || cols < 3 {
        return count;
    }
    for r in 0..rows - 2 {
        'inner: for c in 0..cols - 2 {
            let mut record = HashSet::new();
            for i in 0..3 {
                for j in 0..3 {
                    if grid[r + i][c + j] >= 10 {
                        continue 'inner;
                    }
                    record.insert(grid[r + i][c + j]);
                }
            }
            if record.len() != 9
                || grid[r + 1][c + 1] != 5
                || grid[r][c] + grid[r][c + 1] + grid[r][c + 2] != 15
                || grid[r + 1][c] + grid[r + 1][c + 1] + grid[r + 1][c + 2] != 15
                || grid[r + 1][c] + grid[r + 1][c + 1] + grid[r + 1][c + 2] != 15
                || grid[r][c] + grid[r + 1][c] + grid[r + 2][c] != 15
                || grid[r][c + 1] + grid[r + 1][c + 1] + grid[r + 2][c + 1] != 15
                || grid[r][c + 2] + grid[r + 1][c + 2] + grid[r + 2][c + 2] != 15
                || grid[r][c] + grid[r + 1][c + 1] + grid[r + 2][c + 2] != 15
                || grid[r + 2][c] + grid[r + 1][c + 1] + grid[r][c + 2] != 15
            {
                continue 'inner;
            }
            count = count + 1;
        }
    }
    return count;
}

/// p841
pub fn can_visit_all_rooms(rooms: Vec<Vec<i32>>) -> bool {
    let mut keys: HashSet<i32> = HashSet::new();
    keys.insert(0);
    let count = rooms.len();
    let mut visited = vec![false; count];
    rooms[0].iter().for_each(|key| {
        keys.insert(*key);
    });
    visited[0] = true;
    let mut can_visit = visited
        .iter()
        .enumerate()
        .filter(|(i, v)| (**v == false) && keys.contains(&(*i as i32)))
        .map(|(i, _)| i)
        .collect::<Vec<usize>>();

    while can_visit.len() > 0 {
        can_visit.iter().for_each(|i| {
            visited[*i] = true;
            rooms[*i].iter().for_each(|k| {
                keys.insert(*k);
            });
        });
        can_visit = visited
            .iter()
            .enumerate()
            .filter(|(i, v)| (**v == false) && keys.contains(&(*i as i32)))
            .map(|(i, _)| i)
            .collect::<Vec<usize>>();
    }

    visited.iter().all(|v| *v == true)
}

/// p842
pub fn split_into_fibonacci(num: String) -> Vec<i32> {
    let n = num.len();
    if n < 3 {
        return vec![];
    }
    let mut result = Vec::new();

    fn check(num: &str, idx1: usize, mut idx2: usize, result: &mut Vec<i32>) -> bool {
        let n = num.len();

        let mut val1 = match num[0..idx1].parse::<i32>() {
            Ok(val) if val.to_string().len() == idx1 => val,
            _ => return false,
        };
        let mut val2 = match num[idx1..idx2].parse::<i32>() {
            Ok(val) if val.to_string().len() == idx2 - idx1 => val,
            _ => return false,
        };
        result.push(val1);
        result.push(val2);

        while idx2 < n {
            let val3 = match val1.checked_add(val2) {
                Some(val) => val,
                None => return false,
            };

            let str = val3.to_string();
            let len = str.len();
            if idx2 + len <= n && &num[idx2..idx2 + len] == &str {
                idx2 += len;
                val1 = val2;
                val2 = val3;
                result.push(val3);
            } else {
                return false;
            }
        }

        true
    }

    for i in 1..n - 2 {
        for j in i + 1..n - 1 {
            if check(&num, i, j, &mut result) {
                return result;
            }
            result.clear();
        }
    }

    vec![]
}

/// p844
pub fn backspace_compare(s: String, t: String) -> bool {
    let s_stack = s.chars().fold(vec![], |mut acc, cur| {
        if cur == '#' {
            if acc.len() > 0 {
                acc.pop();
            }
        } else {
            acc.push(cur);
        };
        acc
    });
    let t_stack = t.chars().fold(vec![], |mut acc, cur| {
        if cur == '#' {
            if acc.len() > 0 {
                acc.pop();
            }
        } else {
            acc.push(cur);
        };
        acc
    });
    s_stack == t_stack
}

/// p845
pub fn longest_mountain(arr: Vec<i32>) -> i32 {
    let length = arr.len();
    let mut left = vec![0; length];
    let mut right = vec![0; length];
    for i in 1..length {
        if arr[i] > arr[i - 1] {
            left[i] = left[i - 1] + 1;
        }
    }
    for i in (0..length - 1).rev() {
        if arr[i] > arr[i + 1] {
            right[i] = right[i + 1] + 1;
        }
    }
    arr.iter().enumerate().fold(0, |mut acc, (i, _)| {
        if left[i] > 0 && right[i] > 0 {
            acc = acc.max(left[i] + right[i] + 1)
        }
        acc
    })
}

/// p846
pub fn is_n_straight_hand(mut hand: Vec<i32>, group_size: i32) -> bool {
    let n = hand.len();
    if n as i32 % group_size != 0 {
        return false;
    }

    let mut cache = HashMap::new();
    hand.sort_unstable();

    hand.iter().for_each(|&h| *cache.entry(h).or_insert(0) += 1);
    for &h in hand.iter() {
        if cache.get(&h) == None {
            continue;
        }
        for i in h..h + group_size {
            if cache.get(&i) == None {
                return false;
            }
            let val = *cache.get(&i).unwrap();
            cache.insert(i, val - 1);
            if *cache.get(&i).unwrap() == 0 {
                cache.remove(&i);
            }
        }
    }
    true
}

/// p847
pub fn shortest_path_length(graph: Vec<Vec<i32>>) -> i32 {
    let n = graph.len();
    let mut visited = vec![vec![false; 1 << n]; n];
    let mut q = std::collections::VecDeque::new();
    for i in 0..n {
        q.push_back((i, 1u16 << i, 0));
        visited[i][1usize << i] = true;
    }
    while let Some((u, mask, dist)) = q.pop_front() {
        if mask == (1 << n) - 1 {
            return dist;
        }
        for &v in &graph[u] {
            let new_mask = mask | (1 << v);
            let v = v as usize;
            if visited[v][new_mask as usize] {
                continue;
            }
            visited[v][new_mask as usize] = true;
            q.push_back((v, new_mask, dist + 1));
        }
    }
    0
}

/// p848
pub fn shifting_letters(s: String, mut shifts: Vec<i32>) -> String {
    let n = s.len();
    for i in (0..n).rev() {
        if i + 1 < n {
            shifts[i] = (shifts[i] + shifts[i + 1]) % 26;
        } else {
            shifts[i] %= 26;
        }
    }
    let mut shift_s: Vec<u8> = vec![0; n];
    for (i, b) in s.bytes().enumerate() {
        shift_s[i] = b'a' + (b - b'a' + shifts[i] as u8) % 26;
    }
    String::from_utf8(shift_s).unwrap()
}

/// p849
pub fn max_dist_to_closest(seats: Vec<i32>) -> i32 {
    let mut ans = 0;
    let (mut l, mut r) = (0, seats.len() - 1);

    while seats[l] == 0 {
        l += 1;
    }
    ans = ans.max(l);

    while seats[r] == 0 {
        r -= 1;
    }
    ans = ans.max(seats.len() - r - 1);

    let mut tmp = l;
    while l < r {
        l += 1;
        while seats[l] == 0 {
            l += 1;
        }
        ans = ans.max((l - tmp) / 2);
        tmp = l;
    }
    ans as i32
}

/// p850
pub fn rectangle_area(rectangles: Vec<Vec<i32>>) -> i32 {
    const MOD: i64 = 1E9 as i64 + 7;
    let mut all_recs = vec![];
    for (i, rec) in rectangles.iter().enumerate() {
        let mut res = vec![rec.clone()];
        for j in i + 1..rectangles.len() {
            let mut step = vec![];
            for item in res {
                if rectangles[j][0] >= item[2]
                    || rectangles[j][2] <= item[0]
                    || rectangles[j][1] >= item[3]
                    || rectangles[j][3] <= item[1]
                {
                    step.push(vec![item[0], item[1], item[2], item[3]]);
                } else {
                    if item[0] < rectangles[j][0] {
                        step.push(vec![item[0], item[1], rectangles[j][0], item[3]]);
                    }
                    if item[2] > rectangles[j][2] {
                        step.push(vec![rectangles[j][2], item[1], item[2], item[3]]);
                    }
                    if item[1] < rectangles[j][1] {
                        step.push(vec![
                            item[0].max(rectangles[j][0]),
                            item[1],
                            item[2].min(rectangles[j][2]),
                            rectangles[j][1],
                        ]);
                    }
                    if item[3] > rectangles[j][3] {
                        step.push(vec![
                            item[0].max(rectangles[j][0]),
                            rectangles[j][3],
                            item[2].min(rectangles[j][2]),
                            item[3],
                        ]);
                    }
                }
            }
            res = step;
        }
        all_recs.append(&mut res);
    }
    all_recs.iter().fold(0, |total, item| {
        (total + (item[3] - item[1]) as i64 * (item[2] - item[0]) as i64) % MOD
    }) as i32
}

/// p851
pub fn loud_and_rich(richer: Vec<Vec<i32>>, quiet: Vec<i32>) -> Vec<i32> {
    let n = quiet.len();

    let mut indegree: Vec<i32> = vec![0; n];
    let mut graph: Vec<Vec<usize>> = vec![vec![]; n];
    for v in richer.iter() {
        let (a, b) = (v[0] as usize, v[1] as usize);
        indegree[b] += 1;
        graph[a].push(b);
    }

    let mut result = (0..n as i32).collect::<Vec<_>>();
    let mut q = VecDeque::new();
    for (i, &count) in indegree.iter().enumerate() {
        if count == 0 {
            q.push_back(i);
            result[i] = i as i32;
        }
    }

    while let Some(cur) = q.pop_front() {
        for &next in graph[cur].iter() {
            if quiet[result[cur] as usize] < quiet[result[next] as usize] {
                result[next] = result[cur];
            }
            indegree[next] -= 1;
            if indegree[next] == 0 {
                q.push_back(next);
            }
        }
    }

    result
}

/// p852
pub fn peak_index_in_mountain_array(arr: Vec<i32>) -> i32 {
    let (mut left, mut right) = (1, arr.len() - 2);
    while left <= right {
        let mid = (left + right) / 2;
        if arr[mid] > arr[mid - 1] && arr[mid] > arr[mid + 1] {
            return mid as i32;
        } else if arr[mid] > arr[mid + 1] {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    0
}

/// p853
pub fn car_fleet(target: i32, position: Vec<i32>, speed: Vec<i32>) -> i32 {
    let mut sort_by_position = position
        .iter()
        .zip(speed.iter())
        .collect::<Vec<(&i32, &i32)>>();
    sort_by_position.sort_by(|a, b| b.0.cmp(a.0));
    let mut fleet: Vec<f32> =
        vec![(target as f32 - *sort_by_position[0].0 as f32) / *sort_by_position[0].1 as f32];
    sort_by_position.iter().for_each(|car| {
        let time = (target as f32 - *car.0 as f32) / *car.1 as f32;
        if time > *fleet.last().unwrap() {
            fleet.push(time)
        }
    });
    fleet.len() as i32
}

/// p854
pub fn k_similarity(s1: String, s2: String) -> i32 {
    fn swap(chs: &mut Vec<char>, i: usize, j: usize) {
        let tmp = chs[i];
        chs[i] = chs[j];
        chs[j] = tmp;
    }
    fn get_next_list(curr: String, chs_target: &Vec<char>) -> Vec<String> {
        let mut chs_curr = curr.chars().collect::<Vec<char>>();
        let mut next_list: Vec<String> = vec![];
        let mut i = 0;
        let n = chs_curr.len();
        while i < n {
            if chs_curr[i] != chs_target[i] {
                break;
            }
            i += 1;
        }
        for j in i + 1..n {
            if chs_curr[j] == chs_target[i] && chs_curr[j] != chs_target[j] {
                swap(&mut chs_curr, i, j);
                let s_new: String = chs_curr.iter().collect();
                next_list.push(s_new);
                swap(&mut chs_curr, i, j);
            }
        }
        next_list
    }

    if s1.eq(&s2) {
        return 0;
    }
    let mut queue: VecDeque<String> = VecDeque::new();
    let mut visited: HashSet<String> = HashSet::new();
    queue.push_back(s1.clone());
    visited.insert(s1.clone());
    let mut step = 0;
    let chs_target = s2.chars().collect::<Vec<char>>();
    while !queue.is_empty() {
        let size = queue.len();
        for _ in 0..size {
            let curr = queue.pop_front().unwrap();
            let next_list = get_next_list(curr, &chs_target);
            for i in 0..next_list.len() {
                let next_word = next_list[i].clone();
                if next_word.eq(&s2) {
                    return step + 1;
                }
                if !visited.contains(&next_word) {
                    visited.insert(next_word.clone());
                    queue.push_back(next_word.clone());
                }
            }
        }
        step += 1;
    }
    step
}

/// p855
struct ExamRoom {
    n: i32,
    list: Vec<i32>,
}
impl ExamRoom {
    fn new(n: i32) -> Self {
        ExamRoom {
            n,
            list: Vec::new(),
        }
    }

    fn seat(&mut self) -> i32 {
        if self.list.is_empty() {
            self.list.push(0);
            return 0;
        }
        let mut distance = self.list[0].max(self.n - 1 - self.list[self.list.len() - 1]);
        for i in 0..self.list.len() - 1 {
            distance = distance.max((self.list[i + 1] - self.list[i]) / 2);
        }
        if self.list[0] == distance {
            self.list.insert(0, 0);
            return 0;
        }
        for i in 0..self.list.len() - 1 {
            if (self.list[i + 1] - self.list[i]) / 2 == distance {
                self.list
                    .insert(i + 1, (self.list[i + 1] + self.list[i]) / 2);
                return self.list[i + 1];
            }
        }
        self.list.push(self.n - 1);
        self.n - 1
    }

    fn leave(&mut self, p: i32) {
        for i in 0..self.list.len() {
            if let Some(&curr) = self.list.get(i) {
                if curr == p {
                    self.list.remove(i);
                }
            }
        }
    }
}

/// p856
pub fn score_of_parentheses(s: String) -> i32 {
    let (mut depth, mut score) = (0, 0);
    let chars = s.chars().collect::<Vec<char>>();
    chars.iter().enumerate().for_each(|(i, c)| match c {
        '(' => {
            depth = depth + 1;
        }
        ')' => {
            depth = depth - 1;
            if chars[i - 1] == '(' {
                score = score + (2 as f32).powi(depth) as i32;
            };
        }
        _ => (),
    });
    score
}

/// p857
pub fn mincost_to_hire_workers(quality: Vec<i32>, wage: Vec<i32>, k: i32) -> f64 {
    let (_, mut curr, mut ans) = (quality.len(), 0f64, f64::INFINITY);
    let (mut ratio, mut heap) = (
        wage.into_iter()
            .zip(quality.into_iter())
            .into_iter()
            .map(|(w, q)| (w as f64 / q as f64, q as f64))
            .collect::<Vec<(_, _)>>(),
        BinaryHeap::new(),
    );
    ratio.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (r, q) in ratio.into_iter() {
        curr += q;
        heap.push(q as i64);
        if heap.len() > k as usize {
            curr -= heap.pop().unwrap() as f64;
        }
        if heap.len() == k as usize {
            ans = ans.min(curr * r);
        }
    }
    ans
}

/// p858
pub fn mirror_reflection(p: i32, q: i32) -> i32 {
    fn gcd<T: Copy + std::cmp::PartialOrd + std::ops::Sub<Output = T>>(m: T, n: T) -> T {
        if m > n {
            return gcd(m - n, n);
        };
        if m < n {
            return gcd(n - m, m);
        };
        m
    }
    let d = gcd(p, q);
    let (m, n) = (p / d, q / d);
    if n % 2 == 0 {
        0
    } else if m % 2 == 0 {
        2
    } else {
        1
    }
}

/// p859
pub fn buddy_strings(s: String, goal: String) -> bool {
    if s.len() != goal.len() {
        return false;
    }
    if s == goal {
        let mut count: [usize; 26] = [0; 26];
        s.chars().for_each(|c| {
            let distance = u32::from(c) - u32::from('a');
            count[distance as usize] = count[distance as usize] + 1
        });
        return count.iter().any(|c| *c >= 2);
    } else {
        let mut diff = Vec::new();
        s.chars().zip(goal.chars()).for_each(|(c1, c2)| {
            if c1 != c2 {
                diff.push((c1, c2))
            }
        });
        return diff.len() == 2 && diff[0].0 == diff[1].1 && diff[0].1 == diff[1].0;
    }
}

/// p860
pub fn lemonade_change(bills: Vec<i32>) -> bool {
    let mut changes: HashMap<i32, usize> = HashMap::new();
    for m in bills {
        match m {
            5 => {
                *changes.entry(5).or_insert(0) += 1;
            }
            10 => {
                *changes.entry(10).or_insert(0) += 1;
                if *changes.entry(5).or_insert(0) >= 1 {
                    *changes.entry(5).or_insert(0) -= 1;
                } else {
                    return false;
                }
            }
            20 => {
                *changes.entry(20).or_insert(0) += 1;
                if *changes.entry(5).or_insert(0) >= 1 && *changes.entry(10).or_insert(0) >= 1 {
                    *changes.entry(5).or_insert(0) -= 1;
                    *changes.entry(10).or_insert(0) -= 1;
                } else if *changes.entry(5).or_insert(0) >= 3 {
                    *changes.entry(5).or_insert(0) -= 3;
                } else {
                    return false;
                }
            }
            _ => (),
        }
    }
    return true;
}

/// p861
pub fn matrix_score(grid: Vec<Vec<i32>>) -> i32 {
    let mut grid = grid;
    for i in 0..grid.len() {
        if grid[i][0] == 1 {
            continue;
        }
        for j in 0..grid[i].len() {
            grid[i][j] = if grid[i][j] == 1 { 0 } else { 1 };
        }
    }
    for j in 1..grid[0].len() {
        let mut ones = 0;
        for i in 0..grid.len() {
            ones += grid[i][j];
        }
        if ones >= grid.len() as i32 - ones {
            continue;
        }
        for i in 0..grid.len() {
            grid[i][j] = if grid[i][j] == 1 { 0 } else { 1 };
        }
    }
    let mut ans = 0;
    for v in grid {
        ans += v.iter().fold(0, |acc, &n| acc << 1 | n);
    }
    ans
}

/// p862
pub fn shortest_subarray(nums: Vec<i32>, k: i32) -> i32 {
    use std::collections::VecDeque;
    let (mut ret, mut pre_sum, mut queue) = (i64::MAX, 0, VecDeque::new());
    queue.push_back((0, -1));
    for i in 0..nums.len() {
        pre_sum += nums[i] as i64;
        while !queue.is_empty() && pre_sum <= queue[queue.len() - 1].0 {
            queue.pop_back();
        }
        while !queue.is_empty() && pre_sum - queue[0].0 >= k as i64 {
            ret = ret.min(i as i64 - queue.pop_front().unwrap().1 as i64);
        }
        queue.push_back((pre_sum, i as i32));
    }
    if ret == i64::MAX { -1 } else { ret as i32 }
}

/// p863
pub fn distance_k(
    root: Option<Rc<RefCell<TreeNode>>>,
    target: Option<Rc<RefCell<TreeNode>>>,
    k: i32,
) -> Vec<i32> {
    fn dfs(
        root: Option<Rc<RefCell<TreeNode>>>,
        target: i32,
        k: i32,
        result: &mut Vec<i32>,
    ) -> Option<i32> {
        match root {
            Some(root) => {
                let val = root.borrow().val;
                if val == target {
                    dfs2(Some(root), 0, k, result);
                    return Some(0);
                }

                let left = root.borrow().left.clone();
                let right = root.borrow().right.clone();

                let left_distance = dfs(left.clone(), target, k, result);
                if let Some(left_distance) = left_distance {
                    let cur_distance = left_distance + 1;
                    if cur_distance == k {
                        result.push(val);
                    } else if cur_distance < k {
                        dfs2(right, cur_distance + 1, k, result);
                    }
                    return Some(cur_distance);
                }

                let right_distance = dfs(right, target, k, result);
                if let Some(right_distance) = right_distance {
                    let cur_distance = right_distance + 1;
                    if cur_distance == k {
                        result.push(val);
                    } else if cur_distance < k {
                        dfs2(left, cur_distance + 1, k, result);
                    }
                    return Some(cur_distance);
                }

                None
            }
            None => None,
        }
    }

    fn dfs2(root: Option<Rc<RefCell<TreeNode>>>, distance: i32, k: i32, result: &mut Vec<i32>) {
        if let Some(root) = root {
            if distance == k {
                result.push(root.borrow().val);
                return;
            }

            dfs2(root.borrow_mut().left.take(), distance + 1, k, result);
            dfs2(root.borrow_mut().right.take(), distance + 1, k, result);
        }
    }
    let target = target.unwrap().borrow().val;
    let mut result = Vec::new();
    dfs(root, target, k, &mut result);
    result
}

/// p864
pub fn shortest_path_all_keys(grid: Vec<String>) -> i32 {
    use std::collections::VecDeque;
    let grid = grid
        .iter()
        .map(|s| s.chars().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let (mut queue, mut keys_total, m, n) = (VecDeque::new(), 0, grid.len(), grid[0].len());
    for i in 0..m {
        for j in 0..n {
            if grid[i][j] == '@' {
                queue.push_back((i, j, 0 as i32, 0 as i32));
            }
            if grid[i][j] >= 'a' && grid[i][j] <= 'f' {
                keys_total |= 1 << (grid[i][j] as i32 - 'a' as i32);
            }
        }
    }
    let mut visited = vec![vec![vec![false; keys_total as usize + 1]; n]; m];
    while let Some(curr) = queue.pop_front() {
        let (step, keys_cnt) = (curr.2, curr.3);
        if keys_total == keys_cnt {
            return step;
        }
        for dict in [[1, 0], [-1, 0], [0, 1], [0, -1]] {
            let (x, y) = (curr.0 + dict[0] as usize, curr.1 + dict[1] as usize);
            if x >= m || y >= n || grid[x][y] == '#' {
                continue;
            }
            let new_keys_cnt = keys_cnt | (1 << (grid[x][y] as i32 - 'a' as i32));
            if grid[x][y] >= 'a' && grid[x][y] <= 'f' && !visited[x][y][new_keys_cnt as usize] {
                visited[x][y][new_keys_cnt as usize] = true;
                queue.push_back((x, y, step + 1, new_keys_cnt));
            }
            if (grid[x][y] >= 'A'
                && grid[x][y] <= 'F'
                && keys_cnt >> (grid[x][y] as i32 - 'A' as i32) & 1 == 1
                || grid[x][y] == '@'
                || grid[x][y] == '.')
                && !visited[x][y][keys_cnt as usize]
            {
                visited[x][y][keys_cnt as usize] = true;
                queue.push_back((x, y, step + 1, keys_cnt));
            }
        }
    }
    -1
}

/// p865
pub fn subtree_with_all_deepest(
    root: Option<Rc<RefCell<TreeNode>>>,
) -> Option<Rc<RefCell<TreeNode>>> {
    fn depth_and_ancester(
        root: Option<Rc<RefCell<TreeNode>>>,
    ) -> (usize, Option<Rc<RefCell<TreeNode>>>) {
        if let Some(node) = root.as_ref() {
            let (left_depth, left_ancestor) = depth_and_ancester(node.borrow().left.clone());
            let (right_depth, right_ancestor) = depth_and_ancester(node.borrow().right.clone());
            if left_depth > right_depth {
                return (left_depth + 1, left_ancestor);
            } else if right_depth > left_depth {
                return (right_depth + 1, right_ancestor);
            } else {
                return (left_depth + 1, root);
            }
        } else {
            (0, root)
        }
    }
    depth_and_ancester(root).1
}

/// p866
pub fn prime_palindrome(mut n: i32) -> i32 {
    fn is_palindrome(mut n: i32) -> bool {
        if n % 10 == 0 {
            return false;
        }

        let mut rev = 0;

        while n > rev {
            rev *= 10;
            rev += n % 10;
            n /= 10;
        }

        n == rev || n == rev / 10
    }

    fn is_prime(n: i32) -> bool {
        let mut i = 2;

        while i * i <= n && n % i != 0 {
            i += 1;
        }

        n > 1 && i * i > n
    }

    loop {
        match n {
            999 | 99_999 | 9_999_999 => n = n * 10 + 11,
            n if is_palindrome(n) && is_prime(n) => return n,
            _ => n += 1,
        }
    }
}

/// p867
pub fn transpose(matrix: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let (cols, rows) = (matrix.len(), matrix[0].len());
    let mut new_matrix = vec![vec![0; cols]; rows];
    for i in 0..cols {
        for j in 0..rows {
            new_matrix[j][i] = matrix[i][j];
        }
    }
    new_matrix
}

/// p868
pub fn binary_gap(n: i32) -> i32 {
    format!("{:b}", n)
        .char_indices()
        .fold((-31, 0), |(distance, maximum), i| {
            if i.1 == '1' {
                (1, maximum.max(distance))
            } else {
                (distance + 1, maximum)
            }
        })
        .1
}

/// p869
pub fn reordered_power_of2(n: i32) -> bool {
    let mut target = n.to_string().chars().collect::<Vec<char>>();
    target.sort();
    for i in 0..32 {
        let mut tmp = 2_i32.pow(i).to_string().chars().collect::<Vec<char>>();
        tmp.sort();
        if tmp == target {
            return true;
        }
    }
    false
}

/// p870
pub fn advantage_count(mut nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    let length = nums2.len();
    let mut res = vec![0; length];
    let mut with_index = nums2.into_iter().enumerate().collect::<Vec<(usize, i32)>>();
    with_index.sort_by(|a, b| b.1.cmp(&a.1));
    nums1.sort();
    with_index.iter().for_each(|(i, v)| {
        if nums1.last().unwrap() > v {
            res[*i] = nums1.pop().unwrap();
        } else {
            res[*i] = nums1.remove(0);
        }
    });
    res
}

/// p871
pub fn min_refuel_stops(target: i32, start_fuel: i32, mut stations: Vec<Vec<i32>>) -> i32 {
    stations.push(vec![target, 0]);
    let mut ans = 0;
    let mut miles = start_fuel;
    let mut fuel_heap = BinaryHeap::new();
    for station in stations {
        let position = station[0];
        while !fuel_heap.is_empty() && miles < position {
            miles += fuel_heap.pop().unwrap();
            ans += 1;
        }
        if miles < position {
            return -1;
        }
        fuel_heap.push(station[1]);
    }
    ans
}

/// p872
pub fn leaf_similar(
    root1: Option<Rc<RefCell<TreeNode>>>,
    root2: Option<Rc<RefCell<TreeNode>>>,
) -> bool {
    let (mut leaf1, mut leaf2) = (vec![], vec![]);
    fn get_leaf_seq(root: &Option<Rc<RefCell<TreeNode>>>, seq: &mut Vec<i32>) {
        if let Some(node) = root.as_ref() {
            if node.borrow().left.is_none() && node.borrow().right.is_none() {
                seq.push(node.borrow().val);
            } else {
                get_leaf_seq(&node.borrow().left, seq);
                get_leaf_seq(&node.borrow().right, seq);
            }
        }
    }
    get_leaf_seq(&root1, &mut leaf1);
    get_leaf_seq(&root2, &mut leaf2);
    leaf1 == leaf2
}

/// p873
pub fn len_longest_fib_subseq(arr: Vec<i32>) -> i32 {
    let length = arr.len();
    let mut dp = vec![vec![0; length]; length];
    let mut max = 0;
    for i in 1..length {
        for j in 0..i {
            if arr[j] * 2 < arr[i] {
                continue;
            }
            let target = arr[i] - arr[j];
            for k in 0..j {
                if arr[k] == target {
                    dp[i][j] = (dp[j][k] + 1).max(3);
                    max = max.max(dp[i][j]);
                }
            }
        }
    }
    max
}
