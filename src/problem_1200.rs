use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    iter::repeat,
    rc::Rc,
    sync::{
        Condvar, Mutex,
        mpsc::{Receiver, Sender, channel},
    },
};

use crate::common::TreeNode;

/// 1103
pub fn distribute_candies(mut candies: i32, n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut ans = vec![0; n];
    let mut i = 1;
    while candies > 0 {
        ans[(i - 1) as usize % n] += i.min(candies);
        candies -= i;
        i += 1;
    }
    ans
}

/// 1104
pub fn path_in_zig_zag_tree(label: i32) -> Vec<i32> {
    let mut x = 2_i32.pow((label as f64).log2() as u32);
    let mut label = label;
    let mut ret = vec![label];

    while label > 1 {
        label = x - 1 - ((label / 2) % (x / 2));
        x /= 2;
        ret.push(label);
    }

    ret.reverse();
    ret
}

/// 1105
pub fn min_height_shelves(books: Vec<Vec<i32>>, shelf_width: i32) -> i32 {
    let n = books.len();
    let mut dp = vec![0; n + 1];
    for i in 1..=n {
        let (mut j, mut width, mut max_height) = (i - 1, books[i - 1][0], books[i - 1][1]);
        dp[i] = dp[i - 1] + max_height;
        while j > 0 && width + books[j - 1][0] <= shelf_width {
            max_height = max_height.max(books[j - 1][1]);
            width += books[j - 1][0];
            dp[i] = dp[i].min(dp[j - 1] + max_height);
            j -= 1;
        }
    }
    dp[n]
}

/// 1106
pub fn parse_bool_expr(expression: String) -> bool {
    let mut stack = VecDeque::new();
    expression.chars().for_each(|c| match c {
        '(' | 't' | 'f' | '!' | '&' | '|' => {
            stack.push_back(c);
        }
        ')' => {
            let mut list = vec![];
            while *stack.back().unwrap() != '(' {
                list.push(stack.pop_back().unwrap())
            }
            stack.pop_back();
            match stack.pop_back().unwrap() {
                '&' => {
                    if list.contains(&'f') {
                        stack.push_back('f')
                    } else {
                        stack.push_back('t')
                    }
                }
                '!' => {
                    if list.contains(&'f') {
                        stack.push_back('t')
                    } else {
                        stack.push_back('f')
                    }
                }
                '|' => {
                    if list.contains(&'t') {
                        stack.push_back('t')
                    } else {
                        stack.push_back('f')
                    }
                }
                _ => (),
            }
        }
        ',' => (),
        _ => (),
    });
    stack.pop_back().unwrap() == 't'
}

/// 1108
pub fn defang_i_paddr(address: String) -> String {
    address.replace(".", "[.]")
}

/// 1109
pub fn corp_flight_bookings(bookings: Vec<Vec<i32>>, n: i32) -> Vec<i32> {
    let (mut get_on, mut get_off) = (HashMap::new(), HashMap::new());
    bookings.iter().for_each(|b| {
        let on = get_on.entry(b[0]).or_insert(0);
        *on = *on + b[2];
        let off = get_off.entry(b[1] + 1).or_insert(0);
        *off = *off + b[2];
    });
    let (mut total, mut time_line) = (0, vec![0; n as usize]);
    time_line.iter_mut().enumerate().for_each(|(day, cnt)| {
        let (on, off) = (
            get_on.get(&((day + 1) as i32)).unwrap_or(&0),
            get_off.get(&((day + 1) as i32)).unwrap_or(&0),
        );
        total = total + on - off;
        *cnt = total;
    });
    time_line
}

/// 1110
pub fn del_nodes(
    root: Option<Rc<RefCell<TreeNode>>>,
    mut to_delete: Vec<i32>,
) -> Vec<Option<Rc<RefCell<TreeNode>>>> {
    let mut ret = vec![];
    fn dfs(
        root: Option<Rc<RefCell<TreeNode>>>,
        to_delete: &mut Vec<i32>,
        ret: &mut Vec<Option<Rc<RefCell<TreeNode>>>>,
        is_root: bool,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() {
            return None;
        }
        let is_delete = to_delete.contains(&root.as_ref().unwrap().borrow().val);
        if !is_delete && is_root {
            ret.push(root.clone());
        }
        let (left, right) = (
            root.as_ref().unwrap().borrow().left.clone(),
            root.as_ref().unwrap().borrow().right.clone(),
        );
        root.as_ref().unwrap().borrow_mut().left = dfs(left, to_delete, ret, is_delete);
        root.as_ref().unwrap().borrow_mut().right = dfs(right, to_delete, ret, is_delete);
        if is_delete { None } else { root.clone() }
    }
    dfs(root, &mut to_delete, &mut ret, true);
    ret
}

/// 1111
pub fn max_depth_after_split(seq: String) -> Vec<i32> {
    let (mut d, mut ans) = (0, vec![]);
    seq.chars().for_each(|c| match c {
        '(' => {
            d = d + 1;
            ans.push(d % 2);
        }
        ')' => {
            ans.push(d % 2);
            d = d - 1;
        }
        _ => (),
    });
    ans
}

/// 1114
struct Foo {
    s_1: Sender<()>,
    r_1: Mutex<Receiver<()>>,
    s_2: Sender<()>,
    r_2: Mutex<Receiver<()>>,
}

impl Foo {
    fn new() -> Self {
        let (s_1, r_1) = channel();
        let (s_2, r_2) = channel();
        Foo {
            s_1,
            r_1: Mutex::new(r_1),
            s_2,
            r_2: Mutex::new(r_2),
        }
    }

    fn first<F>(&self, print_first: F)
    where
        F: FnOnce(),
    {
        // Do not change this line
        print_first();
        let _ = self.s_1.send(());
    }

    fn second<F>(&self, print_second: F)
    where
        F: FnOnce(),
    {
        // Do not change this line
        let _ = self.r_1.lock().unwrap().recv();
        print_second();
        let _ = self.s_2.send(());
    }

    fn third<F>(&self, print_third: F)
    where
        F: FnOnce(),
    {
        let _ = self.r_2.lock().unwrap().recv();
        // Do not change this line
        print_third();
    }
}

/// 1115
struct FooBar {
    n: usize,
    state: Mutex<i32>,
    cv: Condvar,
}

impl FooBar {
    fn new(n: usize) -> Self {
        FooBar {
            n,
            state: Mutex::new(0),
            cv: Condvar::new(),
        }
    }

    fn foo<F>(&self, print_foo: F)
    where
        F: Fn(),
    {
        for _ in 0..self.n {
            let mut state = self.state.lock().unwrap();
            while *state != 0 {
                state = self.cv.wait(state).unwrap();
            }
            print_foo();
            *state = 1;
            self.cv.notify_one();
        }
    }

    fn bar<F>(&self, print_bar: F)
    where
        F: Fn(),
    {
        for _ in 0..self.n {
            let mut state = self.state.lock().unwrap();
            while *state != 1 {
                state = self.cv.wait(state).unwrap();
            }
            print_bar();
            *state = 0;
            self.cv.notify_one();
        }
    }
}

/// 1122
pub fn relative_sort_array(mut arr1: Vec<i32>, arr2: Vec<i32>) -> Vec<i32> {
    let record = arr2
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut acc, cur| {
            acc.insert(*cur.1, cur.0);
            acc
        });
    arr1.sort_by(|a, b| {
        let (position_a, position_b) = (
            record.get(a).unwrap_or(&usize::MAX),
            record.get(b).unwrap_or(&usize::MAX),
        );
        if *position_a == usize::MAX && *position_b == usize::MAX {
            return a.cmp(b);
        } else {
            position_a.cmp(position_b)
        }
    });
    arr1
}

/// 1123
pub fn lca_deepest_leaves(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    fn depth(root: Option<Rc<RefCell<TreeNode>>>) -> (usize, Option<Rc<RefCell<TreeNode>>>) {
        if let Some(node) = root.as_ref() {
            let (l_depth, l_root) = depth(node.borrow().left.clone());
            let (r_depth, r_root) = depth(node.borrow().right.clone());
            if l_depth > r_depth {
                return (l_depth + 1, l_root);
            } else if r_depth > l_depth {
                return (r_depth + 1, r_root);
            } else {
                return (r_depth + 1, root);
            }
        } else {
            (0, None)
        }
    }
    depth(root).1
}

/// 1124
pub fn longest_wpi(hours: Vec<i32>) -> i32 {
    let (mut ans, mut map) = (0, HashMap::with_capacity(hours.len()));
    hours.into_iter().enumerate().fold(0, |mut acc, (idx, x)| {
        acc += if x > 8 { 1 } else { -1 };
        match acc > 0 {
            true => ans = idx + 1,
            false => {
                match map.get(&(acc - 1)) {
                    Some(v) => ans = ans.max(idx - v),
                    None => (),
                };
                map.entry(acc).or_insert(idx);
            }
        };
        acc
    });
    ans as i32
}

/// 1125
pub fn smallest_sufficient_team(req_skills: Vec<String>, people: Vec<Vec<String>>) -> Vec<i32> {
    use std::collections::HashMap;
    let (m, n) = (people.len(), req_skills.len());
    let (mut dp, skill_index) = (
        vec![None; 1 << n],
        req_skills
            .into_iter()
            .enumerate()
            .map(|(index, skill)| (skill, index))
            .collect::<HashMap<String, usize>>(),
    );
    dp[0] = Some(vec![]);
    for i in 0..m {
        let curr_skill = people[i]
            .iter()
            .fold(0, |curr_skill, s| curr_skill | 1 << skill_index[s]);
        for prev in 0..dp.len() {
            if dp[prev].is_none() {
                continue;
            }
            let curr = curr_skill | prev;
            if dp[curr].is_none()
                || dp[prev].as_ref().unwrap().len() + 1 < dp[curr].as_ref().unwrap().len()
            {
                dp[curr] = dp[prev].clone();
                dp[curr].as_mut().unwrap().push(i as i32);
            }
        }
    }
    dp[(1 << n) - 1].as_ref().unwrap().clone()
}

/// 1128
pub fn num_equiv_domino_pairs(dominoes: Vec<Vec<i32>>) -> i32 {
    let cnt = dominoes.iter().fold(HashMap::new(), |mut acc, cur| {
        let tag = if cur[0] <= cur[1] {
            (cur[0], cur[1])
        } else {
            (cur[1], cur[0])
        };
        let cnt = acc.entry(tag).or_insert(0);
        *cnt = *cnt + 1;
        acc
    });
    cnt.values().fold(0, |acc, cur| acc + cur * (cur - 1) / 2)
}

/// 1129
pub fn shortest_alternating_paths(
    n: i32,
    red_edges: Vec<Vec<i32>>,
    blue_edges: Vec<Vec<i32>>,
) -> Vec<i32> {
    use std::collections::{HashSet, VecDeque};
    fn build_graph(
        n: usize,
        red_edges: &Vec<Vec<i32>>,
        blue_edges: &Vec<Vec<i32>>,
    ) -> Vec<Vec<i32>> {
        let mut graph = vec![vec![-(n as i32); n as usize]; n as usize];
        for re in red_edges {
            graph[re[0] as usize][re[1] as usize] = 1;
        }
        for be in blue_edges {
            graph[be[0] as usize][be[1] as usize] = if graph[be[0] as usize][be[1] as usize] == 1 {
                0
            } else {
                -1
            };
        }
        graph
    }
    let (graph, mut queue, mut visited, mut answer, mut len) = (
        build_graph(n as usize, &red_edges, &blue_edges),
        VecDeque::new(),
        HashSet::new(),
        vec![i32::MAX; n as usize],
        0,
    );
    queue.push_back((0, 1));
    queue.push_back((0, -1));
    answer[0] = 0;
    while !queue.is_empty() {
        len += 1;
        let size = queue.len();
        for _ in 0..size {
            let curr = queue.pop_front().unwrap();
            let (node, opposite_color) = (curr.0, -curr.1);
            for i in 0..n as usize {
                if graph[node][i] != opposite_color && graph[node][i] != 0 {
                    continue;
                }
                if !visited.insert((i, opposite_color)) {
                    continue;
                }
                queue.push_back((i as usize, opposite_color));
                answer[i] = answer[i].min(len);
            }
        }
    }
    for i in 1..n as usize {
        answer[i] = if answer[i] == i32::MAX { -1 } else { answer[i] };
    }
    answer
}

/// 1130
pub fn mct_from_leaf_values(arr: Vec<i32>) -> i32 {
    use std::collections::VecDeque;
    let (mut queue, mut min_sum) = (VecDeque::new(), 0);
    queue.push_back(i32::MAX);
    for a in arr {
        while queue.back().unwrap() <= &a {
            let top = queue.pop_back().unwrap();
            min_sum += top * a.min(*queue.back().unwrap());
        }
        queue.push_back(a)
    }
    while queue.len() > 2 {
        min_sum += queue.pop_back().unwrap() * queue.back().unwrap();
    }
    min_sum
}

/// 1137
pub fn tribonacci(n: i32) -> i32 {
    if n == 0 {
        return 0;
    } else if n == 1 || n == 2 {
        return 1;
    } else {
        let mut pre_pre = 0;
        let mut pre = 1;
        let mut cur = 1;
        for _ in 3..=n {
            let temp = pre_pre;
            pre_pre = pre;
            pre = cur;
            cur = pre_pre + pre + temp;
        }
        cur
    }
}

/// 1138
pub fn alphabet_board_path(target: String) -> String {
    let locations = (b'a'..=b'z')
        .map(|c| {
            let distance = (c - b'a') as i8;
            (c as char, (distance / 5, distance % 5))
        })
        .collect::<HashMap<char, (i8, i8)>>();
    target
        .chars()
        .fold((String::new(), (0, 0)), |mut acc, cur| {
            let location = locations.get(&cur).unwrap();
            let (y_diff, x_diff) = (location.0 - acc.1.0, location.1 - acc.1.1);
            if acc.1.0 == 5 {
                if y_diff > 0 {
                    for _ in 0..y_diff {
                        acc.0.push('D');
                    }
                } else if y_diff < 0 {
                    for _ in 0..-y_diff {
                        acc.0.push('U');
                    }
                }
                if x_diff > 0 {
                    for _ in 0..x_diff {
                        acc.0.push('R');
                    }
                } else if x_diff < 0 {
                    for _ in 0..-x_diff {
                        acc.0.push('L');
                    }
                }
            } else {
                if x_diff > 0 {
                    for _ in 0..x_diff {
                        acc.0.push('R');
                    }
                } else if x_diff < 0 {
                    for _ in 0..-x_diff {
                        acc.0.push('L');
                    }
                }
                if y_diff > 0 {
                    for _ in 0..y_diff {
                        acc.0.push('D');
                    }
                } else if y_diff < 0 {
                    for _ in 0..-y_diff {
                        acc.0.push('U');
                    }
                }
            }

            acc.0.push('!');
            (acc.0, *location)
        })
        .0
}

/// 1139
pub fn largest1_bordered_square(grid: Vec<Vec<i32>>) -> i32 {
    let (height, width) = (grid.len(), grid[0].len());
    let (mut left, mut up, mut max) = (
        vec![vec![0; width]; height],
        vec![vec![0; width]; height],
        0,
    );
    for r in 0..height {
        for c in 0..width {
            if grid[r][c] == 1 {
                if c - 1 < width {
                    left[r][c] = left[r][c - 1] + 1;
                } else {
                    left[r][c] = 1;
                }
                if r - 1 < height {
                    up[r][c] = up[r - 1][c] + 1;
                } else {
                    up[r][c] = 1;
                }
            }
            for l in 1..=left[r][c].min(up[r][c]) {
                if up[r][c - l + 1] >= l && left[r - l + 1][c] >= l {
                    max = max.max(l);
                }
            }
        }
    }
    (max * max) as i32
}

/// 1140
pub fn stone_game_ii(piles: Vec<i32>) -> i32 {
    let mut sum = 0;
    let len = piles.len();

    let mut dp = vec![vec![0; len + 1]; len];

    for i in (0..len).rev() {
        sum += piles[i];
        for m in 1..len + 1 {
            if i + 2 * m >= len {
                dp[i][m] = sum;
            } else {
                for x in 1..2 * m + 1 {
                    dp[i][m] = dp[i][m].max(sum - dp[i + x][m.max(x)]);
                }
            }
        }
    }
    dp[0][1]
}

/// 1143
pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
    let (ch1, ch2) = (
        text1.chars().collect::<Vec<char>>(),
        text2.chars().collect::<Vec<char>>(),
    );
    let (len1, len2) = (ch1.len(), ch2.len());
    let mut dp = vec![vec![0; len2 + 1]; len1 + 1];
    for i in 1..=len1 {
        for j in 1..=len2 {
            if ch1[i - 1] == ch2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1
            } else {
                dp[i][j] = dp[i][j - 1].max(dp[i - 1][j])
            }
        }
    }
    dp[len1][len2]
}

/// 1144
pub fn moves_to_make_zigzag(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    let mut ans = [0, 0];
    for k in 0..2 {
        for i in (k..n).step_by(2) {
            let mut d = 0;
            if i > 0 {
                d = d.max(nums[i] - nums[i - 1] + 1);
            }
            if i + 1 < n {
                d = d.max(nums[i] - nums[i + 1] + 1);
            }
            ans[k] += d
        }
    }
    ans[0].min(ans[1])
}

/// 1145
pub fn btree_game_winning_move(root: Option<Rc<RefCell<TreeNode>>>, n: i32, x: i32) -> bool {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, x: i32, n: i32, ans: &mut i32) -> i32 {
        match root {
            Some(r) => {
                let left = dfs(r.borrow_mut().left.take(), x, n, ans);
                let right = dfs(r.borrow_mut().right.take(), x, n, ans);
                if r.borrow().val == x {
                    *ans = *[*ans, left, right, n - left - right - 1]
                        .iter()
                        .max()
                        .unwrap();
                }
                left + right + 1
            }
            None => 0,
        }
    }
    let mut ans = 0;
    dfs(root, x, n, &mut ans);
    ans > n - ans
}

/// 1147
pub fn longest_decomposition(text: String) -> i32 {
    let n = text.len();
    for i in 0..n / 2 {
        if text[..i + 1] == text[n - i - 1..] {
            return 2 + longest_decomposition(text[i + 1..n - i - 1].to_owned());
        }
    }
    if n == 0 { 0 } else { 1 }
}

/// 1154
pub fn day_of_year(date: String) -> i32 {
    let month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30];
    let (year, month, day): (i32, i32, i32) = {
        let parts: Vec<&str> = date.split('-').collect();
        (
            parts[0].parse().unwrap(),
            parts[1].parse().unwrap(),
            parts[2].parse().unwrap(),
        )
    };
    let total = month_days[0..(month - 1) as usize].iter().sum::<i32>();
    let leap_year = year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
    let total = if leap_year && month >= 3 {
        total + 1
    } else {
        total
    };
    total + day
}

/// 1155
pub fn num_rolls_to_target(n: i32, k: i32, target: i32) -> i32 {
    let (n, k, target) = (n as usize, k as usize, target as usize);
    let mut dp = vec![vec![0; n + 1]; target + 1];
    dp[0][0] = 1;
    for t in 1..=target.min(n * k) {
        for r in 1..=n {
            for i in 1..=t.min(k) {
                dp[t][r] = (dp[t][r] + dp[t - i][r - 1]) % 1_000_000_007
            }
        }
    }
    dp[target][n]
}

/// 1156
pub fn max_rep_opt1(text: String) -> i32 {
    let (mut cache, mut freq, mut left, text_arr) = (Vec::new(), vec![0; 26], 0, text.as_bytes());
    for i in 0..=text.len() {
        if i < text.len() {
            freq[(text_arr[i] - b'a') as usize] += 1;
        }
        if i == text.len() || i < text.len() && text_arr[i] != text_arr[left] {
            cache.push((text_arr[left], (i - left) as i32));
            left = i;
        }
    }
    let mut ret = 0;
    for i in 0..cache.len() {
        let mut len = cache[i].1;
        if i + 2 < cache.len() && cache[i + 2].0 == cache[i].0 && cache[i + 1].1 == 1 {
            len += cache[i + 2].1;
        }
        ret = ret.max(
            len + (if len < freq[(cache[i].0 - b'a') as usize] {
                1
            } else {
                0
            }),
        );
    }
    ret
}

/// 1157
struct MajorityChecker {
    st: SegmentTree,
    cq: CountQuicker,
}

struct SegmentTree {
    n: i32,
    candidate: Vec<i32>,
    hp: Vec<i32>,
}
impl SegmentTree {
    pub fn new(arr: &mut Vec<i32>) -> Self {
        let n = arr.len() as i32;
        let candidate: Vec<i32> = repeat(0).take(((n + 1) << 2) as usize).collect();
        let hp: Vec<i32> = repeat(0).take(((n + 1) << 2) as usize).collect();
        let mut ans = SegmentTree { n, candidate, hp };
        ans.build(arr, 1, n, 1);
        return ans;
    }

    fn build(&mut self, arr: &mut Vec<i32>, l: i32, r: i32, rt: i32) {
        if l == r {
            self.candidate[rt as usize] = arr[(l - 1) as usize];
            self.hp[rt as usize] = 1;
        } else {
            let m = (l + r) >> 1;
            self.build(arr, l, m, rt << 1);
            self.build(arr, m + 1, r, rt << 1 | 1);
            let lc = self.candidate[(rt << 1) as usize];
            let rc = self.candidate[(rt << 1 | 1) as usize];
            let lh = self.hp[(rt << 1) as usize];
            let rh = self.hp[(rt << 1 | 1) as usize];
            if lc == rc {
                self.candidate[rt as usize] = lc;
                self.hp[rt as usize] = lh + rh;
            } else {
                self.candidate[rt as usize] = if lh >= rh { lc } else { rc };
                self.hp[rt as usize] = i32::abs(lh - rh);
            }
        }
    }

    pub fn query(&mut self, left: i32, right: i32) -> i32 {
        return self.query0(left + 1, right + 1, 1, self.n, 1)[0];
    }

    fn query0(&mut self, ll: i32, rr: i32, l: i32, r: i32, rt: i32) -> Vec<i32> {
        if ll <= l && r <= rr {
            return vec![self.candidate[rt as usize], self.hp[rt as usize]];
        }
        let m = (l + r) >> 1;
        if rr <= m {
            return self.query0(ll, rr, l, m, rt << 1);
        } else if ll > m {
            return self.query0(ll, rr, m + 1, r, rt << 1 | 1);
        } else {
            let mut ansl = self.query0(ll, rr, l, m, rt << 1);
            let mut ansr = self.query0(ll, rr, m + 1, r, rt << 1 | 1);
            if ansl[0] == ansr[0] {
                ansl[1] += ansr[1];
                return ansl;
            } else {
                if ansl[1] >= ansr[1] {
                    ansl[1] -= ansr[1];
                    return ansl;
                } else {
                    ansr[1] -= ansl[1];
                    return ansr;
                }
            }
        }
    }
}
struct CountQuicker {
    cnt: Vec<Vec<i32>>,
}
impl CountQuicker {
    pub fn new(arr: &mut Vec<i32>) -> Self {
        let mut cnt: Vec<Vec<i32>> = vec![];
        let max = *arr.iter().max().unwrap_or(&0);
        for _i in 0..=max {
            cnt.push(vec![]);
        }
        for i in 0..arr.len() as i32 {
            cnt[arr[i as usize] as usize].push(i);
        }
        return Self { cnt };
    }

    pub fn real_times(&mut self, left: i32, right: i32, num: i32) -> i32 {
        self.size(num, right) - self.size(num, left - 1)
    }

    fn size(&mut self, indies_index: i32, index: i32) -> i32 {
        let mut l = 0;
        let mut r = self.cnt[indies_index as usize].len() as i32 - 1;
        let mut m = 0;
        let mut ans = -1;
        while l <= r {
            m = (l + r) / 2;
            if self.cnt[indies_index as usize][m as usize] <= index {
                ans = m;
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return ans + 1;
    }
}

impl MajorityChecker {
    fn new(arr: Vec<i32>) -> Self {
        let mut arr = arr;
        let st = SegmentTree::new(&mut arr);
        let cq = CountQuicker::new(&mut arr);
        Self { st, cq }
    }

    fn query(&mut self, left: i32, right: i32, threshold: i32) -> i32 {
        let candidate = self.st.query(left, right);
        return if self.cq.real_times(left, right, candidate) >= threshold {
            candidate
        } else {
            -1
        };
    }
}

/// 1160
pub fn count_characters(words: Vec<String>, chars: String) -> i32 {
    let cnt = chars.chars().fold(HashMap::new(), |mut acc, cur| {
        *acc.entry(cur).or_insert(0) += 1;
        acc
    });
    words.iter().fold(0, |acc, cur| {
        let (mut is_in, mut record) = (true, HashMap::new());
        for c in cur.chars() {
            let n = record.entry(c).or_insert(0);
            *n = *n + 1;
            let max = cnt.get(&c);
            if let Some(v) = max {
                if v < n {
                    is_in = false;
                    break;
                }
            } else {
                is_in = false;
                break;
            }
        }
        acc + if is_in { cur.len() } else { 0 }
    }) as i32
}

#[test]
fn test_1200() {
    num_rolls_to_target(3, 6, 6);
}
