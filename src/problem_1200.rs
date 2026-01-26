use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashMap, VecDeque},
    i32,
    iter::repeat,
    rc::Rc,
    sync::{
        Condvar, Mutex,
        mpsc::{Receiver, Sender, channel},
    },
};

use crate::common::{ListNode, TreeNode};

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
        let mut m;
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

/// 1161
pub fn max_level_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let (mut level_node, mut level_sum) = (vec![root], vec![]);
    fn bfs(level_node: &mut Vec<Option<Rc<RefCell<TreeNode>>>>, level_sum: &mut Vec<i32>) {
        let mut sum = 0;
        for _ in 0..level_node.len() {
            let v = level_node.remove(0);
            if let Some(node) = v.as_ref() {
                sum = sum + node.borrow().val;
                if node.borrow().left.is_some() {
                    level_node.push(node.borrow().left.clone());
                }
                if node.borrow().right.is_some() {
                    level_node.push(node.borrow().right.clone());
                }
            }
        }
        level_sum.push(sum)
    }
    while level_node.len() != 0 {
        bfs(&mut level_node, &mut level_sum);
    }
    (level_sum
        .iter()
        .position(|v| v == level_sum.iter().max().unwrap())
        .unwrap()
        + 1) as i32
}

/// 1162
pub fn max_distance(grid: Vec<Vec<i32>>) -> i32 {
    let (height, width) = (grid.len(), grid[0].len());
    let mut dp = vec![vec![0; width]; height];
    for r in 0..height {
        for c in 0..width {
            if grid[r][c] == 0 {
                dp[r][c] = i32::MAX;
                if r > 0 && dp[r - 1][c] != i32::MAX {
                    dp[r][c] = dp[r][c].min(dp[r - 1][c] + 1);
                }
                if c > 0 && dp[r][c - 1] != i32::MAX {
                    dp[r][c] = dp[r][c].min(dp[r][c - 1] + 1);
                }
            }
        }
    }
    for r in (0..height).rev() {
        for c in (0..width).rev() {
            if r < height - 1 && dp[r + 1][c] != i32::MAX {
                dp[r][c] = dp[r][c].min(dp[r + 1][c] + 1);
            }
            if c < width - 1 && dp[r][c + 1] != i32::MAX {
                dp[r][c] = dp[r][c].min(dp[r][c + 1] + 1);
            }
        }
    }
    let max = *dp.iter().flatten().max().unwrap();
    return if max == i32::MAX || max == 0 { -1 } else { max };
}

/// 1163
pub fn last_substring(s: String) -> String {
    let (mut i, mut j, mut k, n, s_arr) = (0, 1, 0, s.len(), s.as_bytes());
    while j + k < n {
        if s_arr[i + k] == s_arr[j + k] {
            k += 1;
        } else if s_arr[i + k] < s_arr[j + k] {
            i = j.max(i + k + 1);
            j = i + 1;
            k = 0;
        } else {
            j = j + k + 1;
            k = 0;
        }
    }
    s[i..].to_owned()
}

/// 1169
pub fn invalid_transactions(transactions: Vec<String>) -> Vec<String> {
    use std::collections::{HashMap, HashSet};
    let mut trans_map = HashMap::new(); // name -> (time, count, city)
    let mut invalid_indexes = HashSet::new();
    for (index, s) in transactions.iter().enumerate() {
        let item: Vec<_> = s.split(',').collect();
        let (name, time, count, city) = (
            item[0].to_string(),
            item[1].parse::<u32>().unwrap(),
            item[2].parse::<u32>().unwrap(),
            item[3].to_string(),
        );

        if count > 1000 {
            invalid_indexes.insert(index);
        }

        let transes = trans_map.entry(name).or_insert(vec![]);
        for (time1, city1, index1) in &*transes {
            if (time).abs_diff(*time1) <= 60 && *city1 != city {
                invalid_indexes.insert(index);
                invalid_indexes.insert(*index1);
            }
        }

        transes.push((time, city, index));
    }
    invalid_indexes
        .iter()
        .map(|index| transactions[*index].clone())
        .collect()
}

/// 1170
pub fn num_smaller_by_frequency(queries: Vec<String>, words: Vec<String>) -> Vec<i32> {
    let (_, n) = (queries.len(), words.len());
    fn get_count(word: &String) -> i32 {
        let mut cnt = vec![0; 26];
        word.bytes().for_each(|ch| cnt[(ch - b'a') as usize] += 1);
        for c in cnt {
            if c != 0 {
                return c;
            }
        }
        0
    }
    let (mut words_cnt, mut ret) = (
        words.iter().map(|word| get_count(word)).collect::<Vec<_>>(),
        vec![],
    );
    words_cnt.sort();
    for query in queries {
        let count = get_count(&query);
        let (mut l, mut r) = (0, n - 1);
        while l <= r && r < n {
            let mid = l + ((r - l) >> 1);
            if words_cnt[mid] <= count {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        ret.push((n - l) as i32)
    }
    ret
}

/// 1171
pub fn remove_zero_sum_sublists(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut dummy = Some(Box::new(ListNode { val: 0, next: head }));
    let (mut sum, mut pointer, mut prefix) = (0, &dummy, HashMap::new());
    while let Some(node) = pointer.as_ref() {
        sum = node.val + sum;
        prefix.insert(sum, pointer.clone());
        pointer = &node.next;
    }
    sum = 0;
    let mut pointer = &mut dummy;
    while let Some(node) = pointer.as_mut() {
        sum = node.val + sum;
        let position = prefix.get_mut(&sum).unwrap();
        node.next = position.as_mut().unwrap().next.take();
        pointer = &mut node.next;
    }

    dummy.unwrap().next
}

/// 1172
struct DinnerPlates {
    stack_map: HashMap<i32, Vec<i32>>,
    index: BinaryHeap<i32>,
    max_index: i32,
    capacity: usize,
}

impl DinnerPlates {
    fn new(capacity: i32) -> Self {
        let capacity = capacity as usize;
        Self {
            stack_map: HashMap::new(),
            index: BinaryHeap::new(),
            max_index: 0,
            capacity,
        }
    }

    fn push(&mut self, val: i32) {
        if let Some(mut target) = self.index.pop() {
            target = -target;
            self.max_index = self.max_index.max(target);
            let list = self.stack_map.entry(target).or_insert(vec![]);
            (*list).push(val);
        } else {
            let list = self.stack_map.entry(self.max_index).or_insert(vec![]);
            if (*list).len() != self.capacity {
                (*list).push(val);
            } else {
                self.max_index += 1;
                self.stack_map.insert(self.max_index, vec![val]);
            }
        }
    }

    fn pop(&mut self) -> i32 {
        while let Some(ref_list) = self.stack_map.get_mut(&self.max_index) {
            if let Some(num) = ref_list.pop() {
                self.index.push(-self.max_index);
                return num;
            }
            self.max_index -= 1;
        }
        self.max_index = 0;
        -1
    }

    fn pop_at_stack(&mut self, index: i32) -> i32 {
        if let Some(ref_list) = self.stack_map.get_mut(&index) {
            if let Some(num) = ref_list.pop() {
                self.index.push(-index);
                num
            } else {
                -1
            }
        } else {
            -1
        }
    }
}

/// 1175
pub fn num_prime_arrangements(n: i32) -> i32 {
    if n < 2 {
        return 1;
    }
    let is_prime = |num: i32| -> bool {
        for i in 2..=num / 2 {
            if num % i == 0 {
                return false;
            }
        }
        true
    };

    let (mut ret, prime, not_prime) = (
        1,
        (3..=n).filter(|i| is_prime(*i)).count() as i32 + 1,
        (3..=n).filter(|i| !is_prime(*i)).count() as i32 + 1,
    );
    for i in 1..=prime {
        ret *= i as i64;
        ret %= 1000000007;
    }
    for i in 1..=not_prime {
        ret *= i as i64;
        ret %= 1000000007;
    }
    ret as i32
}

/// 1177
pub fn can_make_pali_queries(s: String, queries: Vec<Vec<i32>>) -> Vec<bool> {
    let (mut prefix, mut ret) = (vec![0; s.len() + 1], vec![]);
    for i in 0..s.len() {
        prefix[i + 1] = prefix[i] ^ 1 << (s.as_bytes()[i] - b'a');
    }
    for query in queries {
        ret.push(
            if (prefix[query[1] as usize + 1] ^ prefix[query[0] as usize] as i32).count_ones()
                as i32
                / 2
                <= query[2]
            {
                true
            } else {
                false
            },
        );
    }
    ret
}

/// 1178
pub fn find_num_of_valid_words(words: Vec<String>, puzzles: Vec<String>) -> Vec<i32> {
    let mut result = Vec::new();

    let mut words_set = vec![0; (2 as usize).pow(27) - 1];
    for word in words {
        let mut posts = 0;
        for character in word.chars() {
            posts |= 1 << (character as i32 - 'a' as i32);
        }
        words_set[posts] += 1;
    }

    for puzzle in puzzles {
        let mut count = 0;

        let mut posts = 0;
        let mut characters = puzzle.chars();
        let head = characters.nth(0).unwrap();
        for character in characters {
            posts |= 1 << (character as i32 - 'a' as i32)
        }
        let mut sub = posts;
        loop {
            sub = (sub - 1) & posts;
            count += words_set[sub + (1 << (head as i32 - 'a' as i32))];
            if sub == posts {
                break;
            }
        }
        result.push(count);
    }

    result
}

/// 1184
pub fn distance_between_bus_stops(distance: Vec<i32>, start: i32, destination: i32) -> i32 {
    if start == destination {
        return 0;
    }
    fn partial_sum(dis: &Vec<i32>, start: usize, destination: usize) -> i32 {
        dis[start..destination].iter().fold(0, |acc, cur| acc + cur)
    }
    let (start, destination) = (start as usize, destination as usize);
    if start == destination {
        return 0;
    } else if start < destination {
        let in_order = partial_sum(&distance, start, destination);
        let reverse = partial_sum(&distance, 0, start)
            + partial_sum(&distance, destination, distance.len() - 1)
            + distance[distance.len() - 1];
        in_order.min(reverse)
    } else {
        let in_order = partial_sum(&distance, destination, start);
        let reverse = partial_sum(&distance, 0, destination)
            + partial_sum(&distance, start, distance.len() - 1)
            + distance[distance.len() - 1];
        in_order.min(reverse)
    }
}

/// 1185
pub fn day_of_the_week(day: i32, month: i32, year: i32) -> String {
    let days = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];

    let (mut y, mut m, d) = (year, month, day);
    if m < 3 {
        m += 12;
        y -= 1;
    }
    let (c, y) = (y / 100, y % 100);
    let mut week = y + y / 4 + c / 4 - 2 * c + 26 * (m + 1) / 10 + d - 1;
    week = (week % 7 + 7) % 7;

    return days[week as usize].to_string();
}

/// 1186
pub fn maximum_sum(arr: Vec<i32>) -> i32 {
    let (mut res, mut dp_0, mut dp_1) = (arr[0], arr[0], 0);
    for i in 1..arr.len() {
        dp_1 = (dp_1 + arr[i]).max(dp_0);
        dp_0 = (dp_0).max(0) + arr[i];
        res = res.max(dp_0).max(dp_1);
    }
    res
}

/// 1187
pub fn make_array_increasing(mut arr1: Vec<i32>, mut arr2: Vec<i32>) -> i32 {
    const MAX: i32 = 1000000001;
    arr2.sort_unstable();
    arr2.dedup();
    arr1.push(MAX);
    arr1.insert(0, -1);

    let mut dp = vec![MAX; arr1.len()];
    dp[0] = 0;

    for i in 1..arr1.len() {
        let j = match arr2.binary_search(&arr1[i]) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };

        for k in 1..=j.min(i - 1) {
            if arr1[i - k - 1] < arr2[j - k] {
                dp[i] = dp[i].min(dp[i - k - 1] + k as i32);
            }
        }

        if arr1[i - 1] < arr1[i] {
            dp[i] = dp[i].min(dp[i - 1]);
        }
    }

    let res = dp[arr1.len() - 1];

    if res >= MAX { -1 } else { res }
}

/// 1189
pub fn max_number_of_balloons(text: String) -> i32 {
    let mut single = HashMap::new();
    let mut double = HashMap::new();
    text.chars().for_each(|c| match c {
        'b' | 'a' | 'n' => {
            let n = single.entry(c).or_insert(0);
            *n = *n + 1;
        }
        'l' | 'o' => {
            let n = double.entry(c).or_insert(0);
            *n = *n + 1;
        }
        _ => {}
    });
    if single.len() != 3 || double.len() != 2 {
        return 0;
    }
    *single
        .values()
        .min()
        .unwrap_or(&0)
        .min(&(double.values().min().unwrap_or(&0) / 2))
}

/// 1190
pub fn reverse_parentheses(s: String) -> String {
    let mut stack = Vec::new();
    s.as_bytes().iter().fold(String::new(), |mut res, &c| {
        match (stack.len(), c) {
            (0, c) if c != b'(' => res.push(c as char),
            (_, b')') => {
                let mut temp = vec![];
                loop {
                    let c = stack.pop().unwrap();
                    if c == b'(' {
                        break;
                    }
                    temp.push(c);
                }
                match stack.len() {
                    0 => res.push_str(&String::from_utf8(temp).unwrap()),
                    _ => stack.append(&mut temp),
                }
            }
            _ => stack.push(c),
        }
        res
    })
}

/// 1191
pub fn k_concatenation_max_sum(arr: Vec<i32>, k: i32) -> i32 {
    fn max_sub_array(nums: &[i32], repeat: usize) -> i32 {
        let n = nums.len();
        let mut ans = 0;
        let mut f = 0;
        for i in 0..n * repeat {
            f = f.max(0) + nums[i % n];
            ans = ans.max(f);
        }
        ans
    }
    if k == 1 {
        return max_sub_array(&arr, 1);
    }
    let mut ans = max_sub_array(&arr, 2) as i64;
    let s = arr.into_iter().sum::<i32>();
    ans += s.max(0) as i64 * (k - 2) as i64;
    (ans % 1_000_000_007) as i32
}

/// 1192
pub fn critical_connections(n: i32, connections: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let n = n as usize;
    let mut g = vec![vec![]; n];

    for e in connections {
        let (a, b) = (e[0] as usize, e[1] as usize);
        g[a].push(b);
        g[b].push(a);
    }

    fn tarjan_search(
        node: usize,
        parent: usize,
        g: &Vec<Vec<usize>>,
        dnf: &mut Vec<i32>,
        low: &mut Vec<i32>,
        time: i32,
        res: &mut Vec<Vec<i32>>,
    ) {
        dnf[node] = time;
        low[node] = time;

        for &child in &g[node] {
            if child == parent {
                continue;
            }

            if dnf[child] == -1 {
                tarjan_search(child, node, g, dnf, low, time + 1, res);
                low[node] = low[node].min(low[child]);
                // get critical edge. aka. bridge.
                if dnf[node] < low[child] {
                    res.push(vec![node as i32, child as i32]);
                }
            } else {
                low[node] = low[node].min(dnf[child]);
            }
        }
    }

    let mut dnf = vec![-1; n];
    let mut low = vec![0; n];
    let mut res = vec![];
    tarjan_search(0, n, &g, &mut dnf, &mut low, 0, &mut res);

    res
}

#[test]
fn test_1200() {
    reverse_parentheses("(ed(et(oc))el)".to_string());
}
