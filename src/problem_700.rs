use core::f64;
use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashMap},
    rc::Rc,
};

use crate::common::TreeNode;

/// p605
pub fn can_place_flowers(flowerbed: Vec<i32>, n: i32) -> bool {
    let m = flowerbed.len();
    let mut n = n;
    let mut i = 0;
    while i < m {
        if (i == 0 || flowerbed[i - 1] == 0)
            && flowerbed[i] == 0
            && (i == m - 1 || flowerbed[i + 1] == 0)
        {
            n -= 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    n <= 0
}

/// p606
pub fn tree2str(root: Option<Rc<RefCell<TreeNode>>>) -> String {
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>) -> Option<String> {
        if let Some(node) = root.as_ref() {
            let mut res = String::new();
            let v = node.borrow().val;
            res = res + &v.to_string();
            let left_res = dfs(&node.borrow().left);
            let right_res = dfs(&node.borrow().right);
            if let Some(left_val) = left_res.as_ref() {
                res = res + "(";
                res = res + &left_val.to_string();
                res = res + ")";
            }
            if let Some(right_val) = right_res {
                if left_res == None {
                    res = res + "()";
                }
                res = res + "(";
                res = res + &right_val.to_string();
                res = res + ")";
            }
            return Some(res);
        } else {
            None
        }
    }
    dfs(&root).unwrap_or("".to_string())
}

/// p609
pub fn find_duplicate(paths: Vec<String>) -> Vec<Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();

    for path in paths.into_iter() {
        let vals = path.split(' ').collect::<Vec<_>>();
        for &s in vals.iter().skip(1) {
            let idx = s.find('(').unwrap();
            let cur_path = vals[0].to_string() + "/" + &s[..idx];
            let cur_content = s[idx..].to_string();
            map.entry(cur_content).or_insert(Vec::new()).push(cur_path);
        }
    }

    map.into_values()
        .filter(|vals| vals.len() > 1)
        .collect::<Vec<_>>()
}

/// p611
pub fn triangle_number(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort();
    let length = nums.len();
    if length < 3 {
        return 0;
    }
    let mut count = 0;
    for i in 0..length - 2 {
        for j in i + 1..length - 1 {
            for k in j + 1..length {
                if nums[i] + nums[j] > nums[k] {
                    count = count + 1;
                }
            }
        }
    }
    count
}

/// p617
pub fn merge_trees(
    root1: Option<Rc<RefCell<TreeNode>>>,
    root2: Option<Rc<RefCell<TreeNode>>>,
) -> Option<Rc<RefCell<TreeNode>>> {
    fn dfs(
        root1: &Option<Rc<RefCell<TreeNode>>>,
        root2: &Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        match (root1.as_ref(), root2.as_ref()) {
            (None, None) => None,
            (Some(node1), Some(node2)) => {
                let mut node = TreeNode::new(node1.borrow().val + node2.borrow().val);
                node.left = dfs(&node1.borrow().left, &node2.borrow().left);
                node.right = dfs(&node1.borrow().right, &node2.borrow().right);
                Some(Rc::new(RefCell::new(node)))
            }
            (Some(node1), None) => Some(node1.clone()),
            (None, Some(node2)) => Some(node2.clone()),
        }
    }
    dfs(&root1, &root2)
}

/// p621
pub fn least_interval(tasks: Vec<char>, n: i32) -> i32 {
    let mut counts = [0; 26];
    let mut max_count = 0;
    let mut count = 0;

    for idx in tasks.iter().map(|&task| ((task as u8) - b'A') as usize) {
        counts[idx] += 1;
        if counts[idx] > max_count {
            count = 1;
            max_count = counts[idx];
        } else if counts[idx] == max_count {
            count += 1;
        }
    }

    i32::max(tasks.len() as i32, count + (max_count - 1) * (n + 1))
}

/// p622
struct MyCircularQueue {
    arr: Vec<i32>,
    rear: usize,
    curr_size: usize,
}

impl MyCircularQueue {
    fn new(k: i32) -> Self {
        MyCircularQueue {
            arr: vec![0; k as usize],
            rear: 0,
            curr_size: 0,
        }
    }

    fn en_queue(&mut self, value: i32) -> bool {
        if self.is_full() {
            false
        } else {
            self.arr[self.rear] = value;
            self.curr_size += 1;
            self.rear = (self.rear + 1) % self.arr.len();
            true
        }
    }

    fn de_queue(&mut self) -> bool {
        if self.is_empty() {
            false
        } else {
            self.curr_size -= 1;
            true
        }
    }

    fn front(&self) -> i32 {
        if self.is_empty() {
            -1
        } else {
            self.arr[(self.rear - self.curr_size + self.arr.len()) % self.arr.len()]
        }
    }

    fn rear(&self) -> i32 {
        if self.is_empty() {
            -1
        } else {
            self.arr[(self.rear - 1 + self.arr.len()) % self.arr.len()]
        }
    }

    fn is_empty(&self) -> bool {
        self.curr_size == 0
    }

    fn is_full(&self) -> bool {
        self.curr_size == self.arr.len()
    }
}

///p623
pub fn add_one_row(
    root: Option<Rc<RefCell<TreeNode>>>,
    val: i32,
    depth: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    use std::collections::VecDeque;
    if depth == 1 {
        let mut node = TreeNode::new(val);
        node.left = root;
        return Some(Rc::new(RefCell::new(node)));
    }
    let mut queue = VecDeque::new();
    queue.push_back(root.clone());
    for i in 0..depth - 2 {
        let size = queue.len();
        for j in 0..size {
            if let Some(node) = queue.pop_front() {
                if node.as_ref().unwrap().borrow().left.is_some() {
                    queue.push_back(node.as_ref().unwrap().borrow().left.clone());
                }
                if node.as_ref().unwrap().borrow().right.is_some() {
                    queue.push_back(node.as_ref().unwrap().borrow().right.clone());
                }
            }
        }
    }
    while !queue.is_empty() {
        if let Some(node) = queue.pop_front() {
            let mut new_node = TreeNode::new(val);
            new_node.left = node.as_ref().unwrap().borrow().left.clone();
            node.as_ref().unwrap().borrow_mut().left = Some(Rc::new(RefCell::new(new_node)));
            let mut new_node = TreeNode::new(val);
            new_node.right = node.as_ref().unwrap().borrow().right.clone();
            node.as_ref().unwrap().borrow_mut().right = Some(Rc::new(RefCell::new(new_node)));
        }
    }
    root
}

/// p624
pub fn max_distance(arrays: Vec<Vec<i32>>) -> i32 {
    let mut res = 0;
    let mut min_val = arrays[0][0];
    let mut max_val = arrays[0][arrays[0].len() - 1];
    for i in 1..arrays.len() {
        let n = arrays[i].len();
        res = res.max((arrays[i][n - 1] - min_val).abs());
        res = res.max((max_val - arrays[i][0]).abs());
        min_val = min_val.min(arrays[i][0]);
        max_val = max_val.max(arrays[i][n - 1]);
    }
    res
}

/// p628
pub fn maximum_product(mut nums: Vec<i32>) -> i32 {
    let n = nums.len();
    nums.sort_unstable();

    let mut max_num = nums[n - 3] * nums[n - 2] * nums[n - 1];
    if nums[1] < 0 {
        max_num = i32::max(max_num, nums[0] * nums[1] * nums[n - 1]);
    }

    max_num
}

/// p629
const MOD: i64 = 1_0000_0000_7;

pub fn k_inverse_pairs(n: i32, k: i32) -> i32 {
    let (n, k) = (n as usize, k as usize);

    let mut dp = vec![vec![0_i64; k + 1]; n + 1];

    for i in 0..=n {
        dp[i][0] = 1;
    }

    for i in 1..=n {
        for j in 1..=k {
            dp[i][j] = if j >= i {
                (dp[i][j - 1] + MOD - dp[i - 1][j - i] + dp[i - 1][j]) % MOD
            } else {
                (dp[i][j - 1] + dp[i - 1][j]) % MOD
            };
        }
    }

    dp[n][k] as i32
}

/// p630
pub fn schedule_course(courses: Vec<Vec<i32>>) -> i32 {
    let mut courses: Vec<_> = courses
        .into_iter()
        .filter(|course| course[0] <= course[1])
        .collect();
    courses.sort_by_key(|c| c[1]);

    if courses.is_empty() {
        return 0;
    }

    let mut cur_time = 0;
    let mut pq = BinaryHeap::new();

    for course in courses {
        let (duration, ddl) = (course[0], course[1]);
        if cur_time + duration <= ddl {
            cur_time += duration;
            pq.push(duration);
        } else {
            if let Some(&d) = pq.peek() {
                if duration < d {
                    cur_time = cur_time - d + duration;
                    pq.pop();
                    pq.push(duration);
                }
            }
        }
    }

    pq.len() as i32
}

/// p632
pub fn smallest_range(nums: Vec<Vec<i32>>) -> Vec<i32> {
    let mut range_left = 0;
    let mut range_right = i32::MAX;
    let size = nums.len();
    let mut next = vec![0; size];
    let mut max_value = i32::MIN;
    let mut pq = BinaryHeap::new();

    for i in 0..size {
        max_value = max_value.max(nums[i][0]);
        pq.push(std::cmp::Reverse((nums[i][0], i)));
    }

    while let Some(std::cmp::Reverse((min_value, row))) = pq.pop() {
        if max_value - min_value < range_right - range_left {
            range_left = min_value;
            range_right = max_value;
        }
        if next[row] == nums[row].len() - 1 {
            break;
        }
        next[row] += 1;
        max_value = max_value.max(nums[row][next[row]]);
        pq.push(std::cmp::Reverse((nums[row][next[row]], row)));
    }

    vec![range_left, range_right]
}

/// p633
pub fn judge_square_sum(c: i32) -> bool {
    let (mut a, mut b) = (0, (c as f64).sqrt() as i32);
    while a <= b {
        if a * a == c - b * b {
            return true;
        }
        if a * a < c - b * b {
            a += 1
        } else {
            b -= 1;
        }
    }
    false
}

/// p636
pub fn exclusive_time(n: i32, logs: Vec<String>) -> Vec<i32> {
    let (mut ret, mut queue) = (vec![0; n as usize], Vec::new());
    for log in logs {
        let arr = log.split(":").collect::<Vec<_>>();
        let (num, t) = (
            arr[0].parse::<i32>().unwrap(),
            arr[2].parse::<i32>().unwrap(),
        );
        if "start" == arr[1] {
            queue.push((num, t))
        } else {
            if let Some(last) = queue.pop() {
                let cost = t - last.1 + 1;
                ret[num as usize] += cost;
                queue.last().map_or((), |last| ret[last.0 as usize] -= cost);
            }
        }
    }
    ret
}

/// p637
pub fn average_of_levels(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<f64> {
    if let Some(node) = root.as_ref() {
        let mut current_level: Vec<Rc<RefCell<TreeNode>>> = vec![node.clone()];
        let mut average: Vec<f64> = vec![];
        loop {
            let length: usize = current_level.len();
            let mut current_sum: f64 = 0.0;
            for i in 0..length {
                let node = current_level.remove(0).clone();
                current_sum = (node.borrow().val as f64) + current_sum;
                if let Some(left) = node.borrow().left.as_ref() {
                    current_level.push(left.clone())
                }
                if let Some(right) = node.borrow().right.as_ref() {
                    current_level.push(right.clone());
                }
            }
            average.push(current_sum / (length as f64));
            if current_level.len() == 0 {
                break;
            }
        }
        return average;
    } else {
        vec![]
    }
}

/// p638
pub fn shopping_offers(price: Vec<i32>, special: Vec<Vec<i32>>, needs: Vec<i32>) -> i32 {
    let n: usize = price.len();
    fn dfs(price: &Vec<i32>, special: &Vec<Vec<i32>>, needs: &Vec<i32>, i: usize, n: usize) -> i32 {
        let mut ans = 0;
        for j in 0..n {
            ans += needs[j] * price[j];
        }
        if special.is_empty() {
            return ans;
        }
        let (curr, mut next_needs) = (&special[i], vec![]);
        for j in 0..n {
            if curr[j] > needs[j] {
                break;
            }
            next_needs.push(needs[j] - curr[j]);
        }

        if next_needs.len() == n {
            ans = ans.min(curr[n] + dfs(price, special, &next_needs, i, n));
        }
        if i + 1 < special.len() {
            ans = ans.min(dfs(price, special, needs, i + 1, n));
        }
        ans
    }

    let mut filter_specials = Vec::new();
    for s in special {
        let (mut total_cnt, mut total_price) = (0, 0);
        for i in 0..n {
            total_cnt += s[i];
            total_price += price[i] * s[i];
        }
        if total_cnt > 0 && total_price > s[s.len() - 1] {
            filter_specials.push(s);
        }
    }

    dfs(&price, &filter_specials, &needs, 0, n)
}

/// p639
pub fn num_decodings(s: String) -> i32 {
    let m = 1_000_000_007;
    let (mut a, mut b, mut c) = (0i64, 1i64, 0i64);
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        c = b * check1digit(bytes[i]) % m;
        if i > 0 {
            c += a * check2digits(bytes[i - 1], bytes[i]) % m;
        }
        c %= m;
        a = b;
        b = c;
    }
    c as i32
}

fn check1digit(b: u8) -> i64 {
    match b {
        b'*' => 9,
        b'0' => 0,
        _ => 1,
    }
}

fn check2digits(b1: u8, b2: u8) -> i64 {
    match (b1, b2) {
        (b'*', b'*') => 15,
        (b'*', x) => {
            if x > b'6' {
                1
            } else {
                2
            }
        }
        (x, b'*') => match x {
            b'1' => 9,
            b'2' => 6,
            _ => 0,
        },
        (b'1', _) => 1,
        (b'2', b'0'..=b'6') => 1,
        _ => 0,
    }
}

/// p640
pub fn solve_equation(equation: String) -> String {
    let arr = equation.split("=").collect::<Vec<_>>();

    fn parse_expression(expression: &str) -> (i32, i32) {
        let expression = expression.replace("+x", "+1x").replace("-x", "-1x");
        let expression = if expression.chars().nth(0) == Some('x') {
            format!("+1{}", expression)
        } else {
            expression
        };
        let expression = expression.replace("-", "+-");
        let split_arr = expression
            .split("+")
            .filter(|x| x.len() > 0)
            .collect::<Vec<_>>();

        let prefix = split_arr
            .iter()
            .filter(|x| x.chars().rev().nth(0) == Some('x'))
            .map(|x| x.trim_end_matches("x"))
            .map(|x| x.parse::<i32>().unwrap())
            .sum::<i32>();
        let value = split_arr
            .iter()
            .filter(|x| x.chars().rev().nth(0) != Some('x'))
            .map(|x| x.parse::<i32>().unwrap())
            .sum::<i32>();
        return (prefix, value);
    }

    let (mut left, mut right) = (parse_expression(arr[0]), parse_expression(arr[1]));
    left.0 -= right.0;
    right.1 -= left.1;
    match (left.0, right.1) {
        (0, 0) => "Infinite solutions".to_string(),
        (0, _) => "No solution".to_string(),
        (x, y) => format!("x={}", y / x),
    }
}

/// p643
pub fn find_max_average(nums: Vec<i32>, k: i32) -> f64 {
    let k = k as usize;
    let mut sum: i32 = nums[0..k].iter().sum();
    let mut max = sum;
    for i in k..nums.len() {
        sum = sum - nums[i - k] + nums[i];
        max = max.max(sum);
    }
    max as f64 / k as f64
}

/// p645
pub fn find_error_nums(nums: Vec<i32>) -> Vec<i32> {
    let mut cnt = vec![0; nums.len() + 1];
    for n in nums {
        cnt[n as usize] += 1;
    }
    let mut ans = vec![0; 2];
    for (i, &c) in cnt.iter().enumerate().skip(1) {
        if c == 2 {
            ans[0] = i as i32;
        } else if c == 0 {
            ans[1] = i as i32;
        }
    }
    ans
}

/// p646
pub fn find_longest_chain(pairs: Vec<Vec<i32>>) -> i32 {
    let mut pairs = pairs;
    pairs.sort_by(|a, b| a[1].cmp(&b[1]));
    let mut cur = i32::MIN;
    let mut count = 0;
    pairs.iter().for_each(|p| {
        if cur < p[0] {
            cur = p[1];
            count = count + 1;
        }
    });
    count
}

/// p647
pub fn count_substrings(s: String) -> i32 {
    let mut ans = 0;
    let s = s.chars().collect::<Vec<_>>();
    let mut dp = vec![vec![false; s.len()]; s.len()];
    for i in (0..s.len()).rev() {
        dp[i][i] = true;
        ans += 1;
        for j in i + 1..s.len() {
            dp[i][j] = s[i] == s[j] && if i + 1 == j { true } else { dp[i + 1][j - 1] };
            ans += if dp[i][j] { 1 } else { 0 };
        }
    }
    ans
}

/// p648
pub fn replace_words(dictionary: Vec<String>, sentence: String) -> String {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let mut dictionary = dictionary;
    dictionary.sort_by(|a, b| a.len().cmp(&b.len()));
    let mut ans: Vec<&str> = vec![];
    for word in words {
        let mut modified = false;
        'b: for root in &dictionary {
            if word.starts_with(root) {
                ans.push(root);
                modified = true;
                break 'b;
            }
        }
        if modified == false {
            ans.push(word)
        }
    }
    ans.join(&" ")
}

/// p649
pub fn predict_party_victory(mut senate: String) -> String {
    let (mut r_alive, mut d_alive) = (true, true);
    let mut mark: i32 = 0;
    while r_alive && d_alive {
        (r_alive, d_alive) = (false, false);
        for c in unsafe { senate.as_bytes_mut() } {
            match c {
                b'0' => {}
                b'R' => {
                    r_alive = true;
                    if mark < 0 {
                        *c = b'0';
                    }
                    mark += 1;
                }
                b'D' => {
                    d_alive = true;
                    if mark > 0 {
                        *c = b'0';
                    }
                    mark -= 1;
                }
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
        }
    }
    match mark.cmp(&0) {
        std::cmp::Ordering::Greater => "Radiant".to_owned(),
        std::cmp::Ordering::Less => "Dire".to_owned(),
        std::cmp::Ordering::Equal => unreachable!(),
    }
}

/// p650
pub fn min_steps(n: i32) -> i32 {
    let n = n as usize;
    let mut dp: Vec<usize> = vec![usize::MAX; n + 1 as usize];
    dp[1] = 0;
    for i in 2..=n {
        for j in 1..i {
            if i % j == 0 {
                dp[i] = dp[i].min(dp[j] + i / j)
            }
        }
    }
    dp[n] as i32
}
