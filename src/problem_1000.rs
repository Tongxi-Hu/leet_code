use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
};

///901
#[derive(Default)]
struct StockSpanner {
    queue: Vec<Vec<i32>>,
}

impl StockSpanner {
    fn new() -> Self {
        Self::default()
    }

    fn next(&mut self, price: i32) -> i32 {
        let mut ret = 1;
        while !self.queue.is_empty() && self.queue[self.queue.len() - 1][0] <= price {
            ret += self.queue.pop().unwrap()[1];
        }
        self.queue.push(vec![price, ret]);
        ret
    }
}

///902
pub fn at_most_n_given_digit_set(digits: Vec<String>, mut n: i32) -> i32 {
    let digits = digits
        .into_iter()
        .map(|i| i.into_bytes()[0] - b'0')
        .collect::<Vec<u8>>();

    let len = digits.len() as i32;
    let mut last_a = 1;
    let mut result = 1;

    let mut n_len = 0;
    let mut b;
    while n != 0 {
        b = (n % 10) as u8;
        n /= 10;
        n_len += 1;

        let mut t = 0;
        for d in &digits {
            t += match d.cmp(&b) {
                Ordering::Greater => break,
                Ordering::Equal => result,
                Ordering::Less => last_a,
            };
        }
        result = t;
        last_a *= len;
    }

    result
        + if len == 1 {
            n_len as i32 - 1
        } else {
            (len.pow(n_len) - len) / (len - 1)
        }
}

///904
pub fn total_fruit(fruits: Vec<i32>) -> i32 {
    let length = fruits.len();
    if length == 0 {
        return 0;
    }
    let (mut left, mut right) = (0, 0);
    let mut total = 0;
    let mut count = HashMap::<i32, i32>::new();
    while right < length {
        let size = count.entry(fruits[right]).or_insert(0);
        *size = *size + 1;
        while count.len() > 2 {
            let size = count.entry(fruits[left]).or_insert(0);
            if *size == 1 {
                count.remove(&fruits[left]);
            } else {
                *size = *size - 1;
            }
            left = left + 1;
        }
        total = total.max(count.values().sum::<i32>());
        right = right + 1;
    }
    total
}

///905
pub fn sort_array_by_parity(nums: Vec<i32>) -> Vec<i32> {
    let (mut even, mut odd) = (vec![], vec![]);
    nums.iter().for_each(|n| {
        if n % 2 == 0 {
            even.push(*n);
        } else {
            odd.push(*n);
        }
    });
    [even, odd].concat()
}

///906
pub fn superpalindromes_in_range(left: String, right: String) -> i32 {
    fn is_palindrome(inp: i64) -> bool {
        let mut rx = 0;
        let mut x = inp;
        while x > 0 {
            rx = 10 * rx + x % 10;
            x /= 10;
        }
        rx == inp
    }
    let left = left.parse::<i64>().unwrap();
    let right = right.parse::<i64>().unwrap();
    const MAGIC: i32 = 1e5 as i32;
    let mut res = 0;
    for k in 1..MAGIC {
        let s = format!("{}", k);
        let s = s.clone() + &s.chars().rev().collect::<String>();
        let v = s.parse::<i64>().unwrap();
        let v = v * v;
        if v > right {
            break;
        }
        if v >= left && is_palindrome(v) {
            res += 1;
        }
    }
    for k in 1..MAGIC {
        let s = format!("{}", k);
        let s = s.clone() + &s.chars().rev().skip(1).collect::<String>();
        let v = s.parse::<i64>().unwrap();
        let v = v * v;
        if v > right {
            break;
        }
        if v >= left && is_palindrome(v) {
            res += 1;
        }
    }
    res
}

///907
pub fn sum_subarray_mins(arr: Vec<i32>) -> i32 {
    let n = arr.len();
    let mut left = vec![-1; n];
    let mut right = vec![n as i32; n];
    let mut stk: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        while !stk.is_empty() && arr[*stk.back().unwrap()] >= arr[i] {
            stk.pop_back();
        }
        if let Some(&top) = stk.back() {
            left[i] = top as i32;
        }
        stk.push_back(i);
    }

    stk.clear();
    for i in (0..n).rev() {
        while !stk.is_empty() && arr[*stk.back().unwrap()] > arr[i] {
            stk.pop_back();
        }
        if let Some(&top) = stk.back() {
            right[i] = top as i32;
        }
        stk.push_back(i);
    }

    const MOD: i64 = 1_000_000_007;
    let mut ans: i64 = 0;
    for i in 0..n {
        ans +=
            ((((right[i] - (i as i32)) * ((i as i32) - left[i])) as i64) * (arr[i] as i64)) % MOD;
        ans %= MOD;
    }
    ans as i32
}

///908
pub fn smallest_range_i(nums: Vec<i32>, k: i32) -> i32 {
    let (max, min) = nums
        .iter()
        .fold((i32::MIN, i32::MAX), |acc, &n| (acc.0.max(n), acc.1.min(n)));
    return if max - min > 2 * k {
        max - min - 2 * k
    } else {
        0
    };
}

///909
pub fn snakes_and_ladders(board: Vec<Vec<i32>>) -> i32 {
    let n = board.len();
    let mut vis = vec![false; n * n + 1];
    vis[1] = true;
    let mut q = vec![1];
    for step in 0.. {
        if q.is_empty() {
            break;
        }
        let tmp = q;
        q = vec![];
        for x in tmp {
            if x == n * n {
                return step;
            }
            for y in x + 1..=(x + 6).min(n * n) {
                let r = (y - 1) / n;
                let mut c = (y - 1) % n;
                if r % 2 > 0 {
                    c = n - 1 - c;
                }
                let mut nxt = board[n - 1 - r][c];
                if nxt < 0 {
                    nxt = y as i32;
                }
                let nxt = nxt as usize;
                if !vis[nxt] {
                    vis[nxt] = true;
                    q.push(nxt);
                }
            }
        }
    }
    -1
}

///910
pub fn smallest_range_ii(mut nums: Vec<i32>, k: i32) -> i32 {
    nums.sort();
    let n = nums.len();
    let mut ans = nums[n - 1] - nums[0];
    for i in (1..n).into_iter() {
        let mx = (nums[i - 1] + k).max(nums[n - 1] - k);
        let mn = (nums[0] + k).min(nums[i] - k);
        ans = ans.min(mx - mn);
    }
    ans
}

///911
struct TopVotedCandidate {
    times: Vec<i32>,
    dp: Vec<i32>,
}

impl TopVotedCandidate {
    fn new(persons: Vec<i32>, times: Vec<i32>) -> Self {
        let n = persons.len();
        let mut map: HashMap<i32, i32> = HashMap::new();
        let mut dp: Vec<i32> = vec![0; n];

        for i in 0..n {
            let count = map.entry(persons[i]).or_insert(0);
            *count += 1;

            dp[i] = if i == 0 {
                persons[i]
            } else {
                let cur_count = *count;
                let prev_person = dp[i - 1];
                let prev_count = map.get(&prev_person).map(|count| *count).unwrap();
                if cur_count >= prev_count {
                    persons[i]
                } else {
                    prev_person
                }
            };
        }

        Self { times, dp }
    }

    fn q(&self, t: i32) -> i32 {
        match self.times.binary_search(&t) {
            Ok(idx) => self.dp[idx],
            Err(idx) => self.dp[idx - 1],
        }
    }
}

///912
pub fn sort_array(nums: Vec<i32>) -> Vec<i32> {
    fn merge(arr_1: &Vec<i32>, arr_2: &Vec<i32>) -> Vec<i32> {
        let (size_1, size_2) = (arr_1.len(), arr_2.len());
        let mut res = vec![];
        let (mut i, mut j) = (0, 0);
        while i < size_1 || j < size_2 {
            if i == size_1 {
                res.push(arr_2[j]);
                j = j + 1;
            } else if j == size_2 {
                res.push(arr_1[i]);
                i = i + 1;
            } else if arr_1[i] <= arr_2[j] {
                res.push(arr_1[i]);
                i = i + 1;
            } else {
                res.push(arr_2[j]);
                j = j + 1;
            }
        }
        res
    }
    let mut ans = nums.iter().map(|&n| vec![n]).collect::<Vec<Vec<i32>>>();
    let mut size = ans.len();
    while size > 1 {
        println!("{:?}", ans);
        let mut merged_arr = vec![];
        for i in (0..size).step_by(2) {
            if i < size - 1 {
                merged_arr.push(merge(&ans[i], &ans[i + 1]));
            }
        }
        if size % 2 != 0 {
            merged_arr.push(ans.last().unwrap().clone())
        }
        ans = merged_arr;
        size = ans.len();
    }
    ans[0].to_vec()
}

///913
#[derive(Clone, PartialEq, Copy)]
enum State {
    NoFound,
    Draw,
    CatWin,
    MouseWin,
}

pub fn cat_mouse_game(graph: Vec<Vec<i32>>) -> i32 {
    fn dfs(
        dp: &mut Vec<Vec<Vec<State>>>,
        mouse: usize,
        cat: usize,
        turn: usize,
        g: &Vec<Vec<i32>>,
        n: usize,
    ) -> State {
        let is_mouse = turn % 2 == 0;
        let cur_move = if is_mouse { mouse } else { cat };
        let default_val = if is_mouse {
            State::CatWin
        } else {
            State::MouseWin
        };
        let mut val = default_val;
        for ele in g[cur_move].iter().filter(|&&x| x != 0 || is_mouse) {
            let r = if is_mouse {
                get_result(*ele as usize, cat, dp, turn + 1, n, g)
            } else {
                get_result(mouse, *ele as usize, dp, turn + 1, n, g)
            };
            if r != default_val {
                val = r;
                if r != State::Draw {
                    break;
                }
            }
        }
        val
    }

    fn get_result(
        mouse: usize,
        cat: usize,
        dp: &mut Vec<Vec<Vec<State>>>,
        turn: usize,
        n: usize,
        g: &Vec<Vec<i32>>,
    ) -> State {
        if turn == 2 * n {
            return State::Draw;
        }
        if dp[mouse][cat][turn] == State::NoFound {
            match mouse {
                0 => dp[mouse][cat][turn] = State::MouseWin,
                i if i == cat => dp[mouse][cat][turn] = State::CatWin,
                _ => dp[mouse][cat][turn] = dfs(dp, mouse, cat, turn, g, n),
            }
        }
        dp[mouse][cat][turn]
    }

    let len = graph.len();
    let mut dp = vec![vec![vec![State::NoFound; 2 * len]; len]; len];
    match get_result(1, 2, &mut dp, 0, len, &graph) {
        State::CatWin => 2,
        State::Draw => 0,
        State::MouseWin => 1,
        State::NoFound => panic!("no found"),
    }
}
