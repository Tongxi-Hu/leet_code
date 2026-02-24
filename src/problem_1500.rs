use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BTreeMap, HashSet, VecDeque},
    i32,
    rc::Rc,
};

use crate::common::TreeNode;

/// 01
pub fn check_overlap(
    radius: i32,
    x_center: i32,
    y_center: i32,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
) -> bool {
    let dx = if x1 > x_center {
        x1 - x_center
    } else if x_center > x2 {
        x_center - x2
    } else {
        0
    };
    let dy = if y1 > y_center {
        y1 - y_center
    } else if y_center > y2 {
        y_center - y2
    } else {
        0
    };
    return dx * dx + dy * dy <= radius * radius;
}

/// 02
pub fn max_satisfaction(mut satisfaction: Vec<i32>) -> i32 {
    satisfaction.sort();
    let n = satisfaction.len();
    let (mut dp, mut res) = (vec![vec![0; n + 1]; n + 1], 0);
    for i in 1..=n {
        for j in 1..=i {
            dp[i][j] = dp[i - 1][j - 1] + satisfaction[i - 1] * j as i32;
            if j < i {
                dp[i][j] = dp[i][j].max(dp[i - 1][j]);
            }
            res = res.max(dp[i][j]);
        }
    }
    res
}

/// 03
pub fn min_subsequence(mut nums: Vec<i32>) -> Vec<i32> {
    nums.sort_by(|a, b| b.cmp(a));
    let (sum, mut cur) = (nums.iter().sum::<i32>(), 0);
    for i in 0..nums.len() {
        cur = cur + nums[i];
        if cur > sum - cur {
            return nums[0..=i].to_vec();
        }
    }
    return vec![];
}

/// 04
pub fn num_steps(mut s: String) -> i32 {
    let mut step = 0;
    loop {
        if s.len() == 1 {
            break;
        }
        step += 1;
        if s.ends_with("1") {
            let mut length = 1;
            s.remove(s.len() - 1);
            while s.len() != 0 && s.ends_with("1") {
                length += 1;
                s.remove(s.len() - 1);
            }
            if s.len() != 0 {
                s.remove(s.len() - 1);
            }
            s.push('1');
            for _i in 0..length {
                s.push('0');
            }
        } else {
            s.remove(s.len() - 1);
        }
    }
    step
}

/// 05
pub fn longest_diverse_string(mut a: i32, mut b: i32, mut c: i32) -> String {
    let (mut a_use, mut b_use, mut c_use) = (0, 0, 0);
    let total = a + b + c;
    let mut list = Vec::new();

    for _ in 0..total {
        if (a >= b && a >= c && a_use != 2) || (b_use == 2 && a > 0) || (c_use == 2 && a > 0) {
            list.push("a");
            a_use += 1;
            a -= 1;
            b_use = 0;
            c_use = 0;
        } else if (b >= a && b >= c && b_use != 2) || (a_use == 2 && b > 0) || (c_use == 2 && b > 0)
        {
            list.push("b");
            b_use += 1;
            b -= 1;
            a_use = 0;
            c_use = 0;
        } else if (c >= a && c >= b && c_use != 2) || (a_use == 2 && c > 0) || (b_use == 2 && c > 0)
        {
            list.push("c");
            c_use += 1;
            c -= 1;
            a_use = 0;
            b_use = 0;
        }
    }
    list.into_iter().collect::<String>()
}

/// 06
pub fn stone_game_iii(st: Vec<i32>) -> String {
    let n = st.len();
    let mut dp = vec![i32::MIN; n];
    dp[n - 1] = st[n - 1];
    if n >= 2 {
        dp[n - 2] = st[n - 2] + dp[n - 1].abs();
    }
    if n >= 3 {
        dp[n - 3] = st[n - 3] + dp[n - 2].abs();
        for i in (0..(n - 3)).rev() {
            dp[i] = (st[i] - dp[i + 1])
                .max(st[i] + st[i + 1] - dp[i + 2])
                .max(st[i] + st[i + 1] + st[i + 2] - dp[i + 3]);
        }
    }
    if dp[0] > 0 {
        "Alice"
    } else if dp[0] == 0 {
        "Tie"
    } else {
        "Bob"
    }
    .to_string()
}

/// 08
pub fn string_matching(mut words: Vec<String>) -> Vec<String> {
    words.sort_by(|a, b| a.len().cmp(&b.len()));
    let mut ans = vec![];
    for i in 0..words.len() - 1 {
        for j in i + 1..words.len() {
            if words[j].contains(&words[i]) {
                ans.push(words[i].clone());
                break;
            }
        }
    }
    ans
}

/// 09
pub fn process_queries(queries: Vec<i32>, m: i32) -> Vec<i32> {
    let (mut p, mut ans) = ((1..=m).collect::<Vec<i32>>(), vec![]);
    queries.iter().for_each(|&q| {
        let pos = p.iter().position(|&e| e == q).unwrap();
        ans.push(pos as i32);
        p.remove(pos);
        p.insert(0, q);
    });
    ans
}

/// 10
pub fn entity_parser(text: String) -> String {
    let (text, mut ans) = (text.chars().collect::<Vec<char>>(), "".to_string());
    let (mut i, n) = (0, text.len());
    while i < n {
        if i + 5 < n
            && text[i] == '&'
            && text[i + 1] == 'q'
            && text[i + 2] == 'u'
            && text[i + 3] == 'o'
            && text[i + 4] == 't'
            && text[i + 5] == ';'
        {
            ans += &'\"'.to_string();
            i += 6;
        } else if i + 5 < n
            && text[i] == '&'
            && text[i + 1] == 'a'
            && text[i + 2] == 'p'
            && text[i + 3] == 'o'
            && text[i + 4] == 's'
            && text[i + 5] == ';'
        {
            ans += &'\''.to_string();
            i += 6;
        } else if i + 4 < n
            && text[i] == '&'
            && text[i + 1] == 'a'
            && text[i + 2] == 'm'
            && text[i + 3] == 'p'
            && text[i + 4] == ';'
        {
            ans += &'&'.to_string();
            i += 5;
        } else if i + 3 < n
            && text[i] == '&'
            && text[i + 1] == 'g'
            && text[i + 2] == 't'
            && text[i + 3] == ';'
        {
            ans += &'>'.to_string();
            i += 4;
        } else if i + 3 < n
            && text[i] == '&'
            && text[i + 1] == 'l'
            && text[i + 2] == 't'
            && text[i + 3] == ';'
        {
            ans += &'<'.to_string();
            i += 4;
        } else if i + 5 < n
            && text[i] == '&'
            && text[i + 1] == 'f'
            && text[i + 2] == 'r'
            && text[i + 3] == 'a'
            && text[i + 4] == 's'
            && text[i + 5] == 'l'
            && text[i + 6] == ';'
        {
            ans += &'/'.to_string();
            i += 7;
        } else {
            ans += &text[i].to_string();
            i += 1;
        }
    }
    return ans;
}

/// 11
pub fn num_of_ways(n: i32) -> i32 {
    let mod_val: i64 = 1000000007;
    let mut fi0: i64 = 6;
    let mut fi1: i64 = 6;

    for _ in 2..=n {
        let new_fi0 = (2 * fi0 + 2 * fi1) % mod_val;
        let new_fi1 = (2 * fi0 + 3 * fi1) % mod_val;
        fi0 = new_fi0;
        fi1 = new_fi1;
    }

    ((fi0 + fi1) % mod_val) as i32
}

/// 13
pub fn min_start_value(nums: Vec<i32>) -> i32 {
    1 - nums
        .iter()
        .fold((0, 0), |acc, cur| {
            let sum = acc.0 + cur;
            let min = acc.1.min(sum);
            (sum, min)
        })
        .1
}

/// 14
pub fn find_min_fibonacci_numbers(k: i32) -> i32 {
    let mut fibo = vec![1, 1];
    while *fibo.last().unwrap() < k {
        fibo.push(fibo[fibo.len() - 1] + fibo[fibo.len() - 2]);
    }
    let (mut remain, mut pointer, mut cnt) = (k, fibo.len() - 1, 0);
    while remain > 0 {
        if remain >= fibo[pointer] {
            remain = remain - fibo[pointer];
            cnt = cnt + 1;
        }
        pointer = pointer - 1;
    }
    cnt
}

/// 15
pub fn get_happy_string(n: i32, k: i32) -> String {
    let (chars, mut cur, mut total) = (vec!['a', 'b', 'c'], vec![], vec![]);
    fn dfs(chars: &Vec<char>, cur: &mut Vec<char>, total: &mut Vec<String>, n: usize) {
        let last_char = *cur.last().unwrap_or(&'_');
        chars.iter().filter(|c| **c != last_char).for_each(|&c| {
            cur.push(c);
            if cur.len() == n {
                total.push(cur.iter().fold("".to_string(), |mut acc, &cur| {
                    acc.push(cur);
                    acc
                }));
            } else {
                dfs(chars, cur, total, n);
            }
            cur.pop();
        });
    }
    dfs(&chars, &mut cur, &mut total, n as usize);
    total
        .get((k - 1) as usize)
        .unwrap_or(&"".to_string())
        .to_string()
}

/// 17
pub fn reformat(s: String) -> String {
    let mut chars = s
        .chars()
        .into_iter()
        .fold((vec![], vec![]), |mut acc, cur| {
            match cur {
                'a'..='z' => acc.0.push(cur),
                '0'..='9' => acc.1.push(cur),
                _ => (),
            }
            (acc.0, acc.1)
        });
    let mut ans = "".to_string();
    if chars.0.len().abs_diff(chars.1.len()) > 1 {
        return ans;
    } else if chars.0.len() > chars.1.len() {
        ans.push(chars.0.pop().unwrap());
        while chars.0.len() > 0 {
            ans.push(chars.1.pop().unwrap());
            ans.push(chars.0.pop().unwrap());
        }
    } else if chars.0.len() < chars.1.len() {
        ans.push(chars.1.pop().unwrap());
        while chars.1.len() > 0 {
            ans.push(chars.0.pop().unwrap());
            ans.push(chars.1.pop().unwrap());
        }
    } else {
        while chars.1.len() > 0 {
            ans.push(chars.0.pop().unwrap());
            ans.push(chars.1.pop().unwrap());
        }
    }
    ans
}

/// 18
pub fn display_table(orders: Vec<Vec<String>>) -> Vec<Vec<String>> {
    use std::str::FromStr;

    let mut food_map = BTreeMap::new();
    orders.iter().for_each(|v| {
        food_map.entry(&v[2]).or_insert(0);
    });

    food_map
        .iter_mut()
        .enumerate()
        .for_each(|(i, (_, x))| *x = i);

    let mut order_map = BTreeMap::new();
    orders.iter().for_each(|v| {
        order_map
            .entry(i32::from_str(&v[1]).unwrap())
            .or_insert_with(|| vec![0; food_map.len()])[food_map[&v[2]]] += 1;
    });

    let mut title = vec!["Table".to_string()];
    title.append(
        &mut food_map
            .iter()
            .map(|(&name, _)| name.to_string())
            .collect::<Vec<String>>(),
    );

    let mut result = vec![title];
    order_map.iter().for_each(|(id, v)| {
        let mut order = vec![id.to_string()];
        order.append(&mut v.iter().map(|x| x.to_string()).collect::<Vec<String>>());
        result.push(order);
    });
    result
}

/// 19
pub fn min_number_of_frogs(croak_of_frogs: String) -> i32 {
    let (mut container, mut valid) = (vec![], true);
    croak_of_frogs.chars().for_each(|c| match c {
        'c' => {
            if let Some((i, _)) = container.iter().enumerate().find(|(_, c)| **c == 'k') {
                container[i] = 'c';
            } else {
                container.push('c');
            }
        }
        'r' => {
            if let Some((i, _)) = container.iter().enumerate().find(|(_, c)| **c == 'c') {
                container[i] = 'r';
            } else {
                valid = false;
            }
        }
        'o' => {
            if let Some((i, _)) = container.iter().enumerate().find(|(_, c)| **c == 'r') {
                container[i] = 'o';
            } else {
                valid = false;
            }
        }
        'a' => {
            if let Some((i, _)) = container.iter().enumerate().find(|(_, c)| **c == 'o') {
                container[i] = 'a';
            } else {
                valid = false;
            }
        }
        'k' => {
            if let Some((i, _)) = container.iter().enumerate().find(|(_, c)| **c == 'a') {
                container[i] = 'k';
            } else {
                valid = false;
            }
        }
        _ => (),
    });
    valid = valid && container.iter().all(|c| *c == 'k');
    return if valid { container.len() as i32 } else { -1 };
}

/// 20
pub fn num_of_arrays(n: i32, m: i32, k: i32) -> i32 {
    let mood = 1000000007;
    let (n, m, k) = (n as usize, m as usize, k as usize);
    let mut dp = vec![vec![vec![0; 55]; 105]; 105];
    for j in 1..m + 1 {
        dp[1][j][1] = 1;
    }
    for i in 1..n + 1 {
        for j in 1..m + 1 {
            for t in 1..k + 1 {
                for _ in 1..j + 1 {
                    dp[i + 1][j][t] += dp[i][j][t];
                    dp[i + 1][j][t] %= mood;
                }
                for jj in j + 1..m + 1 {
                    dp[i + 1][jj][t + 1] += dp[i][j][t];
                    dp[i + 1][jj][t + 1] %= mood;
                }
            }
        }
    }

    let mut result = 0;
    for i in 1..m + 1 {
        result += dp[n][i][k];
        result %= mood;
    }
    result
}

/// 22
pub fn max_score(s: String) -> i32 {
    let chars = s.chars().collect::<Vec<char>>();
    let (mut score, mut ans) = (chars.iter().filter(|&&c| c == '1').count(), 0);

    chars[0..chars.len() - 1].iter().for_each(|&c| {
        if c == '0' {
            score = score + 1;
        } else {
            score = score - 1;
        }
        ans = ans.max(score);
    });
    ans as i32
}

/// 23
pub fn max_score_2(card_points: Vec<i32>, k: i32) -> i32 {
    let (k, size) = (k as usize, card_points.len());
    let mut sum = card_points[0..k].iter().sum::<i32>();
    let mut max = sum;
    for i in (0..=k - 1).rev() {
        sum = sum - card_points[i];
        sum = sum + card_points[size - k + i];
        max = max.max(sum);
    }
    max
}

/// 24
pub fn find_diagonal_order(nums: Vec<Vec<i32>>) -> Vec<i32> {
    let mut intermediate_len = 0;
    for row in 0..nums.len() {
        intermediate_len = intermediate_len.max(row + nums[row].len());
    }
    let mut intermediate_nums = vec![vec![]; intermediate_len];
    for (row, line) in nums.iter().enumerate() {
        for (col, &item) in line.iter().enumerate() {
            intermediate_nums[row + col].push(item);
        }
    }
    intermediate_nums
        .into_iter()
        .flat_map(|row| row.into_iter().rev())
        .collect()
}

/// 31
pub fn kids_with_candies(candies: Vec<i32>, extra_candies: i32) -> Vec<bool> {
    let max = candies.iter().max().unwrap();
    candies
        .iter()
        .map(|c| {
            if c + extra_candies >= *max {
                true
            } else {
                false
            }
        })
        .collect()
}

/// 32
pub fn max_diff(num: i32) -> i32 {
    fn replace(s: &str, x: char, y: char) -> String {
        s.chars().map(|c| if c == x { y } else { c }).collect()
    }
    let mut min_num = num.to_string();
    let mut max_num = num.to_string();
    for digit in max_num.chars() {
        if digit != '9' {
            max_num = replace(&max_num, digit, '9');
            break;
        }
    }
    for (i, digit) in min_num.chars().enumerate() {
        if i == 0 {
            if digit != '1' {
                min_num = replace(&min_num, digit, '1');
                break;
            }
        } else {
            if digit != '0' && digit != min_num.chars().nth(0).unwrap() {
                min_num = replace(&min_num, digit, '0');
                break;
            }
        }
    }

    max_num.parse::<i32>().unwrap() - min_num.parse::<i32>().unwrap()
}

/// 33
pub fn check_if_can_break(s1: String, s2: String) -> bool {
    let (mut s1_c, mut s2_c) = (
        s1.chars().collect::<Vec<char>>(),
        s2.chars().collect::<Vec<char>>(),
    );
    s1_c.sort();
    s2_c.sort();
    s1_c.iter().zip(s2_c.iter()).all(|(a, b)| a >= b)
        || s2_c.iter().zip(s1_c.iter()).all(|(a, b)| a >= b)
}

/// 36
pub fn dest_city(paths: Vec<Vec<String>>) -> String {
    let (mut starts, mut ends) = (HashSet::new(), HashSet::new());
    paths.iter().for_each(|p| {
        let (start, end) = (p[0].clone(), p[1].clone());
        if !ends.remove(&start) {
            starts.insert(start);
        }
        if !starts.remove(&end) {
            ends.insert(end);
        }
    });
    ends.into_iter().collect::<Vec<String>>().pop().unwrap()
}

/// 37
pub fn k_length_apart(nums: Vec<i32>, k: i32) -> bool {
    nums.iter()
        .enumerate()
        .filter(|n| *n.1 == 1)
        .fold((i32::MAX, i32::MIN), |mut acc, cur| {
            if acc.1 == i32::MIN {
                acc.1 = cur.0 as i32;
            } else {
                let gap = cur.0 as i32 - acc.1;
                acc.0 = acc.0.min(gap);
                acc.1 = cur.0 as i32;
            }
            (acc.0, acc.1)
        })
        .0
        > k
}

/// 38
pub fn longest_subarray(nums: Vec<i32>, limit: i32) -> i32 {
    let mut l = 0;
    let mut min_q = VecDeque::new();
    let mut max_q = VecDeque::new();
    for num in &nums {
        while max_q.back().map_or(false, |x| x < num) {
            max_q.pop_back();
        }
        while min_q.back().map_or(false, |x| x > num) {
            min_q.pop_back();
        }
        max_q.push_back(*num);
        min_q.push_back(*num);
        if max_q[0] - min_q[0] > limit {
            if nums[l] == max_q[0] {
                max_q.pop_front();
            }
            if nums[l] == min_q[0] {
                min_q.pop_front();
            }
            l += 1
        }
    }
    (nums.len() - l) as i32
}

/// 39
pub fn kth_smallest(mat: Vec<Vec<i32>>, k: i32) -> i32 {
    use std::collections::BinaryHeap;
    let mut queue = BinaryHeap::new();
    queue.push(0);
    for row in mat {
        let mut next = BinaryHeap::new();
        for prev in queue {
            for curr in &row {
                next.push(prev + curr);
            }
        }
        while next.len() > k as usize {
            next.pop();
        }
        queue = next;
    }
    queue.pop().unwrap()
}

/// 41
pub fn build_array(target: Vec<i32>, n: i32) -> Vec<String> {
    let (mut ans, mut stack, mut cur_target) = (vec![], VecDeque::new(), 0);
    'a: for i in 1..=n {
        if cur_target == target.len() {
            break;
        }
        while i != target[cur_target] {
            stack.push_back(i);
            ans.push("Push".to_string());
            continue 'a;
        }
        while stack.len() != 0
            && (cur_target == 0 || *stack.back().unwrap() != target[cur_target - 1])
        {
            stack.pop_back();
            ans.push("Pop".to_string());
        }
        stack.push_back(i);
        cur_target = cur_target + 1;
        ans.push("Push".to_string());
    }
    ans
}

/// 42
pub fn count_triplets(arr: Vec<i32>) -> i32 {
    let mut ans = 0;
    let mut s: Vec<i32> = vec![0];
    for i in 0..arr.len() {
        s.push(s[i] ^ arr[i]);
    }
    let s = s;
    for i in 0..arr.len() {
        for j in (i + 1)..arr.len() {
            for k in j..arr.len() {
                if s[i] == s[k + 1] {
                    ans += 1;
                }
            }
        }
    }
    ans
}

/// 44
pub fn ways(pizza: Vec<String>, k: i32) -> i32 {
    const MOD: i64 = 1_000_000_007;

    let mut pizza: Vec<Vec<char>> = pizza
        .into_iter()
        .map(|s| s.chars().rev().collect())
        .collect();
    pizza.reverse();

    let (m, n) = (pizza.len(), pizza[0].len());
    let mut pre_sum: Vec<Vec<i64>> = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            let cnt = if pizza[i - 1][j - 1] == 'A' { 1 } else { 0 };
            pre_sum[i][j] = cnt + pre_sum[i - 1][j] + pre_sum[i][j - 1] - pre_sum[i - 1][j - 1];
        }
    }

    let any_apple = |(lm, ln), (sm, sn)| {
        let vec: &Vec<i64> = &pre_sum[lm];
        let vec1: &Vec<i64> = &pre_sum[sm];
        let vec2: &Vec<i64> = &pre_sum[lm];
        let vec3: &Vec<i64> = &pre_sum[sm];

        let (a, b, c, d) = (vec[ln], vec1[ln], vec2[sn], vec3[sn]);
        let cnt: i64 = a - b - c + d;
        cnt >= 1
    };

    let mut dp = vec![vec![0; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if any_apple((i, j), (0, 0)) { 1 } else { 0 };
        }
    }

    for _ in 1..k {
        let mut tmp = vec![vec![0; n + 1]; m + 1];

        for lm in 1..=m {
            for ln in 1..=n {
                for i in 0..lm {
                    if any_apple((lm, ln), (i, 0)) {
                        tmp[lm][ln] = (tmp[lm][ln] + dp[i][ln]) % MOD;
                    }
                }

                for j in 0..ln {
                    if any_apple((lm, ln), (0, j)) {
                        tmp[lm][ln] = (tmp[lm][ln] + dp[lm][j]) % MOD;
                    }
                }
            }
        }

        dp = tmp;
    }

    (dp[m][n] % MOD) as i32
}

/// 46
pub fn max_power(s: String) -> i32 {
    let (mut dp, mut ans) = (vec![], 1);
    dp.push(1);
    let chars = s.chars().collect::<Vec<char>>();
    for i in 1..chars.len() {
        if chars[i] == chars[i - 1] {
            dp.push(dp[i - 1] + 1);
            ans = ans.max(dp[i]);
        } else {
            dp.push(1);
        }
    }
    ans
}

/// 47
pub fn simplified_fractions(n: i32) -> Vec<String> {
    let mut ans = vec![];
    fn gcd(mut a: i32, mut b: i32) -> i32 {
        while a % b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        b
    }
    for i in 2..=n {
        for j in 1..=i - 1 {
            if gcd(i, j) == 1 {
                ans.push(j.to_string() + "/" + &i.to_string());
            }
        }
    }
    ans
}

/// 48
pub fn good_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, max: i32) -> i32 {
        if let Some(node) = root.as_ref() {
            let new_max = node.borrow().val.max(max);
            let left_cnt = dfs(node.borrow().left.clone(), new_max);
            let right_cnt = dfs(node.borrow().right.clone(), new_max);
            if node.borrow().val == new_max {
                return left_cnt + right_cnt + 1;
            } else {
                return left_cnt + right_cnt;
            }
        } else {
            0
        }
    }
    if let Some(node) = root.as_ref() {
        return dfs(root.clone(), node.borrow().val);
    } else {
        0
    }
}

/// 49
pub fn largest_number(cost: Vec<i32>, target: i32) -> String {
    let target = target as usize;
    let mut dp = vec![i32::MIN; target + 1];
    dp[0] = 0;

    for c in &cost {
        let c = *c as usize;
        for j in c..=target {
            dp[j] = dp[j].max(dp[j - c] + 1);
        }
    }

    if dp[target] < 0 {
        return "0".to_string();
    }

    let mut ans = String::new();
    let mut j = target;
    for i in (0..=8).rev() {
        let c = cost[i] as usize;
        while j >= c && dp[j] == dp[j - c] + 1 {
            ans.push_str(&((1 + i).to_string()));
            j -= c;
        }
    }
    ans
}

/// 50
pub fn busy_student(start_time: Vec<i32>, end_time: Vec<i32>, query_time: i32) -> i32 {
    let mut start_before = start_time
        .iter()
        .enumerate()
        .fold(HashSet::new(), |mut acc, cur| {
            if *cur.1 <= query_time {
                acc.insert(cur.0);
            }
            acc
        });
    end_time.iter().enumerate().for_each(|t| {
        if *t.1 < query_time {
            start_before.remove(&t.0);
        }
    });
    start_before.len() as i32
}

/// 51
pub fn arrange_words(text: String) -> String {
    let mut text = text.into_bytes();
    text[0] = text[0].to_ascii_lowercase();
    let text = unsafe { String::from_utf8_unchecked(text) };
    let mut text: Vec<_> = text.split_ascii_whitespace().collect();
    text.sort_by_key(|s| s.len());
    let text = text.join(" ");
    let mut text = text.into_bytes();
    text[0] = text[0].to_ascii_uppercase();
    unsafe { String::from_utf8_unchecked(text) }
}

/// 55
pub fn is_prefix_of_word(sentence: String, search_word: String) -> i32 {
    let pos = sentence
        .split(' ')
        .collect::<Vec<&str>>()
        .iter()
        .enumerate()
        .fold(-1, |mut acc, cur| {
            if acc == -1 && cur.1.starts_with(&search_word) {
                acc = cur.0 as i32
            }
            acc
        });
    if pos == -1 { pos } else { pos + 1 }
}

/// 56
pub fn max_vowels(s: String, k: i32) -> i32 {
    let (k, chars, mut cnt) = (k as usize, s.chars().collect::<Vec<char>>(), vec![0; 5]);
    for i in 0..k {
        match chars[i] {
            'a' => cnt[0] = cnt[0] + 1,
            'e' => cnt[1] = cnt[1] + 1,
            'i' => cnt[2] = cnt[2] + 1,
            'o' => cnt[3] = cnt[3] + 1,
            'u' => cnt[4] = cnt[4] + 1,
            _ => (),
        }
    }
    let mut max: i32 = cnt.iter().sum();
    for i in k..chars.len() {
        match chars[i - k] {
            'a' => cnt[0] = cnt[0] - 1,
            'e' => cnt[1] = cnt[1] - 1,
            'i' => cnt[2] = cnt[2] - 1,
            'o' => cnt[3] = cnt[3] - 1,
            'u' => cnt[4] = cnt[4] - 1,
            _ => (),
        };
        match chars[i] {
            'a' => cnt[0] = cnt[0] + 1,
            'e' => cnt[1] = cnt[1] + 1,
            'i' => cnt[2] = cnt[2] + 1,
            'o' => cnt[3] = cnt[3] + 1,
            'u' => cnt[4] = cnt[4] + 1,
            _ => (),
        }
        max = max.max(cnt.iter().sum());
    }
    max
}

/// 57
pub fn pseudo_palindromic_paths(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let (mut all_path, mut cur_path) = (vec![], vec![0; 10]);
    fn dfs(
        root: Option<Rc<RefCell<TreeNode>>>,
        all_path: &mut Vec<Vec<i32>>,
        cur_path: &mut Vec<i32>,
    ) {
        if let Some(node) = root {
            let left = node.borrow().left.clone();
            let right = node.borrow().right.clone();
            cur_path[node.borrow().val as usize - 1] += 1;
            if left.is_none() && right.is_none() {
                all_path.push(cur_path.clone())
            }
            if left.is_some() {
                dfs(left, all_path, cur_path);
            }
            if right.is_some() {
                dfs(right, all_path, cur_path);
            }
            cur_path[node.borrow().val as usize - 1] -= 1;
        }
    }
    dfs(root.clone(), &mut all_path, &mut cur_path);
    all_path
        .into_iter()
        .filter(|p| {
            p.iter().fold(0, |mut acc, cur| {
                if cur % 2 != 0 {
                    acc = acc + 1;
                }
                acc
            }) <= 1
        })
        .collect::<Vec<Vec<i32>>>()
        .len() as i32
}

/// 58
pub fn max_dot_product(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let mut dp = vec![vec![0; nums2.len()]; nums1.len()];
    for i in 0..nums1.len() {
        for j in 0..nums2.len() {
            let v = nums1[i] * nums2[j];
            dp[i][j] = v;
            if i > 0 {
                dp[i][j] = dp[i][j].max(dp[i - 1][j]);
            }
            if j > 0 {
                dp[i][j] = dp[i][j].max(dp[i][j - 1]);
            }
            if i > 0 && j > 0 {
                dp[i][j] = dp[i][j].max(dp[i - 1][j - 1] + v);
            }
        }
    }
    dp[nums1.len() - 1][nums2.len() - 1]
}

/// 60
pub fn can_be_equal(mut target: Vec<i32>, mut arr: Vec<i32>) -> bool {
    target.sort();
    arr.sort();
    target == arr
}

/// 61
pub fn has_all_codes(s: String, k: i32) -> bool {
    let (k, chars, mut cnt) = (k as usize, s.chars().collect::<Vec<char>>(), HashSet::new());
    for i in k - 1..chars.len() {
        cnt.insert(chars[i - k + 1..=i].iter().collect::<String>());
    }
    cnt.len() == 2_usize.pow(k as u32)
}

/// 62
pub fn check_if_prerequisite(
    num_courses: i32,
    prerequisites: Vec<Vec<i32>>,
    queries: Vec<Vec<i32>>,
) -> Vec<bool> {
    let num_courses = num_courses as usize;
    let mut grid = vec![vec![false; num_courses]; num_courses];
    prerequisites.iter().for_each(|pair| {
        grid[pair[0] as usize][pair[1] as usize] = true;
    });
    for k in 0..num_courses {
        for i in 0..num_courses {
            for j in 0..num_courses {
                grid[i][j] = grid[i][j] || grid[i][k] && grid[k][j];
            }
        }
    }
    let mut ans = vec![];
    queries
        .iter()
        .for_each(|q| ans.push(grid[q[0] as usize][q[1] as usize]));
    ans
}

/// 63
pub fn cherry_pickup(grid: Vec<Vec<i32>>) -> i32 {
    let m = grid.len();
    let n = grid[0].len();
    let mut f = vec![vec![-1; n]; n];
    let mut g = vec![vec![-1; n]; n];

    f[0][n - 1] = grid[0][0] + grid[0][n - 1];
    for i in 1..m {
        for j1 in 0..n {
            for j2 in 0..n {
                let mut best = -1;
                for dj1 in -1..=1 {
                    for dj2 in -1..=1 {
                        let dj1 = j1 as i32 + dj1;
                        let dj2 = j2 as i32 + dj2;
                        if dj1 >= 0
                            && dj1 < n as i32
                            && dj2 >= 0
                            && dj2 < n as i32
                            && f[dj1 as usize][dj2 as usize] != -1
                        {
                            best = best.max(
                                f[dj1 as usize][dj2 as usize]
                                    + if j1 == j2 {
                                        grid[i][j1]
                                    } else {
                                        grid[i][j1] + grid[i][j2]
                                    },
                            );
                        }
                    }
                }
                g[j1][j2] = best;
            }
        }
        std::mem::swap(&mut f, &mut g);
    }

    let mut ans = 0;
    for j1 in 0..n {
        ans = ans.max(*f[j1].iter().max().unwrap_or(&0));
    }
    ans
}

/// 64
pub fn max_product(nums: Vec<i32>) -> i32 {
    let mut max = i32::MIN;
    for i in 1..nums.len() {
        for j in 0..i {
            max = max.max((nums[i] - 1) * (nums[j] - 1));
        }
    }
    max
}

/// 65
pub fn max_area(h: i32, w: i32, mut horizontal_cuts: Vec<i32>, mut vertical_cuts: Vec<i32>) -> i32 {
    horizontal_cuts.push(0);
    horizontal_cuts.push(h);
    vertical_cuts.push(0);
    vertical_cuts.push(w);
    horizontal_cuts.sort();
    vertical_cuts.sort();
    let max_h = horizontal_cuts
        .windows(2)
        .fold(0, |acc, cur| acc.max(cur[1] - cur[0]));
    let max_v = vertical_cuts
        .windows(2)
        .fold(0, |acc, cur| acc.max(cur[1] - cur[0]));
    (max_h as i64 * max_v as i64 % 1000000007) as i32
}

/// 66
pub fn min_reorder(n: i32, connections: Vec<Vec<i32>>) -> i32 {
    let mut g: Vec<Vec<(i32, i32)>> = vec![vec![]; n as usize];
    for e in connections.iter() {
        let a = e[0] as usize;
        let b = e[1] as usize;
        g[a].push((b as i32, 1));
        g[b].push((a as i32, 0));
    }
    fn dfs(a: usize, fa: i32, g: &Vec<Vec<(i32, i32)>>) -> i32 {
        let mut ans = 0;
        for &(b, c) in g[a].iter() {
            if b != fa {
                ans += c + dfs(b as usize, a as i32, g);
            }
        }
        ans
    }
    dfs(0, -1, &g)
}

/// 70
pub fn shuffle(mut nums: Vec<i32>, n: i32) -> Vec<i32> {
    let n = n as usize;
    for i in 0..n {
        let k = nums.remove(i + n);
        nums.insert(i * 2 + 1, k);
    }
    nums
}

/// 71
pub fn get_strongest(mut arr: Vec<i32>, k: i32) -> Vec<i32> {
    let middle = (arr.len() - 1) / 2;
    arr.select_nth_unstable(middle);
    let m = arr[middle];

    let k = k as usize;
    arr.select_nth_unstable_by_key(k - 1, |item| Reverse((item.abs_diff(m), *item)));
    arr.resize(k, 0);
    arr
}

/// 72
struct BrowserHistory {
    urls: Vec<String>,
    curr_index: usize,
}

impl BrowserHistory {
    fn new(homepage: String) -> Self {
        BrowserHistory {
            urls: vec![homepage],
            curr_index: 0,
        }
    }

    fn visit(&mut self, url: String) {
        self.urls.truncate(self.curr_index + 1);
        self.urls.push(url);
        self.curr_index += 1;
    }

    fn back(&mut self, steps: i32) -> String {
        self.curr_index = (self.curr_index as i32 - steps).max(0) as usize;
        return self.urls[self.curr_index].clone();
    }

    fn forward(&mut self, steps: i32) -> String {
        self.curr_index = std::cmp::min(self.curr_index + steps as usize, self.urls.len() - 1);
        return self.urls[self.curr_index].clone();
    }
}

/// 73
pub fn min_cost(houses: Vec<i32>, cost: Vec<Vec<i32>>, m: i32, n: i32, target: i32) -> i32 {
    let m = m as usize;
    let n = n as usize;
    let target = target as usize;
    let mut dp = vec![vec![vec![i32::MAX; target]; n]; m];
    if houses[0] != 0 {
        dp[0][houses[0] as usize - 1][0] = 0;
    } else {
        for color in 0..n {
            dp[0][color][0] = cost[0][color];
        }
    }
    for i in 0..(m - 1) {
        for j in 0..n {
            for k in 0..target {
                if dp[i][j][k] != i32::MAX {
                    if houses[i + 1] == 0 {
                        if k != target - 1 {
                            for jt in 0..n {
                                if jt != j {
                                    dp[i + 1][jt][k + 1] =
                                        dp[i + 1][jt][k + 1].min(dp[i][j][k] + cost[i + 1][jt]);
                                }
                            }
                        }
                        dp[i + 1][j][k] = dp[i + 1][j][k].min(dp[i][j][k] + cost[i + 1][j]);
                    } else {
                        let jt = (houses[i + 1] - 1) as usize;
                        if jt != j && k != target - 1 {
                            dp[i + 1][jt][k + 1] = dp[i + 1][jt][k + 1].min(dp[i][j][k]);
                        }
                        if jt == j {
                            dp[i + 1][jt][k] = dp[i + 1][jt][k].min(dp[i][j][k]);
                        }
                    }
                }
            }
        }
    }
    let mut res = i32::MAX;
    for j in 0..n {
        res = res.min(dp[m - 1][j][target - 1]);
    }
    if res == i32::MAX { -1 } else { res }
}

/// 75
pub fn final_prices(mut prices: Vec<i32>) -> Vec<i32> {
    let mut stack = VecDeque::new();
    for i in 0..prices.len() {
        while !stack.is_empty() && prices[*stack.back().unwrap()] >= prices[i] {
            prices[stack.pop_back().unwrap()] -= prices[i];
        }
        stack.push_back(i);
    }
    prices
}

/// 76
struct SubrectangleQueries {
    rectangle: Vec<Vec<i32>>,
}

impl SubrectangleQueries {
    fn new(rectangle: Vec<Vec<i32>>) -> Self {
        Self { rectangle }
    }

    fn update_subrectangle(&mut self, row1: i32, col1: i32, row2: i32, col2: i32, new_value: i32) {
        let (row1, col1, row2, col2) = (row1 as usize, col1 as usize, row2 as usize, col2 as usize);
        for i in row1..=row2 {
            for j in col1..=col2 {
                self.rectangle[i][j] = new_value;
            }
        }
    }

    fn get_value(&self, row: i32, col: i32) -> i32 {
        self.rectangle[row as usize][col as usize]
    }
}

/// 77
// pub fn min_sum_of_lengths(arr: Vec<i32>, target: i32) -> i32 {

// }

/// 80
pub fn running_sum(nums: Vec<i32>) -> Vec<i32> {
    let mut prefix = vec![];
    for i in 0..nums.len() {
        if i == 0 {
            prefix.push(nums[i]);
        } else {
            prefix.push(nums[i] + prefix[i - 1]);
        }
    }
    prefix
}
