use std::collections::BTreeMap;

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
