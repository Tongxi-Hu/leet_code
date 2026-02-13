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
