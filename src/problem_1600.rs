use std::{cell::RefCell, i32, str::FromStr};

/// 02
pub fn can_make_arithmetic_progression(mut arr: Vec<i32>) -> bool {
    arr.sort();
    arr.windows(2)
        .fold((i32::MAX, true), |acc, cur| {
            if acc.1 == false {
                (acc.0, false)
            } else if acc.0 == i32::MAX {
                (cur[1] - cur[0], true)
            } else if cur[1] - cur[0] != acc.0 {
                (acc.0, false)
            } else {
                (acc.0, true)
            }
        })
        .1
}

/// 03
pub fn get_last_moment(n: i32, mut left: Vec<i32>, mut right: Vec<i32>) -> i32 {
    left.sort();
    right.sort();
    let mut t = 0;
    if right.len() > 0 {
        t = t.max(n - right[0]);
    }
    if left.len() > 0 {
        t = t.max(left[left.len() - 1]);
    }
    t
}

/// 04
pub fn num_submat(mat: Vec<Vec<i32>>) -> i32 {
    let (height, width) = (mat.len(), mat[0].len());
    let mut row = vec![vec![0; width]; height];
    let mut ans = 0;
    for i in 0..height {
        for j in 0..width {
            if mat[i][j] == 0 {
                row[i][j] = 0;
            } else {
                if j == 0 {
                    row[i][j] = 1;
                } else {
                    row[i][j] = 1 + row[i][j - 1];
                }
            }
            let mut cur = row[i][j];
            for k in (0..=i).rev() {
                cur = cur.min(row[k][j]);
                if cur == 0 {
                    break;
                }
                ans = ans + cur;
            }
        }
    }
    ans
}

/// 05
pub fn min_integer(num: String, mut k: i32) -> String {
    let n = num.len();
    let p = num.into_bytes();
    let mut position = vec![vec![]; 10];
    for i in (0..n).rev() {
        position[(p[i] - b'0') as usize].push(i + 1);
    }

    let tree = RefCell::new(vec![0; n + 1]);
    let update = |mut x: i32| {
        loop {
            if x > n as i32 {
                break;
            }
            tree.borrow_mut()[x as usize] += 1;
            x += x & -x;
        }
    };
    let query = |mut x: i32| -> i32 {
        let mut a = 0;
        loop {
            if x == 0 {
                break;
            }
            a += tree.borrow()[x as usize];
            x -= x & -x;
        }
        a
    };
    let sum_range = |l: i32, r: i32| -> i32 { query(r as i32) - query((l - 1) as i32) };
    let mut ans = vec![0_u8; n];
    let mut l = 0;
    for i in 1..=n {
        for j in 0..10 {
            if position[j].len() > 0 {
                let c = *position[j].last().unwrap();
                let b = sum_range(c as i32, n as i32);
                let dist = c as i32 + b - i as i32;
                if dist <= k {
                    update(c as i32);
                    position[j].pop();
                    ans[l] = (j + 48) as u8;
                    l += 1;
                    k -= dist;
                    break;
                }
            }
        }
    }
    unsafe { String::from_utf8_unchecked(ans) }
}

/// 07
pub fn reformat_date(date: String) -> String {
    let vs: Vec<&str> = date.split(' ').collect();
    let ms = vec![
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut i = 0;
    let mut md = 1;
    let mut dd = 0;
    let mut res = String::new();
    for w in vs {
        if i == 0 {
            let l = w.len();
            let t = &w[0..(l - 2)];
            dd = FromStr::from_str(t).unwrap();
        }
        if i == 1 {
            for m in ms.clone() {
                if m == w {
                    break;
                }
                md += 1;
            }
        }
        if i == 2 {
            res.push_str(w);
            res.push('-');
        }
        i += 1;
    }
    if md < 10 {
        res.push('0');
        res.push_str(&md.to_string());
    } else {
        res.push_str(&md.to_string());
    }
    res.push('-');
    if dd < 10 {
        res.push('0');
        res.push_str(&dd.to_string());
    } else {
        res.push_str(&dd.to_string());
    }
    res
}

