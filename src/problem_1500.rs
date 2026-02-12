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
