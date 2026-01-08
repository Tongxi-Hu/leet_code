/// 1001
pub fn grid_illumination(n: i32, lamps: Vec<Vec<i32>>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    const DIRS: [[i32; 2]; 9] = [
        [0, 0],
        [1, 1],
        [-1, -1],
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [-1, 1],
        [1, -1],
    ];

    use std::collections::{HashMap, HashSet};
    fn match_rule(cache: &HashMap<i32, i32>, grid: &i32) -> bool {
        if let Some(v) = cache.get(grid) {
            *v > 0
        } else {
            false
        }
    }

    let (mut axis_x, mut axis_y, mut axis_x_to_y, mut axis_y_to_x, mut is_bright, mut ret) = (
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashSet::new(),
        Vec::new(),
    );

    for lamp in lamps {
        let (x, y) = (lamp[0], lamp[1]);
        if is_bright.contains(&(x, y)) {
            continue;
        }

        is_bright.insert((x, y));
        *axis_x.entry(x).or_insert(0) += 1;
        *axis_y.entry(y).or_insert(0) += 1;
        *axis_x_to_y.entry(x - y).or_insert(0) += 1;
        *axis_y_to_x.entry(x + y).or_insert(0) += 1;
    }

    for query in queries {
        let (x, y) = (query[0], query[1]);
        ret.push(
            (match_rule(&axis_x, &x)
                || match_rule(&axis_y, &y)
                || match_rule(&axis_x_to_y, &(x - y))
                || match_rule(&axis_y_to_x, &(x + y))) as i32,
        );
        for dir in DIRS {
            let (tx, ty) = (x + dir[0], y + dir[1]);
            if tx < 0 || tx >= n || ty < 0 || ty >= n || !is_bright.contains(&(tx, ty)) {
                continue;
            }
            is_bright.remove(&(tx, ty));
            *axis_x.get_mut(&tx).unwrap() -= 1;
            *axis_y.get_mut(&ty).unwrap() -= 1;
            *axis_x_to_y.get_mut(&(tx - ty)).unwrap() -= 1;
            *axis_y_to_x.get_mut(&(tx + ty)).unwrap() -= 1;
        }
    }
    ret
}

/// 1002
pub fn common_chars(words: Vec<String>) -> Vec<String> {
    let mut min_count = vec![usize::MAX; 26];
    words.into_iter().for_each(|word| {
        let mut count = vec![0; 26];
        word.chars().for_each(|c| {
            let index = c as usize - 'a' as usize;
            count[index] = count[index] + 1;
        });
        min_count = min_count
            .iter()
            .zip(count)
            .map(|(min, c)| *min.min(&c))
            .collect::<Vec<usize>>();
    });
    let mut common = vec![];
    min_count.iter().enumerate().for_each(|(i, c)| {
        if *c != 0 {
            for _ in 0..*c {
                common.push((('a' as u8 + i as u8) as char).to_string());
            }
        }
    });
    common
}

/// 1003
pub fn is_valid(s: String) -> bool {
    let mut stack: Vec<char> = vec![];
    s.chars().for_each(|c| {
        if c == 'c'
            && stack.len() >= 2
            && stack[stack.len() - 1] == 'b'
            && stack[stack.len() - 2] == 'a'
        {
            stack.pop();
            stack.pop();
        } else {
            stack.push(c)
        }
    });
    stack.len() == 0
}

/// 1004
pub fn longest_ones(nums: Vec<i32>, k: i32) -> i32 {
    let mut zero_count = 0;
    let mut max_len = 0;

    let mut l = 0;
    for (i, &num) in nums.iter().enumerate() {
        if num == 0 {
            zero_count += 1;
        }

        while zero_count > k {
            if nums[l] == 0 {
                zero_count -= 1;
            }
            l += 1;
        }

        max_len = i32::max(max_len, (i - l + 1) as i32);
    }

    max_len
}

/// 1005
pub fn largest_sum_after_k_negations(nums: Vec<i32>, k: i32) -> i32 {
    let mut a = nums;
    a.sort_unstable();
    let mut k = k;
    for i in a.iter_mut() {
        if k > 0 && *i < 0 {
            *i *= -1;
            k -= 1;
        } else {
            break;
        }
    }
    a.sort_unstable();
    a[0] *= if k & 1 == 1 { -1 } else { 1 };
    a.into_iter().sum()
}

/// 1006
pub fn clumsy(n: i32) -> i32 {
    let mut flag = 0;
    let mut res = 0;
    let mut tmp = n;
    let mut cur = n;
    cur -= 1;

    while cur > 0 {
        match flag {
            0 => tmp *= cur,
            1 => tmp /= cur,
            2 => {
                res = res + tmp + cur;
                tmp = 0;
            }
            3 => {
                tmp = -cur;
            }
            _ => {}
        }
        flag += 1;
        if flag > 3 {
            flag = 0;
        }
        cur -= 1;
    }
    if tmp != 0 {
        res += tmp;
    }
    res
}

/// 1007
pub fn min_domino_rotations(tops: Vec<i32>, bottoms: Vec<i32>) -> i32 {
    fn check(x: i32, tops: &[i32], bottoms: &[i32], n: usize) -> i32 {
        let (mut rotations_a, mut rotations_b) = (0, 0);
        for i in 0..n {
            if tops[i] != x && bottoms[i] != x {
                return -1;
            } else if tops[i] != x {
                rotations_a += 1;
            } else if bottoms[i] != x {
                rotations_b += 1;
            }
        }
        rotations_a.min(rotations_b)
    }
    let n = tops.len();
    let rotations = check(tops[0], &tops, &bottoms, n);
    if rotations != -1 || tops[0] == bottoms[0] {
        return rotations;
    }
    check(bottoms[0], &tops, &bottoms, n)
}

