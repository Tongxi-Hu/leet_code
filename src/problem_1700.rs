/// 01
pub fn maximum_requests(n: i32, mut requests: Vec<Vec<i32>>) -> i32 {
    let mut cache = vec![0; 21];
    fn back_track(n: i32, i: i32, cache: &mut Vec<i32>, requests: &mut Vec<Vec<i32>>) -> i32 {
        if i as usize == requests.len() {
            return if (0..n)
                .into_iter()
                .filter(|j| cache[*j as usize] == 0)
                .count()
                == n as usize
            {
                0
            } else {
                i32::MIN
            };
        }
        cache[requests[i as usize][0] as usize] -= 1;
        cache[requests[i as usize][1] as usize] += 1;
        let take = 1 + back_track(n, i + 1, cache, requests);
        cache[requests[i as usize][0] as usize] += 1;
        cache[requests[i as usize][1] as usize] -= 1;
        return take.max(back_track(n, i + 1, cache, requests));
    }
    back_track(n, 0, &mut cache, &mut requests)
}

/// 03
struct ParkingSystem {
    big: (i32, i32),
    medium: (i32, i32),
    small: (i32, i32),
}

impl ParkingSystem {
    fn new(big: i32, medium: i32, small: i32) -> Self {
        Self {
            big: (big, 0),
            medium: (medium, 0),
            small: (small, 0),
        }
    }

    fn add_car(&mut self, car_type: i32) -> bool {
        match car_type {
            1 => {
                if self.big.1 < self.big.0 {
                    self.big.1 += 1;
                    return true;
                } else {
                    false
                }
            }
            2 => {
                if self.medium.1 < self.medium.0 {
                    self.medium.1 += 1;
                    return true;
                } else {
                    false
                }
            }
            3 => {
                if self.small.1 < self.small.0 {
                    self.small.1 += 1;
                    return true;
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

/// 04
pub fn alert_names(key_name: Vec<String>, key_time: Vec<String>) -> Vec<String> {
    let mut mp = std::collections::HashMap::new();
    let n = key_name.len();

    for i in 0..n {
        let time = key_time[i][0..2].parse::<i32>().unwrap() * 60
            + key_time[i][3..].parse::<i32>().unwrap();
        mp.entry(&key_name[i]).or_insert(vec![]).push(time);
    }

    let mut ans = vec![];

    for (key, val) in mp.iter_mut() {
        val.sort();
        if val.len() < 3 {
            continue;
        }
        for i in 2..val.len() {
            if val[i] - val[i - 2] <= 60 {
                ans.push(key.to_string());
                break;
            }
        }
    }

    ans.sort();
    ans
}

/// 05
pub fn restore_matrix(mut row_sum: Vec<i32>, mut col_sum: Vec<i32>) -> Vec<Vec<i32>> {
    let (mut r, mut c, height, width) = (0, 0, row_sum.len(), col_sum.len());
    let mut grid = vec![vec![0; width]; height];
    while r < height && c < width {
        let min = row_sum[r].min(col_sum[c]);
        grid[r][c] = min;
        row_sum[r] = row_sum[r] - min;
        col_sum[c] = col_sum[c] - min;
        if row_sum[r] == 0 {
            r = r + 1;
        }
        if col_sum[c] == 0 {
            c = c + 1;
        }
    }
    grid
}
