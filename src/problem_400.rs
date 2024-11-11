/// p303
struct NumArray {
    sum: Vec<i32>,
}

impl NumArray {
    fn new(nums: Vec<i32>) -> Self {
        let sum = nums
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let mut sum = 0;
                for i in 0..=index {
                    sum = sum + nums[i]
                }
                return sum;
            })
            .collect::<Vec<i32>>();
        NumArray { sum }
    }

    fn sum_range(&self, left: i32, right: i32) -> i32 {
        if left == 0 {
            return self.sum[right as usize];
        } else {
            return self.sum[right as usize] - self.sum[(left - 1) as usize];
        }
    }
}

/// p304
struct NumMatrix {
    sum: Vec<Vec<i32>>,
}

impl NumMatrix {
    fn new(matrix: Vec<Vec<i32>>) -> Self {
        let m = matrix.len();
        let n = matrix[0].len();
        let mut sum = vec![vec![0; n + 1]; m + 1];
        for (i, row) in matrix.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + x;
            }
        }
        Self { sum }
    }

    fn sum_region(&self, row1: i32, col1: i32, row2: i32, col2: i32) -> i32 {
        let r1 = row1 as usize;
        let c1 = col1 as usize;
        let r2 = row2 as usize + 1;
        let c2 = col2 as usize + 1;
        self.sum[r2][c2] - self.sum[r2][c1] - self.sum[r1][c2] + self.sum[r1][c1]
    }
}

/// p306
pub fn is_additive_number(num: String) -> bool {
    let mut first = 0;
    let num_arr: Vec<char> = num.chars().collect();
    for i in 0..num.len() {
        if i > 0 && num_arr[0] == '0' {
            return false;
        }
        first = first * 10 + (num_arr[i] as u8 - '0' as u8) as i64;
        let mut second = 0;
        for j in i + 1..num.len() {
            second = second * 10 + (num_arr[j] as u8 - '0' as u8) as i64;
            if j > i + 1 && num_arr[i + 1] == '0' {
                break;
            }
            if j + 1 < num.len() && is_can_added(first, second, num.as_str(), j + 1) {
                return true;
            }
        }
    }
    false
}

fn is_can_added(first: i64, second: i64, num: &str, sum_idx: usize) -> bool {
    if sum_idx == num.len() {
        return true;
    }

    let sum_str = i64::to_string(&(first + second));
    if sum_idx + sum_str.len() > num.len() {
        return false;
    }

    let actual_sum = &num[sum_idx..sum_idx + sum_str.len()];
    actual_sum == sum_str && is_can_added(second, first + second, num, sum_idx + sum_str.len())
}
