/// 1103
pub fn distribute_candies(mut candies: i32, n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut ans = vec![0; n];
    let mut i = 1;
    while candies > 0 {
        ans[(i - 1) as usize % n] += i.min(candies);
        candies -= i;
        i += 1;
    }
    ans
}

/// 1104
pub fn path_in_zig_zag_tree(label: i32) -> Vec<i32> {
    let mut x = 2_i32.pow((label as f64).log2() as u32);
    let mut label = label;
    let mut ret = vec![label];

    while label > 1 {
        label = x - 1 - ((label / 2) % (x / 2));
        x /= 2;
        ret.push(label);
    }

    ret.reverse();
    ret
}
