/// p605
pub fn can_place_flowers(flowerbed: Vec<i32>, n: i32) -> bool {
    let m = flowerbed.len();
    let mut n = n;
    let mut i = 0;
    while i < m {
        if (i == 0 || flowerbed[i - 1] == 0)
            && flowerbed[i] == 0
            && (i == m - 1 || flowerbed[i + 1] == 0)
        {
            n -= 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    n <= 0
}
