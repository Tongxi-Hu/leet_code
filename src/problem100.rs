use std::collections::{HashMap, HashSet};

///p1 two sum
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut register: HashMap<i32, i32> = HashMap::new();
    let mut location: Vec<i32> = vec![];
    for (index, val) in nums.iter().enumerate() {
        match register.get(&(target - val)) {
            None => {
                register.insert(*val, index as i32);
            }
            Some(i) => {
                location.push(*i);
                location.push(index as i32);
            }
        }
    }
    return location;
}

#[test]
pub fn test_two_sum() {
    assert_eq!(vec![0, 1], two_sum(vec![2, 7, 11, 15], 9));
    assert_eq!(vec![1, 2], two_sum(vec![3, 2, 4], 6));
}

///p2 add number
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    if l1 == None && l2 == None {
        return None;
    }
    let mut start: Option<Box<ListNode>> = None;
    let mut p1 = &l1;
    let mut p2 = &l2;
    let mut p = &mut start;
    let mut acc: i32 = 0;
    loop {
        match (p1, p2) {
            (Some(val1), Some(val2)) => {
                let mut sum = val1.val + val2.val + acc;
                if sum >= 10 {
                    let temp = sum - 10;
                    acc = (sum - temp) / 10;
                    sum = temp;
                } else {
                    acc = 0;
                }
                *p = Some(Box::new(ListNode::new(sum)));
                p = &mut (p.as_mut().unwrap().next);
                p1 = &val1.next;
                p2 = &val2.next;
            }
            (None, Some(val2)) => {
                let mut sum = val2.val + acc;
                if sum >= 10 {
                    let temp = sum - 10;
                    acc = (sum - temp) / 10;
                    sum = temp;
                } else {
                    acc = 0;
                }
                *p = Some(Box::new(ListNode::new(sum)));
                p = &mut (p.as_mut().unwrap().next);
                p2 = &val2.next;
            }
            (Some(val1), None) => {
                let mut sum = val1.val + acc;
                if sum >= 10 {
                    let temp = sum - 10;
                    acc = (sum - temp) / 10;
                    sum = temp;
                } else {
                    acc = 0;
                }
                *p = Some(Box::new(ListNode::new(sum)));
                p = &mut (p.as_mut().unwrap().next);
                p1 = &val1.next;
            }
            (None, None) => {
                let mut sum = acc;
                if sum >= 10 {
                    let temp = sum - 10;
                    acc = (sum - temp) / 10;
                    sum = temp;
                } else {
                    acc = 0;
                }
                if (sum == 0) {
                    break;
                }
                *p = Some(Box::new(ListNode::new(sum)));
                p = &mut (p.as_mut().unwrap().next);
                break;
            }
        }
    }
    return start;
}

#[test]
pub fn test_add_two_numbers() {
    assert_eq!(
        Some(Box::new(ListNode {
            val: 7,
            next: Some(Box::new(ListNode { val: 1, next: None }))
        })),
        add_two_numbers(
            Some(Box::new(ListNode { val: 8, next: None })),
            Some(Box::new(ListNode { val: 9, next: None }))
        )
    );
}

///p3 longest substring
pub fn length_of_longest_substring(s: String) -> i32 {
    let mut left = 0;
    let mut right = 0;
    let mut max = 0;
    let mut collection = HashSet::<char>::new();
    let char_vec = s.chars().collect::<Vec<char>>();
    while right < char_vec.len() {
        match char_vec.get(right) {
            None => (),
            Some(char) => match collection.get(char) {
                None => {
                    collection.insert(*char);
                    max = collection.len().max(max);
                    right = right + 1;
                }
                Some(_) => {
                    while collection.get(char).is_some() {
                        match char_vec.get(left) {
                            None => (),
                            Some(val) => {
                                collection.remove(&val);
                            }
                        }
                        left = left + 1;
                    }
                }
            },
        }
    }
    return max as i32;
}

#[test]
fn test_length_of_longest_substring() {
    assert_eq!(3, length_of_longest_substring("abcabcbb".to_string()));
    assert_eq!(3, length_of_longest_substring("pwwkew".to_string()));
}

///p4 median of two sorted arrays
pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let mut combine: Vec<i32> = Vec::new();
    let len = nums1.len() + nums2.len();
    let mut p1 = 0;
    let mut p2 = 0;
    for _ in 0..len {
        match (nums1.get(p1), nums2.get(p2)) {
            (Some(val1), Some(val2)) => {
                if val1 <= val2 {
                    combine.push(*val1);
                    p1 = p1 + 1;
                } else {
                    combine.push(*val2);
                    p2 = p2 + 1;
                }
            }
            (Some(val1), None) => {
                combine.push(*val1);
                p1 = p1 + 1;
            }
            (None, Some(val2)) => {
                combine.push(*val2);
                p2 = p2 + 1;
            }
            (None, None) => {}
        }
    }

    if len % 2 == 0 {
        return (combine[len / 2 - 1] as f64 + combine[len / 2] as f64) / 2.0;
    } else {
        return combine[len / 2] as f64;
    }
}

#[test]
fn test_find_median_sorted_arrays() {
    assert_eq!(2.0, find_median_sorted_arrays(vec![1, 3], vec![2]));
    assert_eq!(2.5, find_median_sorted_arrays(vec![1, 2, 3, 4], vec![]));
}

///p7
pub fn reverse(x: i32) -> i32 {
    let mut x = x;
    let mut result = 0;
    while x != 0 {
        if result > i32::MAX / 10 || result < i32::MIN / 10 {
            return 0;
        }
        result = result * 10 + x % 10;
        x = x / 10;
    }
    return result;
}

#[test]
fn test_reverse() {
    assert_eq!(-321, reverse(-123));
    assert_eq!(21, reverse(120));
}

///p8
pub fn my_atoi(s: String) -> i32 {
    let mut sign = '+';
    let mut val: i64 = 0;
    let mut has_content = false;
    for char in s.chars().into_iter() {
        match char.to_digit(10) {
            Some(digit) => {
                val = val * 10 + (digit as i64);
                has_content = true;
                if sign == '-' {
                    if -val < i32::MIN as i64 {
                        return i32::MIN;
                    }
                } else {
                    if val > i32::MAX as i64 {
                        return i32::MAX;
                    }
                }
            }
            None => {
                if char.is_whitespace() {
                    if has_content {
                        break;
                    } else {
                        continue;
                    }
                } else if char == '-' || char == '+' {
                    if has_content {
                        break;
                    } else {
                        has_content = true;
                        sign = char;
                    }
                } else {
                    break;
                }
            }
        }
    }
    if sign == '-' {
        return (val as i32) * -1;
    } else {
        return val as i32;
    }
}

#[test]
fn test_my_atoi() {
    assert_eq!(-42, my_atoi(" -042".to_string()));
    assert_eq!(1337, my_atoi("1337c0d3".to_string()));
}

///p9
pub fn is_palindrome(x: i32) -> bool {
    if x < 0 {
        return false;
    }
    let reverse = x
        .to_string()
        .chars()
        .rev()
        .collect::<String>()
        .parse::<i32>();
    match reverse {
        Ok(val) => {
            return val == x;
        }
        Err(_) => return false,
    }
}

///p10
pub fn is_match(s: String, p: String) -> bool {
    let s: Vec<char> = s.chars().collect();
    let p: Vec<char> = p.chars().collect();
    let match_c = |i, j| -> bool { i != 0 && (p[j - 1] == '.' || s[i - 1] == p[j - 1]) };
    let mut dp = vec![vec![false; p.len() + 1]; s.len() + 1];
    dp[0][0] = true;
    (0..=s.len()).for_each(|i| {
        (1..=p.len()).for_each(|j| {
            dp[i][j] = if p[j - 1] == '*' {
                match_c(i, j - 1) && dp[i - 1][j] || dp[i][j - 2]
            } else {
                match_c(i, j) && dp[i - 1][j - 1]
            };
        })
    });
    dp[s.len()][p.len()]
}

///p11
pub fn max_area(height: Vec<i32>) -> i32 {
    let mut left = 0;
    let mut right = height.len() - 1;
    let mut vol = 0;
    while left < right {
        let left_height = height.get(left).unwrap_or(&0);
        let right_height = height.get(right).unwrap_or(&0);
        let new_vol = (right - left) * (*std::cmp::min(left_height, right_height) as usize);
        if new_vol > vol {
            vol = new_vol;
        };
        if left_height <= right_height {
            left = left + 1;
        } else {
            right = right - 1;
        }
    }
    return vol as i32;
}

#[test]
fn test_max_area() {
    assert_eq!(49, max_area(vec![1, 8, 6, 2, 5, 4, 8, 3, 7]));
    assert_eq!(1, max_area(vec![1, 1]))
}

///p12
pub fn int_to_roman(num: i32) -> String {
    const I: [&'static str; 10] = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"];
    const X: [&'static str; 10] = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"];
    const C: [&'static str; 10] = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"];
    const M: [&'static str; 4] = ["", "M", "MM", "MMM"];

    let n = num as usize;
    let mut s = M[n / 1000].to_string();
    s.push_str(C[(n % 1000) / 100]);
    s.push_str(X[(n % 100) / 10]);
    s.push_str(I[n % 10]);
    s
}

///p13
pub fn roman_to_int(s: String) -> i32 {
    s.chars()
        .fold((0, ' '), |res, ch| match (res.1, ch) {
            ('I', 'V') => (res.0 + 3, 'V'),
            ('I', 'X') => (res.0 + 8, 'X'),
            ('X', 'L') => (res.0 + 30, 'L'),
            ('X', 'C') => (res.0 + 80, 'C'),
            ('C', 'D') => (res.0 + 300, 'D'),
            ('C', 'M') => (res.0 + 800, 'M'),
            (_, 'I') => (res.0 + 1, 'I'),
            (_, 'V') => (res.0 + 5, 'V'),
            (_, 'X') => (res.0 + 10, 'X'),
            (_, 'L') => (res.0 + 50, 'L'),
            (_, 'C') => (res.0 + 100, 'C'),
            (_, 'D') => (res.0 + 500, 'D'),
            (_, 'M') => (res.0 + 1000, 'M'),
            (_, _) => unreachable!(),
        })
        .0
}

///p14
pub fn longest_common_prefix(strs: Vec<String>) -> String {
    let mut ans = Vec::<char>::new();
    for (i, str) in strs.iter().enumerate() {
        if i == 0 {
            str.chars().for_each(|char| ans.push(char));
            continue;
        } else {
            if str.len() < ans.len() {
                ans.drain(str.len()..ans.len());
            }
            'inner: for (j, char) in str.chars().enumerate() {
                match ans.get(j) {
                    None => {
                        break 'inner;
                    }
                    Some(val) => match val == &char {
                        true => {
                            continue;
                        }
                        false => {
                            ans.drain(j..ans.len());
                            break;
                        }
                    },
                }
            }
        }
    }
    return ans.iter().collect::<String>();
}

#[test]
fn test_longest_common_prefix() {
    assert_eq!(
        "a".to_string(),
        longest_common_prefix(vec!["ab".to_string(), "a".to_string()])
    )
}

///P15
pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut nums = nums;
    nums.sort();
    let mut lists: Vec<Vec<i32>> = Vec::new();
    for (i, _) in nums.iter().enumerate() {
        if nums[i] > 0 {
            return lists;
        }
        if i > 0 && nums[i] == nums[i - 1] {
            continue;
        }
        let mut l = i + 1;
        let mut r = nums.len() - 1;
        while l < r {
            let sum = nums[i] + nums[l] + nums[r];
            match sum {
                0 => {
                    lists.push(Vec::from([nums[i], nums[l], nums[r]]));
                    while l < r && nums[l + 1] == nums[l] {
                        l = l + 1;
                    }
                    while l < r && nums[r - 1] == nums[r] {
                        r = r - 1;
                    }
                    l = l + 1;
                    r = r - 1;
                }
                i32::MIN..=-1 => {
                    l = l + 1;
                }
                1..=i32::MAX => {
                    r = r - 1;
                }
            }
        }
    }
    return lists;
}

///p16
pub fn three_sum_closest(nums: Vec<i32>, target: i32) -> i32 {
    let mut nums = nums;
    nums.sort();
    let mut sum = nums[0] + nums[1] + nums[2];
    for (i, _) in nums.iter().enumerate() {
        let mut l = i + 1;
        let mut r = nums.len() - 1;
        while l < r {
            let temp = nums[i] + nums[l] + nums[r];
            if (temp - target).abs() < (sum - target).abs() {
                sum = temp;
            }
            if temp > target {
                r = r - 1;
            } else if temp < target {
                l = l + 1;
            } else {
                return sum;
            }
        }
    }
    return sum;
}

///p17
pub fn letter_combinations(digits: String) -> Vec<String> {
    const RANGE: [(usize, usize); 8] = [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
        (12, 15),
        (15, 19),
        (19, 22),
        (22, 26),
    ];

    let acc = match digits.is_empty() {
        false => vec![String::new()],
        true => vec![],
    };

    digits.as_bytes().iter().fold(acc, |acc, c| {
        let (min, max) = RANGE[usize::from(c - 50)];
        acc.iter()
            .flat_map(|x| {
                std::iter::repeat(x)
                    .zip(min..max)
                    .map(|(x, n)| format!("{}{}", x, (97u8 + n as u8) as char))
            })
            .collect::<Vec<String>>()
    })
}

///p18
pub fn four_sum(nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    let mut result: Vec<Vec<i32>> = Vec::new();
    let mut nums = nums;
    nums.sort();
    let len = nums.len();
    for k in 0..len {
        // 剪枝
        if nums[k] > target && (nums[k] > 0 || target > 0) {
            break;
        }
        // 去重
        if k > 0 && nums[k] == nums[k - 1] {
            continue;
        }
        for i in (k + 1)..len {
            // 剪枝
            if nums[k] + nums[i] > target && (nums[k] + nums[i] >= 0 || target >= 0) {
                break;
            }
            // 去重
            if i > k + 1 && nums[i] == nums[i - 1] {
                continue;
            }
            let (mut left, mut right) = (i + 1, len - 1);
            while left < right {
                if nums[k] + nums[i] > target - (nums[left] + nums[right]) {
                    right -= 1;
                    // 去重
                    while left < right && nums[right] == nums[right + 1] {
                        right -= 1;
                    }
                } else if nums[k] + nums[i] < target - (nums[left] + nums[right]) {
                    left += 1;
                    // 去重
                    while left < right && nums[left] == nums[left - 1] {
                        left += 1;
                    }
                } else {
                    result.push(vec![nums[k], nums[i], nums[left], nums[right]]);
                    // 去重
                    while left < right && nums[right] == nums[right - 1] {
                        right -= 1;
                    }
                    while left < right && nums[left] == nums[left + 1] {
                        left += 1;
                    }
                    left += 1;
                    right -= 1;
                }
            }
        }
    }
    result
}

///p19
pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let mut head = head;
    let mut length = 0;
    let mut pointer = &head;
    while pointer != &None {
        match pointer {
            Some(ref val) => {
                length = length + 1;
                pointer = &val.next;
            }
            None => {
                break;
            }
        }
    }
    if length > n {
        let from_heads = length - n;
        let mut index = 0;
        let mut pointer = &mut head;
        while index < from_heads - 1 {
            match pointer {
                Some(ref mut val) => {
                    pointer = &mut val.next;
                    index = index + 1;
                }
                None => (),
            }
        }
        match pointer {
            Some(ref mut val) => {
                let to_remove = &mut val.next;
                match to_remove {
                    Some(ctx) => {
                        let tail = std::mem::replace(&mut ctx.next, None);
                        val.next = tail;
                    }
                    None => (),
                }
            }
            None => (),
        }
    } else if length == n {
        match head {
            Some(mut val) => head = std::mem::replace(&mut val.next, None),
            None => (),
        }
    }
    return head;
}

///p20
pub fn is_valid(s: String) -> bool {
    let mut dict = std::collections::HashMap::<char, char>::new();
    dict.insert(')', '(');
    dict.insert(']', '[');
    dict.insert('}', '{');
    let keys = dict
        .keys()
        .into_iter()
        .collect::<std::collections::HashSet<&char>>();
    let values = dict
        .values()
        .into_iter()
        .collect::<std::collections::HashSet<&char>>();
    let mut stack = Vec::<char>::new();
    for c in s.chars() {
        if !keys.contains(&c) && !values.contains(&c) {
            return false;
        } else if values.contains(&c) {
            stack.push(c)
        } else if keys.contains(&c) {
            let last = stack.pop();
            match last {
                None => {
                    return false;
                }
                Some(val) => {
                    if &val == dict.get(&c).unwrap() {
                        continue;
                    } else {
                        return false;
                    }
                }
            }
        }
    }
    if stack.len() == 0 {
        return true;
    }
    return false;
}

#[test]
fn test_close() {
    assert_eq!(true, is_valid("()".to_string()))
}

///p21
pub fn merge_two_lists(
    list1: Option<Box<ListNode>>,
    list2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut pointer1 = &list1;
    let mut pointer2 = &list2;
    let mut head: Option<Box<ListNode>> = Some(Box::new(ListNode::new(-1)));
    let mut pointer = &mut head;
    loop {
        let next: Box<ListNode>;
        match (pointer1, pointer2) {
            (Some(node1), Some(node2)) => {
                if node1.val <= node2.val {
                    next = Box::new(ListNode::new(node1.val));
                    pointer1 = &node1.next;
                } else {
                    next = Box::new(ListNode::new(node2.val));
                    pointer2 = &node2.next;
                }
            }
            (Some(node1), None) => {
                next = Box::new(ListNode::new(node1.val));
                pointer1 = &node1.next;
            }
            (None, Some(node2)) => {
                next = Box::new(ListNode::new(node2.val));
                pointer2 = &node2.next;
            }
            (None, None) => {
                break;
            }
        }
        match pointer {
            Some(node) => {
                (*node).next = Some(next);
                pointer = &mut (*node).next;
            }
            _ => (),
        }
    }
    return head.unwrap().next;
}

///p22
pub fn generate_parenthesis(n: i32) -> Vec<String> {
    let mut ans = Vec::<String>::new();
    if n <= 0 {
        let mut zero = Vec::<String>::new();
        zero.push("".to_string());
        return zero;
    } else {
        for i in 0..=n - 1 {
            let left = generate_parenthesis(i);
            let right = generate_parenthesis(n - 1 - i);
            for s1 in left.iter() {
                for s2 in right.iter() {
                    let new = "(".to_string() + s1 + ")" + &s2;
                    ans.push(new)
                }
            }
        }
        return ans;
    }
}

///p23
pub fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    let mut pure_value = Vec::<i32>::new();
    for list in lists {
        let mut pointer = &list;
        loop {
            match pointer {
                Some(val) => {
                    pure_value.push(val.val);
                    pointer = &val.next
                }
                None => {
                    break;
                }
            }
        }
    }
    pure_value.sort();
    let mut head = Some(Box::new(ListNode::new(-1)));
    let mut pointer = &mut head;
    for val in pure_value {
        match pointer {
            None => (),
            Some(node) => {
                node.next = Some(Box::new(ListNode::new(val)));
                pointer = &mut node.next;
            }
        }
    }
    return head.unwrap().next;
}

///p24
pub fn swap_pairs(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    head.and_then(|mut first| match first.next {
        None => Some(first),
        Some(mut second) => {
            first.next = swap_pairs(second.next);
            second.next = Some(first);
            Some(second)
        }
    })
}

///p25
pub fn reverse_k_group(mut head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let mut next_head = &mut head;
    // 获取下一轮头结点
    for _ in 0..k {
        if let Some(node) = next_head.as_mut() {
            next_head = &mut node.next;
        } else {
            return head;
        }
    }
    // 获取除本轮结果
    let mut new_head = reverse_k_group(next_head.take(), k);
    // 翻转本轮k个节点
    for _ in 0..k {
        if let Some(mut node) = head {
            head = node.next.take();
            node.next = new_head.take();
            new_head = Some(node);
        }
    }
    new_head
}

///p26
pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    if nums.len() == 0 || nums.len() == 1 {
        return nums.len() as i32;
    }
    let (mut fast, mut slow) = (1, 1);
    while fast < nums.len() {
        if nums[fast] != nums[fast - 1] {
            nums[slow] = nums[fast];
            slow = slow + 1;
        }
        fast = fast + 1;
    }
    return slow as i32;
}

///p27
pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
    let mut pointer = 0;
    for i in 0..nums.len() {
        if nums[i] != val {
            nums[pointer] = nums[i];
            pointer = pointer + 1;
        }
    }
    return pointer as _;
}

///p28
pub fn str_str(haystack: String, needle: String) -> i32 {
    let res = haystack.find(&needle);
    return match res {
        None => -1,
        Some(val) => val as i32,
    };
}

///p29
pub fn divide(dividend: i32, divisor: i32) -> i32 {
    dividend.saturating_div(divisor)
}

///p30
pub fn find_substring(s: String, words: Vec<String>) -> Vec<i32> {
    macro_rules! helper {
        // 哈希统计，为 0 时移除
        ($diff:expr, $s:expr, $cnt:expr) => {
            let t = $s as &str;
            *$diff.entry(t).or_insert(0) += $cnt;
            if *$diff.get(t).unwrap() == 0 {
                $diff.remove(t);
            }
        };
    }
    let mut diff = HashMap::new();
    let (m, n) = (words.len(), words[0].len());
    let mut ans = vec![];
    for idx in 0..n {
        // 仅需要分为 n 组
        if idx + m * n > s.len() {
            break;
        }
        for i in (idx..idx + m * n).step_by(n) {
            helper!(diff, &s[i..i + n], 1);
        }
        for w in words.iter() {
            helper!(diff, w, -1);
        }
        if diff.is_empty() {
            ans.push(idx as i32)
        }
        for i in (idx + n..s.len() - m * n + 1).step_by(n) {
            helper!(diff, &s[i - n..i], -1); // 移除左边
            helper!(diff, &s[i + (m - 1) * n..i + m * n], 1); // 添加右边
            if diff.is_empty() {
                ans.push(i as i32)
            }
        }
        diff.clear();
    }
    ans
}

///p31
pub fn next_permutation(nums: &mut Vec<i32>) {
    let length = nums.len();
    if length <= 1 {
        return;
    }
    for i in (0..length - 1).rev() {
        if nums[i] < nums[i + 1] {
            for j in (i + 1..length).rev() {
                if nums[i] < nums[j] {
                    nums.swap(i, j);
                    nums[i + 1..].reverse();
                    return;
                }
            }
        }
    }
    nums.reverse()
}

///p32
pub fn longest_valid_parentheses(s: String) -> i32 {
    let chars = s.chars().into_iter().collect::<Vec<char>>();
    let length = chars.len();
    let mut dp = vec![0; length];
    for (index, char) in chars.iter().enumerate() {
        match char {
            '(' => (),
            ')' => {
                let pre = chars.get(index - 1);
                match pre {
                    None => (),
                    Some(val) if *val == '(' => {
                        dp[index] = dp.get(index - 2).unwrap_or(&0) + 2;
                    }
                    Some(val) if *val == ')' => {
                        let c = chars.get(index - dp.get(index - 1).unwrap_or(&0) - 1);
                        if c == Some(&'(') {
                            dp[index] = dp.get(index - 1).unwrap_or(&0)
                                + dp.get(index - dp.get(index - 1).unwrap_or(&0) - 2)
                                    .unwrap_or(&0)
                                + 2;
                        }
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }
    return *dp.iter().max().unwrap_or(&0) as i32;
}

///p33
pub fn search_in_rotated(nums: Vec<i32>, target: i32) -> i32 {
    let length = nums.len();
    if length == 0 {
        return -1;
    }
    if length == 1 {
        if nums[0] == target {
            return 0;
        } else {
            return -1;
        }
    }
    let (mut left, mut right) = (0, length - 1);
    while left <= right {
        let mid = (right + left) / 2;
        if nums[mid] == target {
            return mid as i32;
        }
        if nums[0] <= nums[mid] {
            if nums[0] <= target && target < nums[mid] {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if nums[mid] < target && target <= nums[length - 1] {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
}

///p34
pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
    fn bsearch(nums: &[i32], target: i32, offset: usize) -> (i32, i32) {
        let len = nums.len();

        if len == 0 {
            (-1, -1)
        } else {
            let mid = len >> 1;
            if nums[mid] < target {
                bsearch(&nums[mid + 1..len], target, offset + mid + 1)
            } else if nums[mid] > target {
                bsearch(&nums[0..mid], target, offset)
            } else {
                let l_res = bsearch(&nums[0..mid], target, offset);
                let r_res = bsearch(&nums[mid + 1..len], target, offset + mid + 1);
                (
                    if l_res.0 == -1 {
                        (mid + offset) as i32
                    } else {
                        l_res.0
                    },
                    if r_res.1 == -1 {
                        (mid + offset) as i32
                    } else {
                        r_res.1
                    },
                )
            }
        }
    }

    let res = bsearch(&nums, target, 0);
    vec![res.0, res.1]
}
