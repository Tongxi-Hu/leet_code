use std::{
    collections::{HashMap, HashSet},
    i32,
};

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

///p35
pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
    for (index, num) in nums.iter().enumerate() {
        if *num < target {
            continue;
        } else if *num >= target {
            return index as i32;
        }
    }
    return nums.len() as i32;
}

///p36
pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
    let mut rows: Vec<Vec<usize>> = vec![vec![0; 9]; 9];
    let mut cols: Vec<Vec<usize>> = vec![vec![0; 9]; 9];
    let mut boxes: Vec<Vec<Vec<usize>>> = vec![vec![vec![0; 9]; 3]; 3];
    for (row, content) in board.iter().enumerate() {
        for (col, c) in content.iter().enumerate() {
            match c {
                '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' => {
                    let index = c.to_digit(10).unwrap_or(0) as usize - 1;
                    let location = (row / 3, col / 3);
                    rows[row][index] = rows[row][index] + 1;
                    cols[col][index] = cols[col][index] + 1;
                    boxes[location.0][location.1][index] = boxes[location.0][location.1][index] + 1;
                    if rows[row][index] > 1
                        || cols[col][index] > 1
                        || boxes[location.0][location.1][index] > 1
                    {
                        return false;
                    }
                }
                _ => (),
            }
        }
    }
    return true;
}

///p37
struct Msg {
    rows: [u16; 9],
    cols: [u16; 9],
    blks: [[u16; 3]; 3],
    ok: bool,
}
impl Msg {
    fn new() -> Self {
        Self {
            rows: [0u16; 9],
            cols: [0u16; 9],
            blks: [[0u16; 3]; 3],
            ok: false,
        }
    }
    fn flip(&mut self, i: usize, j: usize, d: u8) {
        let d = d - 1;
        self.rows[i] ^= 1 << d;
        self.cols[j] ^= 1 << d;
        self.blks[i / 3][j / 3] ^= 1 << d;
    }
    fn valid_nums(&self, i: usize, j: usize) -> Vec<u8> {
        let mut ans = vec![];
        let mut b = !(self.rows[i] | self.cols[j] | self.blks[i / 3][j / 3]) & 0x1ff;
        while b > 0 {
            ans.push(b.trailing_zeros() as u8 + 1);
            b &= b - 1;
        }
        ans
    }
}
pub fn solve_sudoku(board: &mut Vec<Vec<char>>) {
    fn dfs(spaces: &[(usize, usize)], msg: &mut Msg, board: &mut Vec<Vec<char>>) {
        if spaces.is_empty() {
            msg.ok = true;
            return;
        }
        let (i, j) = spaces[0];
        for d in msg.valid_nums(i, j) {
            if msg.ok {
                break;
            }
            board[i][j] = (d + 0x30) as char;
            msg.flip(i, j, d);
            dfs(&spaces[1..], msg, board);
            msg.flip(i, j, d);
        }
    }
    let mut msg = Msg::new();
    let mut spaces = vec![];
    board.iter().enumerate().for_each(|(i, row)| {
        row.iter().enumerate().for_each(|(j, &c)| {
            if c == '.' {
                spaces.push((i, j));
            } else {
                let d = c as u8 - '0' as u8;
                msg.flip(i, j, d);
            }
        });
    });
    dfs(&spaces[..], &mut msg, board);
}

///p38
pub fn count_and_say(n: i32) -> String {
    match n {
        1 => {
            return "1".to_string();
        }
        val if val > 1 => {
            let last = count_and_say(n - 1);
            let mut ans: Vec<char> = Vec::new();
            let mut last_char: char = ' ';
            let mut len = 0;
            for char in last.chars().into_iter() {
                if char == last_char {
                    len = len + 1;
                    continue;
                } else {
                    if len != 0 {
                        len.to_string().chars().for_each(|item| ans.push(item));
                        ans.push(last_char);
                    }
                    len = 1;
                    last_char = char;
                }
            }
            if len != 0 {
                len.to_string().chars().for_each(|item| ans.push(item));
                ans.push(last_char);
            }
            return ans.iter().collect::<String>();
        }
        _ => {
            return "".to_string();
        }
    }
}

///P39
pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    let mut combine: Vec<i32> = vec![];
    let mut ans: Vec<Vec<i32>> = vec![];
    dfs(&candidates, target, &mut combine, 0, &mut ans);
    return ans;
}
pub fn dfs(
    candidates: &Vec<i32>,
    mut remain: i32,
    combine: &mut Vec<i32>,
    pointer: usize,
    ans: &mut Vec<Vec<i32>>,
) -> () {
    let len = candidates.len();
    if pointer == len {
        return;
    }
    if remain == 0 {
        ans.push(combine.to_vec());
        return;
    }
    dfs(candidates, remain, combine, pointer + 1, ans);
    if remain - candidates[pointer] >= 0 {
        combine.push(candidates[pointer]);
        dfs(
            candidates,
            remain - candidates[pointer],
            combine,
            pointer,
            ans,
        );
        combine.pop();
    }
}

///p40
pub fn combination_sum2(mut candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    candidates.sort();
    let mut combine: Vec<i32> = vec![];
    let mut ans: Vec<Vec<i32>> = vec![];
    dfs2(&candidates, target, &mut combine, 0, &mut ans);
    return ans;
}
pub fn dfs2(
    candidates: &Vec<i32>,
    mut remain: i32,
    combine: &mut Vec<i32>,
    mut pointer: usize,
    ans: &mut Vec<Vec<i32>>,
) -> () {
    let len = candidates.len();

    if remain == 0 {
        ans.push(combine.to_vec());
        return;
    }
    while pointer < len {
        let new_val = candidates[pointer];
        if new_val > remain {
            break;
        }
        combine.push(candidates[pointer]);
        dfs2(
            candidates,
            remain - candidates[pointer],
            combine,
            pointer + 1,
            ans,
        );
        combine.pop();
        while pointer + 1 < len && candidates[pointer + 1] == candidates[pointer] {
            pointer = pointer + 1;
        }
        pointer = pointer + 1;
    }
}

///p41
pub fn first_missing_positive(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    let l = nums.len() as i32;
    for i in 0..nums.len() {
        let mut n = nums[i];
        while n > 0 && n <= l && nums[(n - 1) as usize] != n {
            std::mem::swap(&mut nums[(n - 1) as usize], &mut n);
        }
    }
    for i in 0..l {
        if nums[i as usize] != i + 1 {
            return i + 1;
        }
    }
    l + 1
}

///p42
pub fn trap(height: Vec<i32>) -> i32 {
    let length = height.len();
    let mut left_max = vec![0; length];
    left_max[0] = height[0];
    let mut right_max = vec![0; length];
    right_max[length - 1] = height[length - 1];
    let mut vol = vec![0; length];
    for l in 0..length {
        let r = length - 1 - l;
        if r != length - 1 {
            right_max[r] = right_max[r + 1].max(height[r]);
        };
        if l != 0 {
            left_max[l] = left_max[l - 1].max(height[l])
        }
    }
    for i in 0..length {
        vol[i] = left_max[i].min(right_max[i]) - height[i];
    }

    return vol.into_iter().reduce(|acc, v| acc + v).unwrap_or(0);
}

///p43
pub fn multiply(num1: String, num2: String) -> String {
    let mut mul: Vec<i32> = vec![0; num1.len() + num2.len()];
    let c1: Vec<i32> = num1
        .chars()
        .rev()
        .map(|x| (x as u8 - '0' as u8) as i32)
        .collect();
    let c2: Vec<i32> = num2
        .chars()
        .rev()
        .map(|x| (x as u8 - '0' as u8) as i32)
        .collect();
    for i in 0..c1.len() {
        for j in 0..c2.len() {
            mul[i + j] += c1[i] * c2[j];
        }
    }
    let mut add = 0i32;
    for i in 0..mul.len() {
        let m = (mul[i] + add) % 10;
        add = (mul[i] + add) / 10;
        mul[i] = m;
    }
    mul.iter()
        .rev()
        .enumerate()
        .skip_while(|(k, x)| x == &&0 && *k != mul.len() - 1)
        .map(|(_, x)| (*x as u8 + '0' as u8) as char)
        .collect()
}

///p44
pub fn is_match_str(s: String, p: String) -> bool {
    let s = s.chars().collect::<Vec<char>>();
    let p = p.chars().collect::<Vec<char>>();
    let mut dp: Vec<Vec<bool>> = vec![vec![false; p.len() + 1]; s.len() + 1];
    dp[0][0] = true;
    for i in 1..=p.len() {
        if p[i - 1] == '*' {
            dp[0][i] = true;
        } else {
            break;
        }
    }
    for i in 1..=s.len() {
        for j in 1..=p.len() {
            if p[j - 1] == '*' {
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            } else if p[j - 1] == '?' || s[i - 1] == p[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }
    return dp[s.len()][p.len()];
}

///p45
pub fn jump(nums: Vec<i32>) -> i32 {
    let mut dp: Vec<i32> = vec![i32::MAX; nums.len()];
    dp[0] = 0;
    for i in 1..dp.len() {
        for j in 0..i {
            if j + nums[j] as usize >= i {
                dp[i] = dp[i].min(dp[j] + 1);
            }
        }
    }
    return dp[nums.len() - 1];
}

///p46
pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut ans: Vec<Vec<i32>> = vec![];
    let mut temp: Vec<i32> = vec![];
    fn dfs(nums: &Vec<i32>, temp: &mut Vec<i32>, ans: &mut Vec<Vec<i32>>) -> () {
        let length = nums.len();
        if temp.len() == length {
            ans.push(temp.to_vec());
            return;
        }
        for i in nums {
            if temp.contains(i) {
                continue;
            } else {
                temp.push(*i);
                dfs(nums, temp, ans);
                temp.pop();
            }
        }
    }
    dfs(&nums, &mut temp, &mut ans);
    return ans;
}

///p47
pub fn permute_unique(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut nums = nums;
    nums.sort();
    let mut ans: Vec<Vec<i32>> = vec![];
    let mut temp: Vec<i32> = vec![];
    fn dfs(
        nums: &Vec<i32>,
        used: &mut Vec<bool>,
        temp: &mut Vec<i32>,
        ans: &mut Vec<Vec<i32>>,
    ) -> () {
        let length = nums.len();
        if temp.len() == length {
            ans.push(temp.to_vec());
            return;
        }
        for (i, &x) in nums.iter().enumerate() {
            if used[i] || i > 0 && !used[i - 1] && nums[i] == nums[i - 1] {
                continue;
            }
            temp.push(x);
            used[i] = true;
            dfs(nums, used, temp, ans);
            used[i] = false;
            temp.pop();
        }
    }
    dfs(&nums, &mut vec![false; nums.len()], &mut temp, &mut ans);
    return ans;
}

///p48
/// [3,3]->[3,0] [2,2]->[2,1] [1,2]->[2,2] [0,1]->[1,3] [3,2]->[2,0] [3,1]->[1,0] [2,3]->[3,1]
pub fn rotate(matrix: &mut Vec<Vec<i32>>) {
    let length = matrix.len();
    for i in 0..length / 2 {
        for j in 0..length {
            let temp = matrix[i][j];
            matrix[i][j] = matrix[length - 1 - i][j];
            matrix[length - 1 - i][j] = temp;
        }
    }
    for i in 0..length {
        for j in 0..i {
            let temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}

///p49
pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
    let mut dict: HashMap<String, Vec<String>> = HashMap::new();
    for str in strs {
        let mut chars = str.chars().collect::<Vec<char>>();
        chars.sort();
        let key = chars.iter().collect::<String>();
        match dict.get_mut(&key) {
            None => {
                dict.insert(key, vec![str.clone()]);
            }
            Some(val) => val.push(str.clone()),
        }
    }
    return dict.into_values().collect::<Vec<Vec<String>>>();
}
///p50
pub fn my_pow(x: f64, n: i32) -> f64 {
    fn quick_mul(x: f64, n: i32) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let y = quick_mul(x, n / 2);
        if n % 2 == 0 {
            return y * y;
        } else {
            return y * y * x;
        }
    }
    if n > 0 {
        quick_mul(x, n)
    } else {
        1.0 / quick_mul(x, -n)
    }
}

///p51
pub fn solve_n_queens(n: i32) -> Vec<Vec<String>> {
    let n = n as usize;
    let mut solutions: Vec<Vec<String>> = vec![];
    let mut queens = vec![std::usize::MAX; n];
    let mut col: Vec<usize> = vec![];
    let mut diag1: Vec<usize> = vec![];
    let mut diag2: Vec<usize> = vec![];

    fn backtrack(
        solutions: &mut Vec<Vec<String>>,
        queens: &mut Vec<usize>,
        n: usize,
        row: usize,
        col: &mut Vec<usize>,
        diag1: &mut Vec<usize>,
        diag2: &mut Vec<usize>,
    ) -> () {
        if row == n {
            solutions.push(generate_board(queens, n))
        } else {
            for i in 0..n {
                if col.contains(&i) {
                    continue;
                }
                let diagonal1 = row - i;
                if diag1.contains(&diagonal1) {
                    continue;
                }
                let diagonal2 = row + i;
                if diag2.contains(&diagonal2) {
                    continue;
                }
                queens[row] = i;
                col.push(i);
                diag1.push(diagonal1);
                diag2.push(diagonal2);
                backtrack(solutions, queens, n, row + 1, col, diag1, diag2);
                queens[row] = core::usize::MAX;
                col.pop();
                diag1.pop();
                diag2.pop();
            }
        }
    }

    fn generate_board(queens: &mut Vec<usize>, n: usize) -> Vec<String> {
        let mut board = vec![];
        for i in 0..n {
            let mut row: Vec<char> = vec!['.'; n];
            row[queens[i]] = 'Q';
            board.push(row.iter().collect::<String>())
        }
        return board;
    }

    backtrack(
        &mut solutions,
        &mut queens,
        n,
        0,
        &mut col,
        &mut diag1,
        &mut diag2,
    );
    return solutions;
}

///p52
pub fn total_n_queens(n: i32) -> i32 {
    let n = n as usize;
    let mut solutions: Vec<Vec<usize>> = vec![];
    let mut queens = vec![std::usize::MAX; n];
    let mut col: Vec<usize> = vec![];
    let mut diag1: Vec<usize> = vec![];
    let mut diag2: Vec<usize> = vec![];

    fn backtrack(
        solutions: &mut Vec<Vec<usize>>,
        queens: &mut Vec<usize>,
        n: usize,
        row: usize,
        col: &mut Vec<usize>,
        diag1: &mut Vec<usize>,
        diag2: &mut Vec<usize>,
    ) -> () {
        if row == n {
            solutions.push(queens.to_vec())
        } else {
            for i in 0..n {
                if col.contains(&i) {
                    continue;
                }
                let diagonal1 = row - i;
                if diag1.contains(&diagonal1) {
                    continue;
                }
                let diagonal2 = row + i;
                if diag2.contains(&diagonal2) {
                    continue;
                }
                queens[row] = i;
                col.push(i);
                diag1.push(diagonal1);
                diag2.push(diagonal2);
                backtrack(solutions, queens, n, row + 1, col, diag1, diag2);
                queens[row] = core::usize::MAX;
                col.pop();
                diag1.pop();
                diag2.pop();
            }
        }
    }

    backtrack(
        &mut solutions,
        &mut queens,
        n,
        0,
        &mut col,
        &mut diag1,
        &mut diag2,
    );
    return solutions.len() as i32;
}

///p53
pub fn max_sub_array(nums: Vec<i32>) -> i32 {
    let length = nums.len();
    let mut max = vec![core::i32::MAX; length];
    max[0] = nums[0];
    for i in 1..length {
        max[i] = nums[i].max(max[i - 1] + nums[i]);
    }
    return max.into_iter().reduce(|max, cur| max.max(cur)).unwrap_or(0);
}

///p54
pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let mut res = Vec::new();
    if matrix.len() == 0 {
        return res;
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let iter_num = std::cmp::min(rows, cols) / 2; //迭代次数  如果是奇数的话 最后一次就是一个行或列 单独考虑

    for iter in 0..iter_num {
        //上面遍历
        for col_id in iter..(cols - iter - 1) {
            res.push(matrix[iter][col_id]);
        }
        //右边遍历
        for row_id in iter..(rows - iter - 1) {
            res.push(matrix[row_id][cols - iter - 1]);
        }
        //下方遍历
        for col_id in (iter + 1..cols - iter).rev() {
            res.push(matrix[rows - iter - 1][col_id]);
        }
        //左边遍历
        for row_id in (iter + 1..rows - iter).rev() {
            res.push(matrix[row_id][iter]);
        }
    }

    //考虑剩下单行的情况
    if rows <= cols && (rows & 1 > 0) {
        for col_id in iter_num..cols - iter_num {
            res.push(matrix[iter_num][col_id]);
        }
    }

    //考虑剩下单列的情况
    if cols < rows && (cols & 1 > 0) {
        for row_id in iter_num..rows - iter_num {
            res.push(matrix[row_id][iter_num]);
        }
    }

    res
}

///p55
pub fn can_jump(nums: Vec<i32>) -> bool {
    let mut max_range = 0;
    for i in 0..nums.len() {
        if i <= max_range && i + nums[i] as usize > max_range {
            max_range = i + (nums[i] as usize);
        }
    }
    return max_range >= nums.len() - 1;
}

///p56
pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut intervals = intervals;
    intervals.sort_by(|first, second| {
        if first[0] > second[0] {
            return std::cmp::Ordering::Greater;
        } else if first[0] == second[0] {
            return std::cmp::Ordering::Equal;
        } else {
            return std::cmp::Ordering::Less;
        }
    });

    let mut ranges: Vec<Vec<i32>> = vec![intervals[0].to_vec()];
    for range in intervals {
        let cur = ranges.last_mut().unwrap();
        if range[0] > cur[1] {
            ranges.push(range.to_vec());
        } else {
            let new_last = range[1].max(cur[1]);
            cur[1] = new_last
        }
    }

    return ranges;
}

///p57
pub fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
    let mut new_intervals: Vec<Vec<i32>> = vec![];
    let mut start = new_interval[0];
    let mut end = new_interval[1];
    for interval in intervals {
        if interval[1] < new_interval[0] {
            new_intervals.push(interval.to_vec());
        } else if interval[0] > new_interval[1] {
            if start != core::i32::MIN {
                new_intervals.push(vec![start, end]);
                start = core::i32::MIN
            }
            new_intervals.push(interval.to_vec());
        } else {
            if interval[0] < start {
                start = interval[0]
            }
            if interval[1] > end {
                end = interval[1];
            }
        }
    }
    if start != core::i32::MIN {
        new_intervals.push(vec![start, end]);
    }
    return new_intervals;
}

///p58
pub fn length_of_last_word(s: String) -> i32 {
    s.split_whitespace().last().unwrap_or("").len() as i32
}

///p59
pub fn generate_matrix(n: i32) -> Vec<Vec<i32>> {
    let mut res = vec![vec![0; n as usize]; n as usize];
    let (mut startX, mut startY, mut offset): (usize, usize, usize) = (0, 0, 1);
    let mut loopIdx = n / 2;
    let mid: usize = loopIdx as usize;
    let mut count = 1;
    let (mut i, mut j): (usize, usize) = (0, 0);
    while loopIdx > 0 {
        i = startX;
        j = startY;

        while j < (startY + (n as usize) - offset) {
            res[i][j] = count;
            count += 1;
            j += 1;
        }

        while i < (startX + (n as usize) - offset) {
            res[i][j] = count;
            count += 1;
            i += 1;
        }

        while j > startY {
            res[i][j] = count;
            count += 1;
            j -= 1;
        }

        while i > startX {
            res[i][j] = count;
            count += 1;
            i -= 1;
        }

        startX += 1;
        startY += 1;
        offset += 2;
        loopIdx -= 1;
    }

    if n % 2 == 1 {
        res[mid][mid] = count;
    }
    res
}

///p60
pub fn get_permutation(n: i32, k: i32) -> String {
    let n = n as usize;
    let k = k as usize;
    let elements = (1..=n).collect();
    let mut all_permutation: Vec<Vec<usize>> = vec![];
    let mut cur = vec![];
    let mut used = vec![false; n];

    fn generate_all_permutation(
        ans: &mut Vec<Vec<usize>>,
        elements: &Vec<usize>,
        cur: &mut Vec<usize>,
        used: &mut Vec<bool>,
        limit: usize,
    ) {
        if ans.len() > limit {
            return;
        }
        let index = used
            .iter()
            .enumerate()
            .filter(|(_, val)| **val == false)
            .map(|item| item.0)
            .collect::<Vec<usize>>();

        if index.len() == 0 {
            ans.push(cur.to_vec());
        } else {
            for i in index {
                used[i] = true;
                cur.push(elements[i]);
                generate_all_permutation(ans, elements, cur, used, limit);
                cur.pop();
                used[i] = false;
            }
        }
    }

    generate_all_permutation(&mut all_permutation, &elements, &mut cur, &mut used, k);

    return all_permutation
        .get(k - 1 as usize)
        .unwrap()
        .into_iter()
        .map(|item| return item.to_string())
        .collect::<String>();
}

#[test]
fn test_get_permutation() {
    assert_eq!("213".to_string(), get_permutation(3, 3))
}

///p61
pub fn rotate_right(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    if head.is_none() || k == 0 {
        return head;
    }
    let mut head = head;
    let mut ptr = &head;
    let mut len = 0;
    while let Some(ref t) = ptr {
        ptr = &t.next;
        len += 1;
    }
    let k = k % len;
    if k == 0 {
        return head;
    }
    let mut ptr = &mut head;
    for _ in 1..len - k {
        ptr = &mut ptr.as_mut().unwrap().next;
    }
    let mut new_head = ptr.as_mut().unwrap().next.take();
    let mut tail = &mut new_head;
    while tail.is_some() && tail.as_ref().unwrap().next.is_some() {
        tail = &mut tail.as_mut().unwrap().next;
    }
    tail.as_mut().unwrap().next = head;
    new_head
}

///p62
pub fn unique_paths(m: i32, n: i32) -> i32 {
    let m = m as usize;
    let n = n as usize;
    let mut dp: Vec<Vec<usize>> = vec![vec![0; n]; m];
    for l in 0..n {
        dp[0][l] = 1;
    }
    for r in 0..m {
        dp[r][0] = 1;
    }
    for r in 1..m {
        for l in 1..n {
            dp[r][l] = dp[r][l - 1] + dp[r - 1][l]
        }
    }
    return dp[m - 1][n - 1] as i32;
}

///p63
pub fn unique_paths_with_obstacles(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let r = obstacle_grid.len();
    let c = obstacle_grid[0].len();
    let mut dp: Vec<Vec<usize>> = vec![vec![0; c]; r];
    for i in 0..r {
        for j in 0..c {
            if obstacle_grid[i][j] == 1 {
                dp[i][j] = 0;
            } else {
                if i == 0 {
                    if j == 0 {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = dp[i][j - 1];
                    }
                } else if j == 0 {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
    }
    return dp[r - 1][c - 1] as i32;
}

///p64
pub fn min_path_sum(grid: Vec<Vec<i32>>) -> i32 {
    let r = grid.len();
    let c = grid[0].len();
    let mut dp_min_sum = vec![vec![0; c]; r];
    for i in 0..r {
        for j in 0..c {
            match (i, j) {
                (0, 0) => dp_min_sum[i][j] = grid[i][j],
                (0, _) => dp_min_sum[i][j] = grid[i][j] + dp_min_sum[i][j - 1],
                (_, 0) => dp_min_sum[i][j] = grid[i][j] + dp_min_sum[i - 1][j],
                _ => dp_min_sum[i][j] = grid[i][j] + dp_min_sum[i - 1][j].min(dp_min_sum[i][j - 1]),
            }
        }
    }
    return dp_min_sum[r - 1][c - 1];
}

///p65
enum State {
    Start,
    Sign,
    Integer,
    Dot,
    InitialDot,
    Decimal,
    E,
    ESign,
    EInteger,
    Illegal,
}

impl State {
    fn next(&self, next: Input) -> Self {
        match self {
            State::Start => match next {
                Input::Digit => State::Integer,
                Input::Dot => State::InitialDot,
                Input::Sign => State::Sign,
                _ => State::Illegal,
            },
            State::Sign => match next {
                Input::Digit => State::Integer,
                Input::Dot => State::InitialDot,
                _ => State::Illegal,
            },
            State::Integer => match next {
                Input::Digit => State::Integer,
                Input::Dot => State::Dot,
                Input::E => State::E,
                _ => State::Illegal,
            },
            State::Dot => match next {
                Input::Digit => State::Decimal,
                Input::E => State::E,
                _ => State::Illegal,
            },
            State::InitialDot => match next {
                Input::Digit => State::Decimal,
                _ => State::Illegal,
            },
            State::Decimal => match next {
                Input::Digit => State::Decimal,
                Input::E => State::E,
                _ => State::Illegal,
            },
            State::E => match next {
                Input::Digit => State::EInteger,
                Input::Sign => State::ESign,
                _ => State::Illegal,
            },
            State::ESign => match next {
                Input::Digit => State::EInteger,
                _ => State::Illegal,
            },
            State::EInteger => match next {
                Input::Digit => State::EInteger,
                _ => State::Illegal,
            },
            _ => State::Illegal,
        }
    }

    fn accept(&self) -> bool {
        match self {
            State::Integer | State::Dot | State::Decimal | State::EInteger => true,
            _ => false,
        }
    }
}

enum Input {
    Digit,
    Sign,
    Dot,
    E,
    Other,
}

impl<T: Into<char>> From<T> for Input {
    fn from(c: T) -> Self {
        match c.into() {
            '0'..='9' => Input::Digit,
            '.' => Input::Dot,
            '+' | '-' => Input::Sign,
            'e' | 'E' => Input::E,
            _ => Input::Other,
        }
    }
}

pub fn is_number(s: String) -> bool {
    let mut state = State::Start;
    s.chars().for_each(|c| state = state.next(c.into()));
    state.accept()
}

///p66
pub fn plus_one(digits: Vec<i32>) -> Vec<i32> {
    let mut digits = digits;
    let mut header: i32 = 1;
    digits.reverse();
    for i in digits.iter_mut() {
        let new = (*i) + header;
        *i = new % 10;
        header = new / 10;
    }
    digits.reverse();
    if header != 0 {
        digits.insert(0, header);
    }
    return digits;
}

///p67
pub fn add_binary(a: String, b: String) -> String {
    let char_a = a.chars().collect::<Vec<char>>();
    let char_b = b.chars().collect::<Vec<char>>();
    let mut header = '0';
    let length = char_a.len().max(char_b.len());
    let val = char_a
        .iter()
        .rev()
        .chain(core::iter::repeat(&'0'))
        .zip(char_b.iter().rev().chain(core::iter::repeat(&'0')))
        .take(length)
        .map(|(val_a, val_b)| match (val_a, val_b, &mut header) {
            ('0', '0', '0') => {
                header = '0';
                return '0';
            }
            ('1', '1', '1') => {
                header = '1';
                return '1';
            }
            ('1', '1', '0') | ('1', '0', '1') | ('0', '1', '1') => {
                header = '1';
                return '0';
            }
            _ => {
                header = '0';
                return '1';
            }
        })
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();

    if header != '0' {
        return header.to_string() + &val;
    } else {
        return val;
    }
}

///p68
pub fn full_justify(words: Vec<String>, max_width: i32) -> Vec<String> {
    let L = words.len();
    let max_width = max_width as usize;
    let mut queue = Vec::new();
    let mut i = 0;
    let mut count = 0;
    let mut tmp = Vec::new();

    while i < L {
        count = 0;
        tmp = Vec::new();

        while i < L && count < max_width {
            // 按max_with将单词分组
            if count + words[i].len() > max_width {
                break;
            }

            count += (words[i].len() + 1);
            tmp.push(words[i].clone());
            i += 1;
        }

        if i == L {
            // 最后一行，在单词后增加空格
            let last_line = tmp[..].join(&" ");
            let len = last_line.len();
            queue.push(last_line + &" ".repeat(max_width - len));
            break;
        }

        if tmp.len() == 1 {
            // 分组长度为1，在单词后增加空格
            queue.push(tmp[..].join(&"") + &" ".repeat(max_width - count + 1));
        } else {
            let T = tmp.len() - 1;
            let paddings = max_width - (count - T - 1);
            let space = paddings / T; // 每个单词后应该增加的平均空格数
            let mut extra = (paddings % T) as i32; // 如果每个单词后增加的空格数不能被平均，将额外空格分摊到前几个单词后面
            let res = tmp
                .into_iter()
                .enumerate()
                .map(|(i, x)| {
                    let times = space + if extra <= 0 { 0 } else { 1 };
                    extra -= 1;

                    if i == T {
                        x
                    } else {
                        x + &" ".repeat(times)
                    }
                })
                .collect::<String>();
            queue.push(res);
        }
    }

    queue
}

///p69
pub fn my_sqrt(x: i32) -> i32 {
    let x = x as usize;
    let (mut l, mut r) = (0, x);
    let mut ans: i32 = -1;
    while l <= r {
        let mid = (l + r) / 2;
        if mid * mid <= x {
            ans = mid as i32;
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return ans;
}

///p70
pub fn climb_stairs(n: i32) -> i32 {
    let n = n as usize;
    if n == 1 {
        return 1;
    }
    let mut dp = vec![0; n];
    dp[0] = 1;
    dp[1] = 2;
    for step in 2..n {
        dp[step] = dp[step - 1] + dp[step - 2];
    }
    return dp[n - 1];
}

///71
pub fn simplify_path(path: String) -> String {
    let paths = path
        .split('/')
        .map(|item| item.to_string())
        .collect::<Vec<String>>();
    let mut stack: Vec<String> = vec![];
    for item in paths {
        match item.as_str() {
            "" => (),
            ".." => {
                stack.pop();
            }
            "." => (),
            val => {
                stack.push(val.to_string());
            }
        }
    }
    if stack.len() == 0 {
        return "/".to_owned();
    }
    return stack
        .into_iter()
        .map(|item| return "/".to_string() + &item)
        .collect::<String>();
}

///p72
pub fn min_distance(word1: String, word2: String) -> i32 {
    let n = word1.len();
    let m = word2.len();
    if n * m == 0 {
        return (m + n) as i32;
    }

    let mut dp: Vec<Vec<usize>> = vec![vec![0; m + 1]; n + 1];

    for i in 0..n + 1 {
        dp[i][0] = i;
    }

    for j in 0..m + 1 {
        dp[0][j] = j;
    }

    for i in 1..n + 1 {
        for j in 1..m + 1 {
            let left = dp[i - 1][j] + 1;
            let down = dp[i][j - 1] + 1;
            let mut left_down = dp[i - 1][j - 1];
            if word1.chars().nth(i - 1) != word2.chars().nth(j - 1) {
                left_down = left_down + 1;
            }
            dp[i][j] = left.min(down.min(left_down));
        }
    }
    return dp[n][m] as i32;
}

///p73
pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
    use std::collections::HashSet;
    let mut rows: HashSet<usize> = HashSet::new();
    let mut cols: HashSet<usize> = HashSet::new();
    for (row, item) in matrix.iter().enumerate() {
        for (col, val) in item.iter().enumerate() {
            if *val == 0 {
                rows.insert(row);
                cols.insert(col);
            }
        }
    }
    for row in rows.iter() {
        matrix[*row].iter_mut().for_each(|val| {
            *val = 0;
        })
    }

    for col in cols.iter() {
        matrix.iter_mut().for_each(|item| item[*col] = 0)
    }
}

///p74
pub fn search_matrix(matrix: Vec<Vec<i32>>, target: i32) -> bool {
    use std::collections::HashSet;
    let mut pool: HashSet<(usize, usize)> = HashSet::new();
    pool.insert((0, 0));

    fn has_target(
        pool: &HashSet<(usize, usize)>,
        matrix: &Vec<Vec<i32>>,
        target: i32,
    ) -> (bool, HashSet<(usize, usize)>) {
        let rows = matrix.len();
        let cols = matrix[0].len();
        let mut new_pool: HashSet<(usize, usize)> = HashSet::new();
        for location in pool.iter() {
            if matrix[location.0][location.1] == target {
                return (true, HashSet::new());
            } else if matrix[location.0][location.1] < target {
                if location.1 + 1 < cols {
                    new_pool.insert((location.0, location.1 + 1));
                }
                if location.0 + 1 < rows {
                    new_pool.insert((location.0 + 1, location.1));
                }
            }
        }
        return (false, new_pool);
    }

    while pool.len() != 0 {
        let (has, new_pool) = has_target(&pool, &matrix, target);
        if has == true {
            return true;
        }
        if new_pool.len() == 0 {
            return false;
        }
        pool = new_pool;
    }

    return false;
}

///p75
pub fn sort_colors(nums: &mut Vec<i32>) {
    if nums.len() <= 1 {
        return;
    }
    let mut eq_l = 0;
    let mut eq_r = 0;
    for i in 0..nums.len() {
        if nums[i] < 1 {
            nums.swap(i, eq_l);
            if eq_l < eq_r {
                nums.swap(eq_r, i);
            }
            eq_l += 1;
            eq_r += 1;
        } else if nums[i] == 1 {
            nums.swap(i, eq_r);
            eq_r += 1;
        }
    }
}

///p76
pub fn min_window(S: String, t: String) -> String {
    let s = S.as_bytes();
    let m = s.len();
    let mut ans_left = 0;
    let mut ans_right = m;
    let mut left = 0;
    let mut less = 0;
    let mut cnt_s = [0; 128]; // s 子串字母的出现次数
    let mut cnt_t = [0; 128]; // t 中字母的出现次数
    for c in t.bytes() {
        let c = c as usize;
        if cnt_t[c] == 0 {
            less += 1; // 有 less 种字母的出现次数 < t 中的字母出现次数
        }
        cnt_t[c] += 1;
    }
    for (right, &c) in s.iter().enumerate() {
        // 移动子串右端点
        let c = c as usize;
        cnt_s[c] += 1; // 右端点字母移入子串
        if cnt_s[c] == cnt_t[c] {
            less -= 1; // c 的出现次数从 < 变成 >=
        }
        while less == 0 {
            // 涵盖：所有字母的出现次数都是 >=
            if right - left < ans_right - ans_left {
                // 找到更短的子串
                ans_left = left; // 记录此时的左右端点
                ans_right = right;
            }
            let x = s[left] as usize; // 左端点字母
            if cnt_s[x] == cnt_t[x] {
                less += 1; // x 的出现次数从 >= 变成 <
            }
            cnt_s[x] -= 1; // 左端点字母移出子串
            left += 1;
        }
    }
    if ans_right < m {
        unsafe { String::from_utf8_unchecked(s[ans_left..=ans_right].to_vec()) }
    } else {
        String::new()
    }
}

///p77
pub fn combine(n: i32, k: i32) -> Vec<Vec<i32>> {
    let mut result: Vec<Vec<i32>> = vec![];
    let mut current: Vec<i32> = vec![];

    fn dfs(begin: i32, n: i32, k: i32, current: &mut Vec<i32>, result: &mut Vec<Vec<i32>>) {
        if current.len() + ((n - begin + 1) as usize) < k as usize {
            return;
        }
        if current.len() as i32 == k {
            result.push(current.clone());
            return;
        }
        current.push(begin);
        dfs(begin + 1, n, k, current, result);
        current.pop();
        dfs(begin + 1, n, k, current, result);
    }
    dfs(1, n, k, &mut current, &mut result);
    return result;
}

///p78
pub fn subsets(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut result: Vec<Vec<i32>> = vec![];
    let mut pointer: usize = 0;
    let length = nums.len();
    let mut current: Vec<i32> = vec![];

    fn dfs(
        pointer: usize,
        length: usize,
        nums: &Vec<i32>,
        current: &mut Vec<i32>,
        result: &mut Vec<Vec<i32>>,
    ) {
        if pointer >= length {
            result.push(current.to_vec());
            return;
        }
        current.push(nums[pointer]);
        dfs(pointer + 1, length, nums, current, result);
        current.pop();
        dfs(pointer + 1, length, nums, current, result)
    }

    dfs(0, length, &nums, &mut current, &mut result);
    return result;
}

///p79
pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {
    let chars: Vec<char> = word.chars().collect();
    fn check(
        i: usize,
        j: usize,
        board: &Vec<Vec<char>>,
        visited: &mut Vec<Vec<bool>>,
        chars: &Vec<char>,
        k: usize,
    ) -> bool {
        if board[i][j] != chars[k] {
            return false;
        } else if chars.len() - 1 == k {
            return true;
        }
        visited[i][j] = true;
        let direction: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let mut result = false;
        for dir in direction {
            let (new_i, new_j) = (i + dir.0 as usize, j + dir.1 as usize);
            if new_i >= 0 && new_i < board.len() && new_j >= 0 && new_j < board[0].len() {
                if visited[new_i][new_j] != true {
                    let next_result = check(new_i, new_j, board, visited, chars, k + 1);
                    if next_result == true {
                        result = true;
                        break;
                    }
                }
            }
        }
        visited[i][j] = false;
        return result;
    }
    let mut visited = vec![vec![false; board[0].len()]; board.len()];
    for i in 0..board.len() {
        for j in 0..board[0].len() {
            let flag = check(i, j, &board, &mut visited, &chars, 0);
            if flag == true {
                return true;
            }
        }
    }
    return false;
}

///p80
pub fn remove_duplicates_2(nums: &mut Vec<i32>) -> i32 {
    let length = nums.len();
    if length <= 2 {
        return length as i32;
    }
    let (mut slow, mut fast) = (2, 2);
    while fast < length {
        if nums[slow - 2] != nums[fast] {
            nums[slow] = nums[fast];
            slow = slow + 1;
        }
        fast = fast + 1;
    }
    return slow as i32;
}

///p81
pub fn search(nums: Vec<i32>, target: i32) -> bool {
    for i in 0..nums.len() {
        if nums[i] == target {
            return true;
        }
    }
    return false;
}

///p82
pub fn delete_duplicates(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut res = Some(Box::new(ListNode::new(0)));
    let mut p = res.as_mut().unwrap();
    let mut pre = 101;
    while let Some(mut node) = head {
        head = node.next.take();
        if (head.is_some() && head.as_ref().unwrap().val == node.val) || node.val == pre {
            pre = node.val;
        } else {
            pre = node.val;
            p.next = Some(node);
            p = p.next.as_mut().unwrap();
        }
    }
    res.and_then(|node| node.next)
}

///p83
pub fn delete_duplicates_2(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut dummy = Some(Box::new(ListNode { val: 0, next: None }));
    let mut p = &mut dummy;
    let mut pre_val = core::i32::MAX;
    while let Some(mut node) = head {
        head = node.next.take();
        if node.val != pre_val {
            pre_val = node.val;
            p.as_mut().unwrap().next = Some(node);
            p = &mut p.as_mut().unwrap().next;
        }
    }
    return dummy.and_then(|node| node.next);
}

///p84
pub fn largest_rectangle_area(heights: Vec<i32>) -> i32 {
    let mut left: Vec<i32> = vec![-1; heights.len()];
    let mut mono_stack: Vec<usize> = vec![];
    for i in 0..heights.len() {
        while !mono_stack.is_empty() && heights[*mono_stack.last().unwrap()] >= heights[i] {
            mono_stack.pop();
        }
        if !mono_stack.is_empty() {
            left[i] = *mono_stack.last().unwrap() as i32;
        }
        mono_stack.push(i);
    }
    let mut right: Vec<i32> = vec![heights.len() as i32; heights.len()];
    mono_stack = vec![];
    for i in (0..heights.len()).rev() {
        while !mono_stack.is_empty() && heights[*mono_stack.last().unwrap()] >= heights[i] {
            mono_stack.pop();
        }
        if !mono_stack.is_empty() {
            right[i] = *mono_stack.last().unwrap() as i32;
        }
        mono_stack.push(i);
    }
    return left
        .iter()
        .zip(right.iter())
        .zip(heights.iter())
        .fold(0, |acc, cur| {
            return acc.max(cur.1 * (cur.0 .1 - cur.0 .0 - 1));
        });
}

///p85
pub fn maximal_rectangle(matrix: Vec<Vec<char>>) -> i32 {
    let row = matrix.len();
    let column = matrix[0].len();
    let mut res = i32::MIN;

    let mut horizon = vec![vec![0; column + 1]; row + 1];
    for i in 1..=row {
        for j in 1..=column {
            if matrix[i - 1][j - 1] == '1' {
                horizon[i][j] = horizon[i - 1][j] + 1;
            }
        }
    }

    for i in 1..=row {
        let mut stack: Vec<usize> = Vec::new();
        let mut left: Vec<usize> = (0..=column).collect(); // left[j] 存以horizon[i][j]为最小值的最长子数组的最左端
        for j in 1..=column {
            let mut left_limit = j; // 表示以*stack.last().unwrap()为最小值的最左边的idx
            while !stack.is_empty() && horizon[i][*stack.last().unwrap()] >= horizon[i][j] {
                // 递增栈
                left_limit = stack.pop().unwrap();
            }
            left[j] = left[left_limit];
            stack.push(j);
        }
        let mut right: Vec<usize> = (0..=column).collect(); // 存以horizon[i][j]为最小值的最长子数组的最右端
        let mut stack: Vec<usize> = Vec::new();
        for j in (1..=column).rev() {
            let mut right_limit = j; // 表示以*stack.last().unwrap()为最小值的最左边的idx
            while !stack.is_empty() && horizon[i][*stack.last().unwrap()] >= horizon[i][j] {
                // 递增栈
                right_limit = stack.pop().unwrap();
            }
            right[j] = right[right_limit];
            stack.push(j);
        }
        for j in 1..=column {
            res = res.max((right[j] - left[j] + 1) as i32 * horizon[i][j]);
        }
    }
    res
}

///p86
pub fn partition(head: Option<Box<ListNode>>, x: i32) -> Option<Box<ListNode>> {
    let mut head = &head;
    let mut small_head = Some(Box::new(ListNode {
        val: core::i32::MIN,
        next: None,
    }));
    let mut small = &mut small_head;
    let mut large_head = Some(Box::new(ListNode { val: x, next: None }));
    let mut large = &mut large_head;
    while let Some(ref node) = head {
        if node.val < x {
            small.as_mut().unwrap().next = Some(Box::new(ListNode {
                val: node.val,
                next: None,
            }));
            small = &mut small.as_mut().unwrap().next;
        } else {
            large.as_mut().unwrap().next = Some(Box::new(ListNode {
                val: node.val,
                next: None,
            }));
            large = &mut large.as_mut().unwrap().next;
        }
        head = &head.as_ref().unwrap().next;
    }

    if large_head.as_ref().unwrap().next.is_some() {
        small.as_mut().unwrap().next = large_head.as_mut().unwrap().next.take();
    }

    return small_head.as_mut().unwrap().next.take();
}

///p87
pub fn is_scramble(s1: String, s2: String) -> bool {
    let n = s1.len();
    let mut records = vec![vec![vec![None; n + 1]; n]; n];
    fn check(
        s1: &str,
        beg1: usize,
        end1: usize,
        s2: &str,
        beg2: usize,
        end2: usize,
        records: &mut Vec<Vec<Vec<Option<bool>>>>,
    ) -> bool {
        let len = end1 - beg1;
        if records[beg1][beg2][len].is_some() {
            return records[beg1][beg2][len].unwrap();
        }

        let flag = if len == 1 {
            &s1[beg1..end1] == &s2[beg2..end2]
        } else {
            (1..len).any(|i| {
                (check(s1, beg1, beg1 + i, s2, beg2, beg2 + i, records)
                    && check(s1, beg1 + i, end1, s2, beg2 + i, end2, records))
                    || (check(s1, beg1, beg1 + i, s2, end2 - i, end2, records)
                        && check(s1, beg1 + i, end1, s2, beg2, end2 - i, records))
            })
        };

        records[beg1][beg2][len] = Some(flag);
        flag
    }

    check(&s1, 0, n, &s2, 0, n, &mut records)
}

///p88
pub fn merge_2(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    if m == 0 && n == 0 {
        return;
    }

    let mut idx = (m + n - 1) as usize;
    let mut i = m - 1;
    let mut j = n - 1;

    while i >= 0 && j >= 0 {
        if nums1[i as usize] < nums2[j as usize] {
            nums1[idx] = nums2[j as usize];
            j -= 1;
        } else {
            nums1[idx] = nums1[i as usize];
            i -= 1;
        }
        idx -= 1;
    }

    while j >= 0 {
        nums1[idx] = nums2[j as usize];
        idx -= 1;
        j -= 1;
    }
}

///p89
pub fn gray_code(n: i32) -> Vec<i32> {
    let mut ans: Vec<i32> = vec![0; 1 << n];
    for i in 0..ans.len() {
        ans[i] = ((i >> 1) ^ i) as i32;
    }
    return ans;
}

///p90
/// Input: nums = [1,2,2]
/// Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
pub fn subsets_with_dup(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut nums = nums;
    nums.sort();
    let mut ans: Vec<Vec<i32>> = vec![];
    let mut cur: Vec<i32> = vec![];
    fn dfs(option: &Vec<i32>, p: usize, chosen: bool, cur: &mut Vec<i32>, ans: &mut Vec<Vec<i32>>) {
        if p == option.len() {
            ans.push(cur.clone());
            return;
        }
        if chosen || (p > 0 && option[p - 1] != option[p]) {
            cur.push(option[p]);
            dfs(option, p + 1, true, cur, ans);
            cur.pop();
        }
        dfs(option, p + 1, false, cur, ans);
    }
    dfs(&nums, 0, true, &mut cur, &mut ans);
    return ans;
}
