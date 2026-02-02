use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use crate::common::TreeNode;

/// 1301
pub fn paths_with_max_score(board: Vec<String>) -> Vec<i32> {
    let board = board
        .iter()
        .map(|s| s.chars().collect::<Vec<char>>())
        .collect::<Vec<Vec<char>>>();
    let mut f = board
        .iter()
        .map(|s| {
            s.iter()
                .map(|&c| match c {
                    'X' => (-1, -1),
                    'E' => (0, 1),
                    'S' => (0, 1),
                    _ => (c.to_digit(10).unwrap() as i32, 0),
                })
                .collect::<Vec<(i32, i32)>>()
        })
        .collect::<Vec<Vec<(i32, i32)>>>();

    for i in (0..board.len()).rev() {
        for j in (0..board[i].len()).rev() {
            if f[i][j].0 == -1 || (i == board.len() - 1 && j == board[i].len() - 1) {
                continue;
            } else if i == board.len() - 1 && j < board[i].len() - 1 {
                f[i][j] = (f[i][j].0 + f[i][j + 1].0, f[i][j + 1].1);
            } else if j == board[i].len() - 1 && i < board.len() - 1 {
                f[i][j] = (f[i][j].0 + f[i + 1][j].0, f[i + 1][j].1);
            } else {
                let v = Vec::<(i32, i32)>::from([f[i + 1][j + 1], f[i + 1][j], f[i][j + 1]]);
                let max = match v.iter().filter(|&&x| x.1 != -1).max() {
                    Some(x) => x,
                    None => &(-1, -1),
                };
                if max.0 == -1 {
                    f[i][j] = (-1, -1);
                } else {
                    let max_count = v.iter().filter(|&x| x.0 == max.0).map(|x| x.1).sum::<i32>();
                    f[i][j] = (max.0 + f[i][j].0, max_count % 1000000007);
                }
            }
        }
    }
    match f[0][0].0 {
        -1 => vec![0, 0],
        _ => vec![f[0][0].0, f[0][0].1],
    }
}

/// 1302
pub fn deepest_leaves_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn bfs(nodes: &mut Vec<Option<Rc<RefCell<TreeNode>>>>, sum: &mut i32) {
        let size = nodes.len();
        *sum = (0..size).into_iter().fold(0, |mut acc, _| {
            if let Some(node) = nodes.remove(0) {
                acc = acc + node.borrow().val;
                if node.borrow().left.is_some() {
                    nodes.push(node.borrow().left.clone());
                }
                if node.borrow().right.is_some() {
                    nodes.push(node.borrow().right.clone());
                }
            }
            acc
        });
    }
    let (mut nodes, mut sum) = (vec![root], 0);
    while nodes.len() != 0 {
        bfs(&mut nodes, &mut sum);
    }
    sum
}

/// 1304
pub fn sum_zero(n: i32) -> Vec<i32> {
    (-n / 2..=n / 2)
        .filter(|&i| if n % 2 == 0 { i != 0 } else { true })
        .collect()
}

/// 1305
pub fn get_all_elements(
    root1: Option<Rc<RefCell<TreeNode>>>,
    root2: Option<Rc<RefCell<TreeNode>>>,
) -> Vec<i32> {
    fn in_order(root: Option<Rc<RefCell<TreeNode>>>, vals: &mut VecDeque<i32>) {
        if let Some(node) = root {
            in_order(node.borrow().left.clone(), vals);
            vals.push_back(node.borrow().val);
            in_order(node.borrow().right.clone(), vals);
        }
    }
    let (mut v1, mut v2) = (VecDeque::new(), VecDeque::new());
    in_order(root1, &mut v1);
    in_order(root2, &mut v2);
    let mut ans = vec![];
    while !v1.is_empty() || !v2.is_empty() {
        let (num1, num2) = (
            if v1.is_empty() {
                i32::MAX
            } else {
                *v1.front().unwrap()
            },
            if v2.is_empty() {
                i32::MAX
            } else {
                *v2.front().unwrap()
            },
        );
        ans.push(if num1 < num2 {
            v1.pop_front().unwrap()
        } else {
            v2.pop_front().unwrap()
        })
    }
    ans
}

/// 1306
pub fn can_reach(arr: Vec<i32>, start: i32) -> bool {
    let mut next = VecDeque::new();
    next.push_back(start as usize);
    let (mut reach, mut visited) = (false, vec![false; arr.len()]);
    while next.len() != 0 {
        let index = next.pop_front().unwrap() as usize;
        if arr[index] == 0 {
            reach = true;
            break;
        }
        let dis = arr[index] as usize;
        visited[index] = true;
        if index + dis < arr.len() && visited[index + dis] != true {
            if arr[index + dis] == 0 {
                reach = true;
                break;
            }
            next.push_back(index + dis);
        }
        if index - dis < arr.len() && visited[index - dis] != true {
            if arr[index - dis] == 0 {
                reach = true;
                break;
            }
            next.push_back(index - dis);
        }
    }
    reach
}

/// 1309
pub fn freq_alphabets(s: String) -> String {
    let chars = s.chars().collect::<Vec<char>>();
    let mut s = "".to_string();
    let mut i = 0;
    while i < chars.len() {
        if i + 2 < chars.len() && chars[i + 2] == '#' {
            s.push(
                ((chars[i].to_digit(10).unwrap() - 0) as u8 * 10
                    + (chars[i + 1].to_digit(10).unwrap() - 1) as u8
                    + b'a') as char,
            );
            i = i + 3;
        } else {
            s.push(((chars[i].to_digit(10).unwrap() - 1) as u8 + b'a') as char);
            i = i + 1;
        }
    }
    s
}

/// 1310
pub fn xor_queries(arr: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    let n = arr.len();
    let mut prefix = vec![0; n + 1];

    for i in 0..n {
        prefix[i + 1] = prefix[i] ^ arr[i];
    }

    queries
        .iter()
        .map(|v| prefix[v[0] as usize] ^ prefix[v[1] as usize + 1])
        .collect::<Vec<_>>()
}

/// 1311
pub fn watched_videos_by_friends(
    watched_videos: Vec<Vec<String>>,
    friends: Vec<Vec<i32>>,
    id: i32,
    level: i32,
) -> Vec<String> {
    use std::collections::{HashMap, HashSet, VecDeque};

    let mut q = VecDeque::new();
    let mut state = HashSet::new();
    let mut map = HashMap::new();
    q.push_back((id as usize, 0));
    while !q.is_empty() {
        let (top_id, top_deep) = q.pop_front().unwrap();
        if state.contains(&top_id) {
            continue;
        }
        state.insert(top_id);
        if top_deep == level {
            let watched_vec = &watched_videos[top_id];
            for s in watched_vec {
                map.entry(s).and_modify(|cnt| *cnt += 1).or_insert(0);
            }
            continue;
        }
        let top_friends = &friends[top_id];
        for f_id in top_friends {
            if top_deep + 1 <= level {
                q.push_back((*f_id as usize, top_deep + 1));
            }
        }
    }
    let mut vec = Vec::with_capacity(map.len());
    for item in map {
        vec.push(item);
    }

    vec.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap()
            .then(a.0.partial_cmp(b.0).unwrap())
    });
    let mut ans = Vec::new();
    for (s, _) in vec {
        ans.push(String::from(s));
    }
    ans
}
